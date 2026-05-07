"""
C_prototypical.py
=================

此版本聚焦兩條明確流程：
1) classification：CCLE 5-fold 訓練，固定啟用 proto refinement 對 TCGA target 做 adaptation。
2) regression：
   - Stage 1 使用 ResponseRegressor 預測 continuous neg_log2_auc。
   - CCLE validation / CCLE independent test 仍計算 regression metrics：
     MAE, RMSE, R2, Pearson, Spearman。
   - Prototypical refinement 固定啟用：
     將 neg_log2_auc >= -log2(0.5)=1.0 轉成 binary responder label，
     再套用原始 DAPL-style projector / relation model / contrastive learning /
     pseudo-label retraining。
   - TCGA eval 使用 predicted neg_log2_auc >= 1.0 轉成 binary prediction 後，
     計算 AUC, AUPR, Accuracy, F1, Precision, Recall。

輸入 latent 預設使用：
- `--pretrain_dir`（遞迴搜尋 `CCLE_latent_representation.pkl` / `TCGA_latent_representation.pkl`）
- `--drug_latent_pkl`


輸出包含：
- 每 fold learning curve 與最佳模型
- CCLE fold-level test metrics + mean/std
- TCGA fold-level eval metrics + mean/std
- CCLE/TCGA 每筆樣本 prediction
"""

import argparse
import math
import os
import pickle
import random
from collections import Counter
from copy import deepcopy
from itertools import chain

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from tools.dataprocess import cat_tensor_with_drug
from tools.model import (
    Classify,
    Classifydim2,
    projector,
    projector_decoder,
    relation_model,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# AUC original scale:
#   AUC <= 0.5 indicates stronger response / responder.
# neg_log2_auc scale:
#   neg_log2_auc = -log2(AUC)
#   -log2(0.5) = 1.0
#
# For --task_type regression:
#   - proto uses binary labels generated from continuous neg_log2_auc
#   - TCGA eval converts predicted neg_log2_auc to binary prediction
#
# responder = 1 if neg_log2_auc >= 1.0
REGRESSION_BINARY_THRESHOLD = -math.log2(0.5)
DEFAULT_PRISM_CCLE_GT = "data/TCGA/temdata/cclelabel_PRISM_format.csv"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_message(log_path, message):
    print(message)
    with open(log_path, "a") as handle:
        handle.write(str(message) + "\n")


def normalize_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def tcga_patient_key(sample_id):
    sample_id = str(sample_id)
    parts = sample_id.split("-")
    if sample_id.startswith("TCGA-") and len(parts) >= 3:
        return "-".join(parts[:3])
    return sample_id


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def discover_all_pretrain_latent_pairs(pretrain_dir):
    if not pretrain_dir or not os.path.isdir(pretrain_dir):
        raise ValueError(f"--pretrain_dir does not exist or is not a directory: {pretrain_dir}")
    pairs = []
    for root, _, files in os.walk(pretrain_dir):
        if "CCLE_latent_representation.pkl" in files and "TCGA_latent_representation.pkl" in files:
            pairs.append(
                {
                    "pretrain_combo": os.path.relpath(root, pretrain_dir),
                    "combo_dir": root,
                    "ccle_latent_pkl": os.path.join(root, "CCLE_latent_representation.pkl"),
                    "tcga_latent_pkl": os.path.join(root, "TCGA_latent_representation.pkl"),
                }
            )
    if not pairs:
        raise ValueError(
            "No valid pretrain latent pair found under --pretrain_dir "
            "(needs CCLE_latent_representation.pkl and TCGA_latent_representation.pkl in same folder)"
        )
    pairs = sorted(pairs, key=lambda x: x["pretrain_combo"])
    return pairs


def resolve_tcga_eval_input_from_gt(gt_input_path):
    gt_input_path = str(gt_input_path)
    if "cclelabel_PRISM_format.csv" in gt_input_path:
        return gt_input_path.replace("cclelabel_PRISM_format.csv", "tcgalabel_PRISM_format.csv")
    base_dir = os.path.dirname(gt_input_path)
    return os.path.join(base_dir, "tcgalabel_PRISM_format.csv")


def validate_drug_latent_dict(drug_latent):
    validated = {}
    lengths = set()
    for drug_id, values in drug_latent.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        validated[str(drug_id)] = arr.tolist()
        lengths.add(arr.size)
    if len(lengths) != 1:
        raise ValueError(
            "Inconsistent drug latent dimensions detected in --drug_latent_pkl. "
            "Please fix the previous step output instead of using compatibility conversion. "
            f"Observed dimensions: {sorted(lengths)[:10]}"
        )
    return validated


def normalize_tcga_latent_dict(tcga_latent):
    """Normalize TCGA latent keys to patient-level ID and deduplicate by first occurrence."""
    normalized = {}
    for key, value in tcga_latent.items():
        pid = tcga_patient_key(key)
        if pid not in normalized:
            normalized[pid] = value
    return normalized


def load_drug_name_map(path, drug_id_col, drug_name_col):
    df = pd.read_csv(path)
    missing = [col for col in [drug_id_col, drug_name_col] if col not in df.columns]
    if missing:
        raise ValueError(f"missing drug mapping columns in {path}: {missing}")
    tmp = df[[drug_id_col, drug_name_col]].dropna().copy()
    tmp["normalized_name"] = tmp[drug_name_col].map(normalize_name)
    mapping = {}
    ambiguous = []
    for normalized_name, group in tmp.groupby("normalized_name"):
        unique_ids = sorted(group[drug_id_col].astype(str).unique().tolist())
        if len(unique_ids) == 1:
            mapping[normalized_name] = unique_ids[0]
        else:
            ambiguous.append(
                {
                    "normalized_drug_name": normalized_name,
                    "candidate_broad_ids": "|".join(unique_ids),
                    "candidate_count": len(unique_ids),
                }
            )
    return mapping, ambiguous


def load_cancer_type_map(path, sample_id_col, cancer_type_col):
    df = pd.read_csv(path)
    missing = [col for col in [sample_id_col, cancer_type_col] if col not in df.columns]
    if missing:
        raise ValueError(f"missing cancer type columns in {path}: {missing}")
    df = df[[sample_id_col, cancer_type_col]].dropna()
    return dict(zip(df[sample_id_col].astype(str), df[cancer_type_col].astype(str)))


def prepare_prism(args):
    df = pd.read_csv(args.gt_input)
    required = [args.prism_sample_id_col, args.prism_drug_id_col]
    label_col = args.binary_label_col if args.task_type == "classification" else args.regression_label_col
    required.append(label_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"PRISM input missing required columns: {missing}")
    ccle_cancer_type = load_cancer_type_map(args.ccle_cancer_type_input, args.ccle_cancer_sample_id_col, args.ccle_cancer_type_col)
    out = pd.DataFrame(
        {
            "sample_id": df[args.prism_sample_id_col].astype(str),
            "drug_id": df[args.prism_drug_id_col].astype(str),
            "domain": "CCLE",
            "cancer_type": df[args.prism_sample_id_col].astype(str).map(ccle_cancer_type),
            "ground_truth": df[label_col],
            "original_drug_name": df[args.prism_drug_id_col].astype(str),
            "source_table": "PRISM",
        }
    )
    return out


def prepare_tcga_dapl(args):
    df = pd.read_csv(args.gt_input)
    required = [args.tcga_sample_id_col, args.tcga_drug_name_col, args.tcga_label_col, args.tcga_cancer_type_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"TCGA DAPL input missing required columns: {missing}")
    # TCGA eval always uses binary Label as ground truth.
    label_col = args.tcga_label_col
    drug_name_map, ambiguous = load_drug_name_map(args.drug_smiles_input, args.drug_id_col, args.drug_name_col)
    if ambiguous:
        pd.DataFrame(ambiguous).to_csv(
            os.path.join(args.output_dir, "ambiguous_drug_name_mapping.csv"), index=False
        )
    has_direct_id = args.tcga_drug_id_col in df.columns
    if has_direct_id:
        direct_raw = df[args.tcga_drug_id_col]
        direct_ids = direct_raw.astype(str)
        mapped_from_name = df[args.tcga_drug_name_col].map(
            lambda value: drug_name_map.get(normalize_name(value), np.nan)
        )
        # Prefer explicit broad_id / PubCHEM column, fallback to normalized drug-name mapping.
        drug_ids = np.where(
            direct_raw.notna() & (direct_ids.str.lower() != "nan") & (direct_ids.str.len() > 0),
            direct_ids,
            mapped_from_name,
        )
    else:
        drug_ids = df[args.tcga_drug_name_col].map(lambda value: drug_name_map.get(normalize_name(value), np.nan))
    out = pd.DataFrame(
        {
            "sample_id": df[args.tcga_sample_id_col].astype(str).map(tcga_patient_key),
            "drug_id": drug_ids,
            "domain": "TCGA",
            "cancer_type": df[args.tcga_cancer_type_col].astype(str),
            "ground_truth": df[label_col],
            "original_drug_name": df[args.tcga_drug_name_col].astype(str),
            "source_table": "TCGA_DAPL",
        }
    )
    return out


def prepare_gt(args):
    return prepare_prism(args)


def validate_ground_truth(df, task_type):
    df = df.copy()
    missing_gt = df["ground_truth"].isna()
    df = df.loc[~missing_gt].copy()
    if task_type == "classification":
        values = pd.to_numeric(df["ground_truth"], errors="coerce")
        invalid = values.isna() | ~values.isin([0, 1])
        if invalid.any():
            bad = df.loc[invalid, "ground_truth"].head(10).tolist()
            raise ValueError(f"classification labels must be 0/1; examples: {bad}")
        df["ground_truth"] = values.astype(np.float32)
    else:
        values = pd.to_numeric(df["ground_truth"], errors="coerce")
        df = df.loc[~values.isna()].copy()
        df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce").astype(np.float32)
    return df, int(missing_gt.sum())


def build_prefix_map(latent_dict):
    prefix_map = {}
    for key in latent_dict:
        prefix = tcga_patient_key(key)
        prefix_map.setdefault(prefix, str(key))
    return prefix_map


def find_sample_latent(row, ccle_latent, tcga_latent, tcga_prefix_map):
    sample_id = str(row["sample_id"])
    domain = str(row["domain"]).upper()
    if domain == "CCLE":
        return ccle_latent.get(sample_id), sample_id
    if domain == "TCGA":
        if sample_id in tcga_latent:
            return tcga_latent[sample_id], sample_id
        mapped_key = tcga_prefix_map.get(tcga_patient_key(sample_id))
        if mapped_key:
            return tcga_latent[mapped_key], mapped_key
    return None, None


def align_latents(gt_df, ccle_latent, tcga_latent, drug_latent, task_type, missing_gt_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tcga_prefix_map = build_prefix_map(tcga_latent)
    rows = []
    missing_samples = []
    missing_drugs = []
    tcga_mapping_rows = []
    for _, row in gt_df.iterrows():
        sample_vec, latent_sample_key = find_sample_latent(row, ccle_latent, tcga_latent, tcga_prefix_map)
        drug_id = str(row["drug_id"]) if not pd.isna(row["drug_id"]) else ""
        drug_vec = drug_latent.get(drug_id)
        if sample_vec is None:
            missing_samples.append(str(row["sample_id"]))
        if drug_vec is None:
            missing_drugs.append(drug_id or str(row["original_drug_name"]))
        if sample_vec is None or drug_vec is None:
            continue
        if str(row["domain"]).upper() == "TCGA":
            tcga_mapping_rows.append(
                {
                    "input_sample_id": str(row["sample_id"]),
                    "mapped_latent_sample_key": str(latent_sample_key),
                    "used_prefix_mapping": int(str(row["sample_id"]) != str(latent_sample_key)),
                }
            )
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "latent_sample_key": latent_sample_key,
                "drug_id": drug_id,
                "domain": str(row["domain"]).upper(),
                "cancer_type": str(row["cancer_type"]),
                "ground_truth": float(row["ground_truth"]),
                "original_drug_name": str(row["original_drug_name"]),
                "source_table": str(row["source_table"]),
                "feature": np.concatenate([np.asarray(sample_vec, dtype=np.float32), np.asarray(drug_vec, dtype=np.float32)]),
            }
        )
    aligned = pd.DataFrame(rows)
    if missing_samples:
        with open(os.path.join(output_dir, "missing_sample_ids.txt"), "w") as handle:
            handle.write("\n".join(sorted(set(missing_samples))))
    if missing_drugs:
        with open(os.path.join(output_dir, "missing_drug_ids.txt"), "w") as handle:
            handle.write("\n".join(sorted(set(missing_drugs))))
    if tcga_mapping_rows:
        pd.DataFrame(tcga_mapping_rows).drop_duplicates().to_csv(
            os.path.join(output_dir, "tcga_sample_id_mapping.csv"), index=False
        )
    report = {
        "raw_gt_total_rows": len(gt_df) + missing_gt_count,
        "gt_missing_excluded_rows": missing_gt_count,
        "rows_after_gt_validation": len(gt_df),
        "sample_latent_missing_rows": len(missing_samples),
        "drug_latent_missing_rows": len(missing_drugs),
        "both_latents_found_rows": len(aligned),
        "final_training_rows": len(aligned),
        "CCLE_rows": int((aligned["domain"] == "CCLE").sum()) if not aligned.empty else 0,
        "TCGA_rows": int((aligned["domain"] == "TCGA").sum()) if not aligned.empty else 0,
        "cancer_type_count": int(aligned["cancer_type"].nunique()) if not aligned.empty else 0,
        "drug_id_count": int(aligned["drug_id"].nunique()) if not aligned.empty else 0,
        "sample_id_count": int(aligned["sample_id"].nunique()) if not aligned.empty else 0,
        "task_type": task_type,
    }
    report_rows = [report]
    for domain_name in ["CCLE", "TCGA"]:
        raw_domain = gt_df[gt_df["domain"].astype(str).str.upper() == domain_name]
        usable_domain = aligned[aligned["domain"] == domain_name] if not aligned.empty else pd.DataFrame()
        denom = max(1, len(raw_domain))
        report_rows.append(
            {
                "raw_gt_total_rows": len(raw_domain),
                "gt_missing_excluded_rows": int(raw_domain["ground_truth"].isna().sum()) if "ground_truth" in raw_domain.columns else 0,
                "rows_after_gt_validation": len(raw_domain),
                "sample_latent_missing_rows": int(
                    sum(1 for s in missing_samples if s in set(raw_domain["sample_id"].astype(str).tolist()))
                ),
                "drug_latent_missing_rows": int(
                    sum(1 for d in missing_drugs if d in set(raw_domain["drug_id"].astype(str).tolist()))
                ),
                "both_latents_found_rows": len(usable_domain),
                "final_training_rows": len(usable_domain),
                "CCLE_rows": int((usable_domain["domain"] == "CCLE").sum()) if not usable_domain.empty else 0,
                "TCGA_rows": int((usable_domain["domain"] == "TCGA").sum()) if not usable_domain.empty else 0,
                "cancer_type_count": int(usable_domain["cancer_type"].nunique()) if not usable_domain.empty else 0,
                "drug_id_count": int(usable_domain["drug_id"].nunique()) if not usable_domain.empty else 0,
                "sample_id_count": int(usable_domain["sample_id"].nunique()) if not usable_domain.empty else 0,
                "task_type": task_type,
                "domain": domain_name,
                "usable_ratio": float(len(usable_domain) / denom),
            }
        )
    pd.DataFrame(report_rows).to_csv(os.path.join(output_dir, "data_alignment_report.csv"), index=False)
    if aligned.empty:
        raise ValueError("no aligned rows remain after matching sample and drug latent representations")
    return aligned


def make_stratify_labels(df, task_type, n_required, allow_label=True):
    candidates = []

    if allow_label:
        if task_type == "classification":
            split_label = df["ground_truth"].astype(int).astype(str)
            candidates.append(
                df["domain"].astype(str)
                + "|"
                + df["cancer_type"].astype(str)
                + "|"
                + split_label
            )

        elif task_type == "regression":
            split_label = (
                binarize_neg_log2_auc(df["ground_truth"].values)
                .astype(int)
                .astype(str)
            )
            candidates.append(
                df["domain"].astype(str)
                + "|"
                + df["cancer_type"].astype(str)
                + "|"
                + split_label
            )

    candidates.append(df["domain"].astype(str) + "|" + df["cancer_type"].astype(str))
    candidates.append(df["domain"].astype(str))
    for labels in candidates:
        counts = Counter(labels)
        if len(counts) > 1 and min(counts.values()) >= n_required:
            return labels
    return None


# ============================================================
# Prototypical Learning Functions (ported from prototypical.py)
# ============================================================

def label1_2(tensor):
    """Convert [n] binary labels to [n, 2] one-hot."""
    return torch.cat([1 - tensor.view(-1, 1), tensor.view(-1, 1)], dim=1)


@torch.no_grad()
def m_encoder_update(encoder, m_encoder, m=0.9):
    """Update momentum encoder parameters."""
    for param_q, param_k in zip(encoder.parameters(), m_encoder.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)


def proto_train_pm(Data, c2, encoder, decoder, pm_epochs=1000, pm_patience=20):
    """Train projector, decoder, and auxiliary classifier (from prototypical.py train_pm)."""
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    step1_train_models = [c2, encoder, decoder]
    step1_parameters = []
    for m in step1_train_models:
        m.train()
        step1_parameters.append(m.parameters())
    reloss = nn.MSELoss()
    closs = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(chain(*step1_parameters), lr=0.001)
    tolerance = 0
    best_eval_auc = 0
    best_encoder = deepcopy(encoder)
    best_decoder = deepcopy(decoder)
    best_c2 = deepcopy(c2)
    for epoch in range(pm_epochs):
        ccle_trainemb = source_train_data[0]
        ccle_lowemb = encoder(ccle_trainemb)
        ccle_trainemb_re = decoder(ccle_lowemb)
        recon_loss = reloss(ccle_trainemb_re, ccle_trainemb)
        c2_predict = c2(ccle_lowemb)
        c2_loss = closs(c2_predict, label1_2(source_train_data[1]))
        loss = recon_loss + c2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ccle_evalemb = source_test_data[0]
        ccle_evallowemb = encoder(ccle_evalemb)
        ccle_eval_predict = c2(ccle_evallowemb)
        ccle_eval_true = label1_2(source_test_data[1]).cpu().detach().numpy()
        try:
            epoch_eval_auc = roc_auc_score(ccle_eval_true, ccle_eval_predict.cpu().detach().numpy())
        except ValueError:
            epoch_eval_auc = 0.5
        if epoch_eval_auc > best_eval_auc:
            tolerance = 0
            best_eval_auc = epoch_eval_auc
            best_encoder = deepcopy(encoder)
            best_decoder = deepcopy(decoder)
            best_c2 = deepcopy(c2)
        else:
            tolerance += 1
        if tolerance >= pm_patience:
            break
    return best_encoder, best_decoder, best_c2


def proto_init_prototypes(Data, Mencoder):
    """Initialize class prototypes from momentum encoder (from prototypical.py init_prototypes)."""
    source_train_data = Data[0]
    cclez = source_train_data[0]
    ccle_compress = Mencoder(cclez)
    label_0_indices = torch.where(source_train_data[1] == 0)[0]
    label_1_indices = torch.where(source_train_data[1] == 1)[0]
    ccle_compress_0 = ccle_compress[label_0_indices]
    ccle_compress_1 = ccle_compress[label_1_indices]
    prototype0 = F.normalize(torch.mean(ccle_compress_0.clone(), dim=0), p=2, dim=0)
    prototype1 = F.normalize(torch.mean(ccle_compress_1.clone(), dim=0), p=2, dim=0)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)
    return prototypes


def proto_step_relationmodel(Data, prototypes, r_model, low_encoder,
                             relation_epochs=1000, relation_patience=10):
    """Train relation model (from prototypical.py step_relationmodel)."""
    source_train_data = Data[0]
    bceloss = nn.BCELoss()
    low_encoder.eval()
    r_model.train()
    optimizer = optim.AdamW(r_model.parameters(), lr=0.001)
    min_loss = float("inf")
    tolerance = 0
    best_r_model = deepcopy(r_model)
    for epoch in range(relation_epochs):
        label_0_indices = torch.where(source_train_data[1] == 0)[0]
        label_1_indices = torch.where(source_train_data[1] == 1)[0]
        lowtrain = low_encoder(source_train_data[0])
        cat_p0_data = cat_tensor_with_drug(lowtrain, prototypes[0, :])
        cat_p0_label = torch.zeros_like(source_train_data[1])
        cat_p0_label[label_0_indices] = 1
        cat_p0_predict = r_model(cat_p0_data)
        cat_p0_loss = bceloss(cat_p0_predict, cat_p0_label)
        cat_p1_data = cat_tensor_with_drug(lowtrain, prototypes[1, :])
        cat_p1_label = torch.zeros_like(source_train_data[1])
        cat_p1_label[label_1_indices] = 1
        cat_p1_predict = r_model(cat_p1_data)
        cat_p1_loss = bceloss(cat_p1_predict, cat_p1_label)
        loss = cat_p0_loss + cat_p1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            tolerance = 0
            best_r_model = deepcopy(r_model)
        else:
            tolerance += 1
        if tolerance >= relation_patience:
            break
    r_model.load_state_dict(best_r_model.state_dict())


def proto_train_contrastive(Data, step3optimizer, encoder, Mencoder, epoch,
                            c1, lin, r_model, args):
    """Contrastive learning step (from prototypical.py train_contrastive)."""
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    encoder.train()
    lin.train()
    c1.eval()
    Mencoder.eval()
    bcelogitsloss = nn.BCEWithLogitsLoss()
    closs = nn.CrossEntropyLoss()

    ccle_trainemb = source_train_data[0]
    ccle_lowemb = encoder(ccle_trainemb)

    source_train_data_ref = Data[0]
    cclez = source_train_data_ref[0]
    ccle_compress = Mencoder(cclez)
    label_0_indices = torch.where(source_train_data_ref[1] == 0)[0]
    label_1_indices = torch.where(source_train_data_ref[1] == 1)[0]
    ccle_compress_0 = lin(ccle_compress[label_0_indices])
    ccle_compress_1 = lin(ccle_compress[label_1_indices])
    mean_ccle_0 = torch.mean(ccle_compress_0.clone(), dim=0)
    mean_ccle_1 = torch.mean(ccle_compress_1.clone(), dim=0)
    prototype0 = mean_ccle_0 / torch.norm(mean_ccle_0, p=2)
    prototype1 = mean_ccle_1 / torch.norm(mean_ccle_1, p=2)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)

    tcgacat = target_data[0]
    tcga_compress = encoder(tcgacat)
    tcga_compress_m = Mencoder(tcgacat)
    tcga_mm = torch.mm(tcga_compress, tcga_compress_m.t())
    tcga_conlabel = torch.eye(tcga_mm.shape[0], tcga_mm.shape[0]).to(device)
    conloss_tcga = closs(tcga_mm, tcga_conlabel)

    temperature = 0.1
    logits_proto = torch.mm(tcga_compress, prototypes.t())
    logits_proto_raw = logits_proto.detach().clone()
    alpha = 0.5
    tcga_c1_predict = c1(tcgacat)

    soft_c2predict = torch.zeros(tcga_c1_predict.shape[0], 2).to(device)
    soft_c2predict[:, 1] = torch.sigmoid(tcga_c1_predict)
    soft_c2predict[:, 0] = 1 - torch.sigmoid(tcga_c1_predict)
    unlabel_soft = alpha * soft_c2predict + (1 - alpha) * F.softmax(logits_proto_raw, dim=1)
    yuzhi = min(args.pseudo_label_max, args.pseudo_label_start + epoch * args.pseudo_label_step)
    index1 = torch.where(unlabel_soft[:, 1] > yuzhi)[0]
    index0 = torch.where(unlabel_soft[:, 0] > yuzhi)[0]

    unlabel_soft_1 = F.softmax(logits_proto_raw, dim=1)
    index1 = list(set(torch.where(unlabel_soft_1[:, 1] > yuzhi)[0].tolist()) | set(index1.tolist()))
    index0 = list(set(torch.where(unlabel_soft_1[:, 0] > yuzhi)[0].tolist()) | set(index0.tolist()))

    rindex1_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype1))
    rindex1 = torch.where(rindex1_predict > yuzhi)[0]
    rindex0_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype0))
    rindex0 = torch.where(rindex0_predict > yuzhi)[0]

    index1 = torch.tensor(list(set(rindex1.tolist()) | set(index1)), device=device).long()
    index0 = torch.tensor(list(set(rindex0.tolist()) | set(index0)), device=device).long()

    if len(index0) == 0 or len(index1) == 0:
        loss = conloss_tcga
        step3optimizer.zero_grad()
        loss.backward(retain_graph=True)
        step3optimizer.step()
        m_encoder_update(encoder, Mencoder, m=0.9)
        return loss.item()

    fai0_c0 = torch.norm(torch.mean(tcga_compress[index0] - prototypes[0, :].unsqueeze(0), dim=0), p=2) / (
        len(index0) * math.log(len(index0) + 10.0) + 1e-7
    )
    fai0_c1 = torch.norm(torch.mean(tcga_compress[index0] - prototypes[1, :].unsqueeze(0), dim=0), p=2) / (
        len(index0) * math.log(len(index0) + 10.0) + 1e-7
    )
    fai1_c0 = torch.norm(torch.mean(tcga_compress[index1] - prototypes[0, :].unsqueeze(0), dim=0), p=2) / (
        len(index1) * math.log(len(index1) + 10.0) + 1e-7
    )
    fai1_c1 = torch.norm(torch.mean(tcga_compress[index1] - prototypes[1, :].unsqueeze(0), dim=0), p=2) / (
        len(index1) * math.log(len(index1) + 10.0) + 1e-7
    )
    posl = torch.cat(
        (
            torch.mm(tcga_compress[index0], prototypes[0, :].unsqueeze(0).t()) / fai0_c0,
            torch.mm(tcga_compress[index1], prototypes[1, :].unsqueeze(0).t()) / fai1_c1,
        ),
        dim=0,
    )
    negl = torch.cat(
        (
            torch.mm(tcga_compress[index0], prototypes[1, :].unsqueeze(0).t()) / fai0_c1,
            torch.mm(tcga_compress[index1], prototypes[0, :].unsqueeze(0).t()) / fai1_c0,
            torch.mm(prototypes[0, :].unsqueeze(0), prototypes[1, :].unsqueeze(0).t()) / temperature,
        ),
        dim=0,
    )
    l = torch.cat((posl, negl), dim=0).to(device)
    lb = torch.cat((torch.ones_like(posl), torch.zeros_like(negl)), dim=0).float().to(device)
    try:
        conloss_unlabeltopro = bcelogitsloss(l, lb)
    except Exception:
        conloss_unlabeltopro = 0
    loss = conloss_tcga + conloss_unlabeltopro
    step3optimizer.zero_grad()
    loss.backward(retain_graph=True)
    step3optimizer.step()
    m_encoder_update(encoder, Mencoder, m=0.9)
    return loss.item()


def proto_retrain_classifier(Data, step4optimizer, encoder, c1, epoch,
                             mencoder, lin, r_model, args):
    """Retrain classifier with pseudo-labels (from prototypical.py retrain_classifier)."""
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    bcelogitsloss = nn.BCEWithLogitsLoss()
    closs = nn.CrossEntropyLoss()
    c1.train()
    encoder.eval()
    lin.train()

    ccle_trainemb = source_train_data[0]
    ccle_c1_predict = c1(ccle_trainemb)

    source_train_data_ref = Data[0]
    cclez = source_train_data_ref[0]
    ccle_compress = mencoder(cclez)
    label_0_indices = torch.where(source_train_data_ref[1] == 0)[0]
    label_1_indices = torch.where(source_train_data_ref[1] == 1)[0]
    ccle_compress_0 = lin(ccle_compress[label_0_indices])
    ccle_compress_1 = lin(ccle_compress[label_1_indices])
    mean_ccle_0 = torch.mean(ccle_compress_0.clone(), dim=0)
    mean_ccle_1 = torch.mean(ccle_compress_1.clone(), dim=0)
    prototype0 = mean_ccle_0 / torch.norm(mean_ccle_0, p=2)
    prototype1 = mean_ccle_1 / torch.norm(mean_ccle_1, p=2)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)

    tcgacat = target_data[0]
    tcga_c1_predict = c1(tcgacat)
    tcga_compress = encoder(tcgacat)

    alpha = 0.5
    logits_proto = torch.mm(tcga_compress, prototypes.t())
    logits_proto_raw = logits_proto.detach().clone()

    soft_c2predict = torch.zeros(tcga_c1_predict.shape[0], 2).to(device)
    soft_c2predict[:, 1] = torch.sigmoid(tcga_c1_predict)
    soft_c2predict[:, 0] = 1 - torch.sigmoid(tcga_c1_predict)
    unlabel_soft = alpha * soft_c2predict + (1 - alpha) * F.softmax(logits_proto_raw, dim=1)

    yuzhi = min(args.pseudo_label_max, args.pseudo_label_start + epoch * args.pseudo_label_step)
    index1 = torch.where(unlabel_soft[:, 1] > yuzhi)[0]
    index0 = torch.where(unlabel_soft[:, 0] > yuzhi)[0]

    unlabel_soft_1 = F.softmax(logits_proto_raw, dim=1)
    index1 = list(set(torch.where(unlabel_soft_1[:, 1] > yuzhi)[0].tolist()) | set(index1.tolist()))
    index0 = list(set(torch.where(unlabel_soft_1[:, 0] > yuzhi)[0].tolist()) | set(index0.tolist()))

    rindex1_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype1))
    rindex1 = torch.where(rindex1_predict > yuzhi)[0]
    rindex0_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype0))
    rindex0 = torch.where(rindex0_predict > yuzhi)[0]

    index1 = torch.tensor(list(set(rindex1.tolist()) | set(index1)), device=device).long()
    index0 = torch.tensor(list(set(rindex0.tolist()) | set(index0)), device=device).long()
    correct_idx = torch.cat((index1, index0), dim=0)

    loss_cls_hard_ccle = bcelogitsloss(ccle_c1_predict, source_train_data[1])
    if len(correct_idx) != 0:
        p_label = torch.zeros(index1.shape[0] + index0.shape[0]).to(device)
        p_label[: index1.shape[0]] = 1
        loss_cls_soft_unlabel = bcelogitsloss(tcga_c1_predict[correct_idx], p_label)
        loss_cls = loss_cls_hard_ccle + loss_cls_soft_unlabel
    else:
        loss_cls = loss_cls_hard_ccle
    step4optimizer.zero_grad()
    loss_cls.backward()
    step4optimizer.step()
    return correct_idx


def train_step1_classifier(source_features, source_labels, eval_features,
                           eval_labels, input_dim, num_epochs, patience,
                           lr, log_path):
    """Train initial Classify model on source data (equivalent to fine_tune step 1)."""
    classifymodel = Classify(input_dim=input_dim).to(device)
    classification_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifymodel.parameters(), lr=lr)
    best_auc = 0
    tolerance = 0
    best_state = deepcopy(classifymodel.state_dict())
    classifymodel.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predict = classifymodel(source_features)
        loss = classification_loss(predict, source_labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            eval_pred = classifymodel(eval_features).cpu().detach().numpy()
            eval_true = eval_labels.cpu().detach().numpy()
            try:
                eval_auc = roc_auc_score(eval_true, eval_pred)
            except ValueError:
                eval_auc = 0.5
            if eval_auc >= best_auc:
                best_auc = eval_auc
                best_state = deepcopy(classifymodel.state_dict())
                tolerance = 0
            else:
                tolerance += 1
            if tolerance >= patience:
                break
    classifymodel.load_state_dict(best_state)
    return classifymodel


def prototypical_refinement(source_train_features, source_train_labels,
                            source_eval_features, source_eval_labels,
                            target_features, target_labels,
                            classifymodel, input_dim, args, log_path):
    """Full prototypical refinement (equivalent to p_fine_tune from prototypical.py).

    Returns:
        (refined_classifymodel, result_prediction_on_target)
    """
    proj_dim = args.proj_dim
    Data = (
        (source_train_features, source_train_labels),
        (source_eval_features, source_eval_labels),
        (target_features, target_labels),
    )
    source_train_data = Data[0]
    source_test_data = Data[1]
    target_data = Data[2]

    # Step 1: Train projector + decoder + auxiliary classifier
    lowencoder = projector(in_dim=input_dim, out_dim=proj_dim).to(device)
    lowdecoder = projector_decoder(in_dim=proj_dim, out_dim=input_dim).to(device)
    auxiliary_classifier = Classifydim2(input_dim=proj_dim).to(device)
    lowencoder, lowdecoder, auxiliary_classifier = proto_train_pm(
        Data=Data, c2=auxiliary_classifier, encoder=lowencoder, decoder=lowdecoder,
        pm_epochs=args.proto_pm_epochs, pm_patience=args.proto_patience,
    )
    log_message(log_path, "  proto: train_pm completed")

    # Step 2: Init momentum encoder and prototypes
    M_lowencoder = projector(in_dim=input_dim, out_dim=proj_dim).to(device)
    for param_q, param_k in zip(lowencoder.parameters(), M_lowencoder.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    prototypes = proto_init_prototypes(Data=Data, Mencoder=M_lowencoder)
    log_message(log_path, "  proto: prototypes initialized")

    # Step 3: Train relation model
    r_model = relation_model(indim=proj_dim * 2).to(device)
    proto_step_relationmodel(
        Data=Data, prototypes=prototypes, r_model=r_model,
        low_encoder=lowencoder,
        relation_epochs=args.proto_relation_epochs,
        relation_patience=args.proto_patience,
    )
    log_message(log_path, "  proto: relation model trained")

    # Step 4: Contrastive learning
    lin = nn.Linear(proj_dim, proj_dim).to(device)
    con_optimizer = optim.AdamW(
        chain(*[lowencoder.parameters(), lin.parameters(), r_model.parameters()]),
        lr=0.0001,
    )
    best_lowencoder = deepcopy(lowencoder)
    best_M_encoder = deepcopy(M_lowencoder)
    best_lin = deepcopy(lin)
    best_r_model = deepcopy(r_model)
    minloss = float("inf")
    tolerance = 0
    for epoch in range(args.proto_contrastive_epochs):
        epoch_loss = proto_train_contrastive(
            Data, con_optimizer, lowencoder, M_lowencoder, epoch,
            classifymodel, lin, r_model, args,
        )
        if minloss > epoch_loss:
            minloss = epoch_loss
            best_lowencoder = deepcopy(lowencoder)
            best_M_encoder = deepcopy(M_lowencoder)
            best_lin = deepcopy(lin)
            best_r_model = deepcopy(r_model)
            tolerance = 0
        else:
            tolerance += 1
        if tolerance >= args.proto_patience:
            break
    log_message(log_path, f"  proto: contrastive learning completed (epochs used: {epoch + 1})")

    # Step 5: Retrain classifier with pseudo-labels
    retrain_optimizer = optim.AdamW(
        chain(*[best_lin.parameters(), classifymodel.parameters()]),
        lr=0.0001,
    )
    best_eval_auc = 0
    best_c1_afterp = deepcopy(classifymodel)
    result_prediction = None
    for epoch in range(args.proto_retrain_epochs):
        idx = proto_retrain_classifier(
            Data, retrain_optimizer, best_lowencoder, classifymodel,
            epoch, best_M_encoder, best_lin, best_r_model, args,
        )
        with torch.no_grad():
            eval_y_pred = classifymodel(source_eval_features).cpu().detach().numpy()
            eval_y_true = source_eval_labels.cpu().detach().numpy()
            test_y_pred = classifymodel(target_features).cpu().detach().numpy()
            try:
                eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            except ValueError:
                eval_auc = 0.5
            if eval_auc >= best_eval_auc:
                best_eval_auc = eval_auc
                best_c1_afterp = deepcopy(classifymodel)
                result_prediction = test_y_pred
        if len(idx) == 0:
            break
    log_message(log_path, f"  proto: retrain completed (best eval AUC={best_eval_auc:.4f})")
    return best_c1_afterp, result_prediction


class ResponseRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def to_feature_matrix(df):
    return np.vstack(df["feature"].values).astype(np.float32)


def make_loader(x, y, batch_size, shuffle):
    tensor_x = torch.from_numpy(x.astype(np.float32))
    tensor_y = torch.from_numpy(y.astype(np.float32))
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict(model, x, batch_size):
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds)


def classification_metrics(y_true, logits_or_probs, from_logits=True, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_or_probs)) if from_logits else np.asarray(logits_or_probs)
    labels = (probs >= threshold).astype(int)
    metrics = {
        "AUC": np.nan,
        "AUPR": np.nan,
        "Accuracy": float(accuracy_score(y_true, labels)),
        "F1": float(f1_score(y_true, labels, zero_division=0)),
        "Precision": float(precision_score(y_true, labels, zero_division=0)),
        "Recall": float(recall_score(y_true, labels, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["AUC"] = float(roc_auc_score(y_true, probs))
        metrics["AUPR"] = float(average_precision_score(y_true, probs))
    return metrics


def regression_metrics(y_true, preds):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)
    pearson = stats.pearsonr(y_true, preds)[0] if len(y_true) > 1 and np.std(y_true) > 0 and np.std(preds) > 0 else np.nan
    spearman = stats.spearmanr(y_true, preds)[0] if len(y_true) > 1 else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
        "R2": float(r2_score(y_true, preds)) if len(y_true) > 1 else np.nan,
        "Pearson": float(pearson) if np.isfinite(pearson) else np.nan,
        "Spearman": float(spearman) if np.isfinite(spearman) else np.nan,
    }


def binarize_neg_log2_auc(values):
    """
    Convert continuous neg_log2_auc values to binary responder labels.

    Original AUC threshold:
        AUC <= 0.5 means sensitive / responder.

    Since:
        neg_log2_auc = -log2(AUC)
        -log2(0.5) = 1.0

    Therefore:
        responder = 1 if neg_log2_auc >= 1.0
        non-responder = 0 otherwise
    """
    values = np.asarray(values, dtype=float)
    return (values >= REGRESSION_BINARY_THRESHOLD).astype(np.float32)


def tcga_eval_classification_metrics(y_true_binary, pred_values, task_type):
    """
    TCGA eval always reports binary classification metrics.

    classification:
        pred_values are classification probabilities.
        Use threshold=0.5.

    regression:
        pred_values are continuous predicted neg_log2_auc.
        Convert to binary prediction using REGRESSION_BINARY_THRESHOLD = 1.0.
    """
    y_true_binary = np.asarray(y_true_binary).astype(int)
    pred_values = np.asarray(pred_values, dtype=float)

    if task_type == "classification":
        return classification_metrics(
            y_true=y_true_binary,
            logits_or_probs=pred_values,
            from_logits=False,
            threshold=0.5,
        )

    if task_type == "regression":
        pred_binary = (pred_values >= REGRESSION_BINARY_THRESHOLD).astype(int)
        metrics = {
            "AUC": np.nan,
            "AUPR": np.nan,
            "Accuracy": float(accuracy_score(y_true_binary, pred_binary)),
            "F1": float(f1_score(y_true_binary, pred_binary, zero_division=0)),
            "Precision": float(precision_score(y_true_binary, pred_binary, zero_division=0)),
            "Recall": float(recall_score(y_true_binary, pred_binary, zero_division=0)),
            "threshold": float(REGRESSION_BINARY_THRESHOLD),
            "tcga_eval_mode": "regression_prediction_thresholded_to_binary",
        }
        # This intentionally evaluates the thresholded binary prediction.
        # It is not a continuous-score AUC.
        if len(np.unique(y_true_binary)) > 1:
            metrics["AUC"] = float(roc_auc_score(y_true_binary, pred_binary))
            metrics["AUPR"] = float(average_precision_score(y_true_binary, pred_binary))
        return metrics

    raise ValueError(f"unsupported task_type for TCGA eval: {task_type}")


def train_one_fold(fold_id, ccle_train_df, ccle_val_df, tcga_proto_features,
                   input_dim, args, ft_param, output_dir, log_path):
    """Train one fold. Stage 1 uses CCLE only. Proto uses TCGA latent as target.

    Args:
        ccle_train_df: CCLE training data with 'feature' and 'ground_truth'.
        ccle_val_df: CCLE validation data.
        tcga_proto_features: np.ndarray of TCGA latent features for proto target (no labels).
        ft_param: dict with 'ftlr' and 'scheduler_flag'.
    """
    x_train = to_feature_matrix(ccle_train_df)
    y_train = ccle_train_df["ground_truth"].values.astype(np.float32)
    x_val = to_feature_matrix(ccle_val_df)
    y_val = ccle_val_df["ground_truth"].values.astype(np.float32)

    ftlr = ft_param["ftlr"]
    scheduler_flag = ft_param["scheduler_flag"]

    if args.task_type == "classification":
        model = Classify(input_dim=input_dim).to(device)
    else:
        model = ResponseRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ftlr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.BCEWithLogitsLoss() if args.task_type == "classification" else nn.L1Loss()
    loader = make_loader(x_train, y_train, args.train_batch_size, shuffle=True)
    best_state = deepcopy(model.state_dict())
    best_metric = -np.inf if args.task_type == "classification" else np.inf
    best_epoch = -1
    tolerance = 0
    curve_rows = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if scheduler_flag:
            scheduler.step()
        val_raw = predict(model, x_val, args.train_batch_size)
        train_loss = float(np.mean(losses))
        if args.task_type == "classification":
            val_loss = float(nn.BCEWithLogitsLoss()(torch.from_numpy(val_raw), torch.from_numpy(y_val)).item())
            val_metrics = classification_metrics(y_val, val_raw, from_logits=True)
            main_metric = val_metrics["AUC"] if np.isfinite(val_metrics["AUC"]) else val_metrics["AUPR"]
            improved = main_metric > best_metric
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "stage": "stage1_train",
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "val_metric": main_metric,
                "task_type": args.task_type,
                "best_flag": 0,
            }
            row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        else:
            val_loss = float(nn.L1Loss()(torch.from_numpy(val_raw), torch.from_numpy(y_val)).item())
            val_metrics = regression_metrics(y_val, val_raw)
            main_metric = val_metrics["MAE"]
            improved = main_metric < best_metric
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "stage": "stage1_train",
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "val_metric": main_metric,
                "task_type": args.task_type,
                "best_flag": 0,
            }
            row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        curve_rows.append(row)
        if improved:
            best_metric = main_metric
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            tolerance = 0
        else:
            tolerance += 1
        if tolerance >= args.patience:
            log_message(log_path, f"fold {fold_id} early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    metric_name = "auc" if args.task_type == "classification" else "mae"
    torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_{args.task_type}_fold{fold_id}_{metric_name}.pt"))
    curve_df = pd.DataFrame(curve_rows)
    if best_epoch >= 0 and not curve_df.empty:
        curve_df.loc[curve_df["epoch"] == best_epoch, "best_flag"] = 1
    curve_df.to_csv(os.path.join(output_dir, f"learning_curve_fold{fold_id}.csv"), index=False)
    plot_learning_curve(curve_df, os.path.join(output_dir, f"learning_curve_fold{fold_id}.png"), f"Fold {fold_id}")

    # --- Prototypical Refinement ---
    # Classification:
    #   use original binary labels.
    # Regression:
    #   keep ResponseRegressor as the main model, but run DAPL-style proto by
    #   converting continuous neg_log2_auc labels into binary labels:
    #   responder = 1 if neg_log2_auc >= REGRESSION_BINARY_THRESHOLD.
    proto_classifier = None
    has_tcga_proto = (
        tcga_proto_features is not None
        and len(tcga_proto_features) > 0
    )

    if has_tcga_proto:
        log_message(log_path, f"fold {fold_id}: starting prototypical refinement")
        ccle_train_x = torch.from_numpy(x_train).to(device)
        tcga_train_x = torch.from_numpy(tcga_proto_features.astype(np.float32)).to(device)
        ccle_val_x = torch.from_numpy(x_val).to(device)

        if args.task_type == "classification":
            proto_train_y_np = y_train.astype(np.float32)
            proto_val_y_np = y_val.astype(np.float32)
            proto_label_message = "classification binary labels"
        else:
            proto_train_y_np = binarize_neg_log2_auc(y_train)
            proto_val_y_np = binarize_neg_log2_auc(y_val)
            proto_label_message = (
                f"regression labels binarized by neg_log2_auc >= "
                f"{REGRESSION_BINARY_THRESHOLD}"
            )

        ccle_train_y = torch.from_numpy(proto_train_y_np.astype(np.float32)).to(device)
        ccle_val_y = torch.from_numpy(proto_val_y_np.astype(np.float32)).to(device)

        # Dummy labels for TCGA target.
        # TCGA true labels are not used during proto training.
        # Pseudo-labels are generated inside proto_train_contrastive / proto_retrain_classifier.
        tcga_dummy_y = torch.zeros(len(tcga_train_x), device=device)

        unique_labels = torch.unique(ccle_train_y)
        if len(unique_labels) >= 2:
            log_message(log_path, f"fold {fold_id}: starting proto with {proto_label_message}")
            step1_classifier = train_step1_classifier(
                ccle_train_x, ccle_train_y, ccle_val_x, ccle_val_y,
                input_dim, args.epochs, args.patience, ftlr, log_path,
            )
            log_message(log_path, f"fold {fold_id}: step1 classifier trained for proto")

            proto_classifier, _ = prototypical_refinement(
                source_train_features=ccle_train_x,
                source_train_labels=ccle_train_y,
                source_eval_features=ccle_val_x,
                source_eval_labels=ccle_val_y,
                target_features=tcga_train_x,
                target_labels=tcga_dummy_y,
                classifymodel=step1_classifier,
                input_dim=input_dim,
                args=args,
                log_path=log_path,
            )
            log_message(log_path, f"fold {fold_id}: prototypical refinement completed")
        else:
            log_message(log_path, f"fold {fold_id}: skipping proto (single class in proto labels)")
    else:
        log_message(log_path, f"fold {fold_id}: skipping proto (no TCGA proto features)")

    return model, curve_df, proto_classifier


def plot_learning_curve(curve_df, output_path, title):
    if curve_df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve_df["epoch"], curve_df["train_loss"], label="train_loss")
    plt.plot(curve_df["epoch"], curve_df["validation_loss"], label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_cv_splitter(train_val_df, args, log_path):
    stratify = make_stratify_labels(train_val_df, args.task_type, args.n_splits)
    if stratify is not None:
        return StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed), stratify
    log_message(log_path, "CV stratification labels are too sparse; fallback to KFold")
    return KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed), None


def add_fold_metric_summary(fold_metrics_df):
    metric_cols = []
    for col in fold_metrics_df.columns:
        if col in ["fold", "domain"]:
            continue
        numeric_vals = pd.to_numeric(fold_metrics_df[col], errors="coerce")
        if numeric_vals.notna().any():
            metric_cols.append(col)
    rows = [fold_metrics_df]
    summary = []
    for domain_name, subset in fold_metrics_df[fold_metrics_df["fold"].astype(str).str.startswith("fold")].groupby("domain"):
        for stat_name, func in [("mean", np.nanmean), ("std", np.nanstd)]:
            row = {"fold": stat_name, "domain": domain_name}
            for col in metric_cols:
                row[col] = func(pd.to_numeric(subset[col], errors="coerce").values.astype(float))
            summary.append(row)
    if summary:
        rows.append(pd.DataFrame(summary))
    return pd.concat(rows, ignore_index=True)


def aggregate_metric_mean_std(metrics_by_fold_df):
    metric_cols = []
    for col in metrics_by_fold_df.columns:
        if col == "fold":
            continue
        numeric_vals = pd.to_numeric(metrics_by_fold_df[col], errors="coerce")
        if numeric_vals.notna().any():
            metric_cols.append(col)
    rows = []
    for stat_name, func in [("mean", np.nanmean), ("std", np.nanstd)]:
        row = {"stat": stat_name}
        for col in metric_cols:
            row[col] = func(pd.to_numeric(metrics_by_fold_df[col], errors="coerce").values.astype(float))
        rows.append(row)
    return pd.DataFrame(rows)


def split_ccle_independent_test(ccle_df, args, log_path):
    if args.test_size == 0:
        log_message(log_path, "independent test split disabled (test_size=0)")
        return ccle_df.reset_index(drop=True), ccle_df.iloc[0:0].copy().reset_index(drop=True)

    sample_level = ccle_df.groupby("sample_id", as_index=False)["ground_truth"].mean()
    sample_ids = sample_level["sample_id"].astype(str).values
    stratify = None
    if args.task_type == "classification":
        stratify = sample_level["ground_truth"].round().astype(int).values
    elif args.task_type == "regression":
        stratify = binarize_neg_log2_auc(sample_level["ground_truth"].values).astype(int)
    try:
        cv_samples, test_samples = train_test_split(
            sample_ids,
            test_size=args.test_size,
            random_state=args.random_seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as err:
        log_message(log_path, f"independent test stratified split failed ({err}); fallback random split")
        cv_samples, test_samples = train_test_split(
            sample_ids,
            test_size=args.test_size,
            random_state=args.random_seed,
            shuffle=True,
            stratify=None,
        )
    cv_df = ccle_df[ccle_df["sample_id"].astype(str).isin(set(cv_samples))].reset_index(drop=True)
    test_df = ccle_df[ccle_df["sample_id"].astype(str).isin(set(test_samples))].reset_index(drop=True)
    return cv_df, test_df


def make_finetune_dir_name(ft_param):
    """Build fine-tune subdirectory name."""
    return "ftlr_" + str(ft_param["ftlr"]) + ",CosAL_" + str(ft_param["scheduler_flag"])


def prepare_tcga_eval_data(args, tcga_latent, drug_latent):
    """Load TCGA_drug_response_from_DAPL.csv for eval only. Returns DataFrame with features."""
    if args.tcga_eval_format == "prism_format":
        tcga_eval_input = resolve_tcga_eval_input_from_gt(args.gt_input)
        raw = pd.read_csv(tcga_eval_input)
        required = [args.tcga_sample_id_col, args.tcga_label_col, args.tcga_drug_id_col]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"TCGA PRISM-format eval input missing required columns: {missing}")
        cancer_type = raw[args.tcga_cancer_type_col].astype(str) if args.tcga_cancer_type_col in raw.columns else "TCGA_eval"
        tcga_eval_df = pd.DataFrame(
            {
                "sample_id": raw[args.tcga_sample_id_col].astype(str).map(tcga_patient_key),
                "drug_id": raw[args.tcga_drug_id_col].astype(str),
                "domain": "TCGA",
                "cancer_type": cancer_type,
                "ground_truth": pd.to_numeric(raw[args.tcga_label_col], errors="coerce"),
                "original_drug_name": raw[args.tcga_drug_id_col].astype(str),
                "source_table": "TCGA_PRISM_FORMAT",
            }
        )
    elif args.tcga_eval_format == "tcga_dapl":
        args_tcga = deepcopy(args)
        args_tcga.gt_input = args.tcga_eval_input
        tcga_eval_df = prepare_tcga_dapl(args_tcga)
        # For eval, always use binary Label column regardless of task_type
        tcga_eval_df["ground_truth"] = pd.to_numeric(
            pd.read_csv(args.tcga_eval_input)[args.tcga_label_col], errors="coerce"
        )
    else:
        raise ValueError(f"unsupported tcga_eval_format: {args.tcga_eval_format}")
    tcga_eval_df = tcga_eval_df.dropna(subset=["ground_truth", "drug_id"]).copy()
    tcga_eval_df["ground_truth"] = pd.to_numeric(tcga_eval_df["ground_truth"], errors="coerce")
    tcga_eval_df = tcga_eval_df.dropna(subset=["ground_truth"]).copy()
    invalid_label = ~tcga_eval_df["ground_truth"].isin([0, 1])
    if invalid_label.any():
        bad = tcga_eval_df.loc[invalid_label, "ground_truth"].head(10).tolist()
        raise ValueError(f"TCGA eval labels must be binary 0/1; examples: {bad}")
    tcga_eval_df["ground_truth"] = tcga_eval_df["ground_truth"].astype(np.float32)

    tcga_prefix_map = build_prefix_map(tcga_latent)
    rows = []
    for _, row in tcga_eval_df.iterrows():
        sample_vec, _ = find_sample_latent(row, {}, tcga_latent, tcga_prefix_map)
        drug_id = str(row["drug_id"]) if not pd.isna(row["drug_id"]) else ""
        drug_vec = drug_latent.get(drug_id)
        if sample_vec is None or drug_vec is None:
            continue
        rows.append({
            "sample_id": str(row["sample_id"]),
            "drug_id": drug_id,
            "domain": "TCGA",
            "cancer_type": str(row["cancer_type"]),
            "ground_truth": float(row["ground_truth"]),
            "original_drug_name": str(row["original_drug_name"]),
            "source_table": "TCGA_DAPL",
            "feature": np.concatenate([
                np.asarray(sample_vec, dtype=np.float32),
                np.asarray(drug_vec, dtype=np.float32),
            ]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_tcga_proto_features(tcga_eval_df):
    """Build proto target features from actual TCGA target set (same style as original prototypical.py target_data)."""
    if tcga_eval_df is None or tcga_eval_df.empty:
        return None
    return to_feature_matrix(tcga_eval_df)


def parse_args():
    parser = argparse.ArgumentParser("DAPL C_prototypical multi-drug response predictor (B flow)")
    parser.add_argument("--gt_input", default=DEFAULT_PRISM_CCLE_GT)
    parser.add_argument("--task_type", choices=["classification", "regression"], default="regression")
    parser.add_argument("--drug_id_col", default="broad_id")
    parser.add_argument("--binary_label_col", default="sensitivity")
    parser.add_argument("--regression_label_col", default="neg_log2_auc")
    parser.add_argument("--prism_sample_id_col", default="depmap_id")
    parser.add_argument("--prism_drug_id_col", default="broad_id")
    parser.add_argument("--tcga_eval_format", choices=["prism_format", "tcga_dapl"], default="prism_format")
    parser.add_argument("--tcga_eval_input", default="data_Winnie/TCGA_drug_response_from_DAPL.csv")
    parser.add_argument("--tcga_sample_id_col", default="Patient_id")
    parser.add_argument("--tcga_drug_name_col", default="drug_name")
    parser.add_argument("--tcga_drug_id_col", default="PubCHEM")
    parser.add_argument("--tcga_label_col", default="Label")
    parser.add_argument("--tcga_cancer_type_col", default="cancers")
    parser.add_argument("--ccle_cancer_type_input", default="data_Winnie/CCLE_cancer_type.csv")
    parser.add_argument("--ccle_cancer_sample_id_col", default="Sample_ID")
    parser.add_argument("--ccle_cancer_type_col", default="Cancer_type")
    parser.add_argument("--drug_smiles_input", default="data_Winnie/drug_smiles.csv")
    parser.add_argument("--drug_name_col", default="name")
    # Pretrain output directory (contains subdirectories for each param combo)
    parser.add_argument("--pretrain_dir", required=True)
    parser.add_argument("--pseudo_label_start", type=float, default=0.5)
    parser.add_argument("--pseudo_label_max", type=float, default=1.0)
    parser.add_argument("--pseudo_label_step", type=float, default=0.01)
    parser.add_argument("--drug_latent_pkl", required=True, help="B_precontext drug_latent_representation.pkl")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--output_dir", default="output_dir/C_prototypical")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size for Stage 1 predictor training")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    # Prototypical learning parameters
    parser.add_argument("--proj_dim", type=int, default=10)
    parser.add_argument("--proto_pm_epochs", type=int, default=1000)
    parser.add_argument("--proto_relation_epochs", type=int, default=1000)
    parser.add_argument("--proto_contrastive_epochs", type=int, default=100)
    parser.add_argument("--proto_retrain_epochs", type=int, default=100)
    parser.add_argument("--proto_patience", type=int, default=20)
    return parser.parse_args()


def run_one_finetune_combo(ft_param, ccle_cv_df, ccle_test_df, tcga_proto_features, tcga_eval_df,
                           input_dim, args, combo_output_dir, log_path):
    """Run 5-fold CV for one fine-tune parameter combo."""
    ft_dir_name = make_finetune_dir_name(ft_param)
    ft_dir = os.path.join(combo_output_dir, ft_dir_name)
    ensure_dir(ft_dir)
    log_message(log_path, f"  finetune combo: {ft_param} -> {ft_dir_name}")

    # --- CCLE CV split (independent test set already split before this function) ---
    ccle_df = ccle_cv_df.copy()
    stratify = make_stratify_labels(ccle_df, args.task_type, args.n_splits)
    splitter_cls = StratifiedKFold if stratify is not None else KFold
    splitter = splitter_cls(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    split_target = stratify if stratify is not None else ccle_df

    fold_metric_rows = []
    curve_dfs = []
    ccle_prediction_rows = []
    tcga_prediction_rows = []
    ccle_metric_rows = []
    tcga_metric_rows = []
    x_tcga_eval = to_feature_matrix(tcga_eval_df) if not tcga_eval_df.empty else None
    x_ccle_test = to_feature_matrix(ccle_test_df) if not ccle_test_df.empty else None

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(ccle_df, split_target), start=1):
        train_df = ccle_df.iloc[train_idx].reset_index(drop=True)
        val_df = ccle_df.iloc[val_idx].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            log_message(log_path, f"  fold {fold_id}: empty split, skip")
            continue

        model, curve_df, proto_classifier = train_one_fold(
            fold_id, train_df, val_df, tcga_proto_features,
            input_dim, args, ft_param, ft_dir, log_path,
        )
        curve_dfs.append(curve_df)

        x_val = to_feature_matrix(val_df)
        if args.task_type == "classification":
            val_raw = predict(model, x_val, args.train_batch_size)
            val_pred = 1.0 / (1.0 + np.exp(-val_raw))
        else:
            val_pred = predict(model, x_val, args.train_batch_size)
        val_metrics = (
            classification_metrics(val_df["ground_truth"].values, val_pred, from_logits=False)
            if args.task_type == "classification"
            else regression_metrics(val_df["ground_truth"].values, val_pred)
        )
        val_metric_row = {"fold": f"fold{fold_id}", "domain": "CCLE_val"}
        val_metric_row.update(val_metrics)
        fold_metric_rows.append(val_metric_row)

        # --- Predict on CCLE independent test set ---
        if x_ccle_test is not None and len(x_ccle_test) > 0:
            if args.task_type == "classification":
                test_raw = predict(model, x_ccle_test, args.train_batch_size)
                test_pred = 1.0 / (1.0 + np.exp(-test_raw))
            else:
                test_pred = predict(model, x_ccle_test, args.train_batch_size)
            ccle_metrics_fold = (
                classification_metrics(ccle_test_df["ground_truth"].values, test_pred, from_logits=False)
                if args.task_type == "classification"
                else regression_metrics(ccle_test_df["ground_truth"].values, test_pred)
            )
            ccle_metric_row = {"fold": f"fold{fold_id}", "domain": "CCLE_test"}
            ccle_metric_row.update(ccle_metrics_fold)
            ccle_metric_rows.append({"fold": f"fold{fold_id}", **ccle_metrics_fold})
            fold_metric_rows.append(ccle_metric_row)
            fold_ccle_out = ccle_test_df[
                ["sample_id", "drug_id", "cancer_type", "ground_truth", "original_drug_name"]
            ].copy()
            fold_ccle_out["domain"] = "CCLE"
            fold_ccle_out["fold"] = fold_id
            if args.task_type == "classification":
                fold_ccle_out["prediction_probability"] = test_pred
                fold_ccle_out["prediction_binary"] = (test_pred >= 0.5).astype(int)
                fold_ccle_out["prediction"] = test_pred
                fold_ccle_out["prediction_type"] = "classification_probability"
                fold_ccle_out["threshold"] = 0.5
            else:
                # Main CCLE regression output.
                # This remains continuous neg_log2_auc and is evaluated with regression metrics.
                fold_ccle_out["prediction_neg_log2_auc"] = test_pred
                fold_ccle_out["prediction"] = test_pred
                fold_ccle_out["prediction_type"] = "regression_neg_log2_auc"
                # Optional reference columns only.
                # These are not used for CCLE main regression metrics.
                fold_ccle_out["threshold_binary_label"] = (
                    ccle_test_df["ground_truth"].values >= REGRESSION_BINARY_THRESHOLD
                ).astype(int)
                fold_ccle_out["threshold_binary_prediction"] = (
                    test_pred >= REGRESSION_BINARY_THRESHOLD
                ).astype(int)
                fold_ccle_out["threshold"] = float(REGRESSION_BINARY_THRESHOLD)
            ccle_prediction_rows.append(fold_ccle_out)

        # --- Predict on TCGA eval set ---
        if x_tcga_eval is not None and len(x_tcga_eval) > 0:
            tcga_proto_prob = None
            if args.task_type == "classification" and proto_classifier is not None:
                proto_classifier.eval()
                with torch.no_grad():
                    xt = torch.from_numpy(x_tcga_eval.astype(np.float32)).to(device)
                    raw = proto_classifier(xt).cpu().numpy()
                tcga_pred = 1.0 / (1.0 + np.exp(-raw))
            elif args.task_type == "classification":
                raw = predict(model, x_tcga_eval, args.train_batch_size)
                tcga_pred = 1.0 / (1.0 + np.exp(-raw))
            else:
                # Main regression prediction: continuous neg_log2_auc.
                # TCGA eval converts this value to binary using REGRESSION_BINARY_THRESHOLD.
                tcga_pred = predict(model, x_tcga_eval, args.train_batch_size)
                # Auxiliary proto classifier prediction.
                # This is not the main regression output, but records whether the
                # thresholded proto branch improves TCGA binary classification.
                if proto_classifier is not None:
                    proto_classifier.eval()
                    with torch.no_grad():
                        xt = torch.from_numpy(x_tcga_eval.astype(np.float32)).to(device)
                        proto_raw = proto_classifier(xt).cpu().numpy()
                    tcga_proto_prob = 1.0 / (1.0 + np.exp(-proto_raw))
            tcga_metrics_fold = tcga_eval_classification_metrics(
                y_true_binary=tcga_eval_df["ground_truth"].values,
                pred_values=tcga_pred,
                task_type=args.task_type,
            )
            if args.task_type == "regression" and tcga_proto_prob is not None:
                proto_metrics = classification_metrics(
                    y_true=tcga_eval_df["ground_truth"].values,
                    logits_or_probs=tcga_proto_prob,
                    from_logits=False,
                    threshold=0.5,
                )
                for key, value in proto_metrics.items():
                    tcga_metrics_fold[f"proto_classifier_{key}"] = value
            tcga_metric_row = {"fold": f"fold{fold_id}", "domain": "TCGA_eval"}
            tcga_metric_row.update(tcga_metrics_fold)
            tcga_metric_rows.append({"fold": f"fold{fold_id}", **tcga_metrics_fold})
            fold_metric_rows.append(tcga_metric_row)
            fold_tcga_out = tcga_eval_df[
                ["sample_id", "drug_id", "cancer_type", "ground_truth", "original_drug_name"]
            ].copy()
            fold_tcga_out["domain"] = "TCGA"
            fold_tcga_out["fold"] = fold_id
            if args.task_type == "classification":
                fold_tcga_out["prediction_probability"] = tcga_pred
                fold_tcga_out["prediction_binary"] = (tcga_pred >= 0.5).astype(int)
                fold_tcga_out["prediction"] = tcga_pred
                fold_tcga_out["prediction_type"] = "classification_probability"
                fold_tcga_out["threshold"] = 0.5
            else:
                fold_tcga_out["prediction_neg_log2_auc"] = tcga_pred
                fold_tcga_out["prediction_binary"] = (
                    tcga_pred >= REGRESSION_BINARY_THRESHOLD
                ).astype(int)
                fold_tcga_out["prediction"] = tcga_pred
                fold_tcga_out["prediction_type"] = "regression_neg_log2_auc_thresholded"
                fold_tcga_out["threshold"] = float(REGRESSION_BINARY_THRESHOLD)
                if tcga_proto_prob is not None:
                    fold_tcga_out["proto_prediction_probability"] = tcga_proto_prob
                    fold_tcga_out["proto_prediction_binary"] = (tcga_proto_prob >= 0.5).astype(int)
            tcga_prediction_rows.append(fold_tcga_out)

    # --- Save learning curves ---
    if curve_dfs:
        all_curves = pd.concat(curve_dfs, ignore_index=True)
        all_curves.to_csv(os.path.join(ft_dir, "learning_curve_all_folds.csv"), index=False)
        plot_learning_curve(
            all_curves.groupby("epoch", as_index=False)[["train_loss", "validation_loss"]].mean(),
            os.path.join(ft_dir, "learning_curve_all_folds.png"), "All folds",
        )

    combo_result = {
        "ft_dir": ft_dir,
        "ft_dir_name": ft_dir_name,
        "has_tcga_eval": False,
    }

    if ccle_prediction_rows:
        ccle_pred_all = pd.concat(ccle_prediction_rows, ignore_index=True)
        ccle_pred_all.to_csv(os.path.join(ft_dir, "ccle_test_predictions.csv"), index=False)
    if tcga_prediction_rows:
        tcga_pred_all = pd.concat(tcga_prediction_rows, ignore_index=True)
        tcga_pred_all.to_csv(os.path.join(ft_dir, "tcga_eval_predictions.csv"), index=False)

    if tcga_metric_rows:
        tcga_by_fold_df = pd.DataFrame(tcga_metric_rows)
        tcga_by_fold_df.to_csv(os.path.join(ft_dir, "tcga_eval_metrics_by_fold.csv"), index=False)
        tcga_summary_df = aggregate_metric_mean_std(tcga_by_fold_df)
        tcga_summary_df.to_csv(os.path.join(ft_dir, "tcga_eval_metrics.csv"), index=False)
        mean_row = tcga_summary_df[tcga_summary_df["stat"] == "mean"]
        if not mean_row.empty:
            metric_cols = [col for col in mean_row.columns if col != "stat"]
            for col in metric_cols:
                combo_result[col] = float(mean_row.iloc[0][col])
        combo_result["has_tcga_eval"] = True

    if ccle_metric_rows:
        ccle_by_fold_df = pd.DataFrame(ccle_metric_rows)
        ccle_by_fold_df.to_csv(os.path.join(ft_dir, "ccle_test_metrics_by_fold.csv"), index=False)
        ccle_summary_df = aggregate_metric_mean_std(ccle_by_fold_df)
        ccle_summary_df.to_csv(os.path.join(ft_dir, "ccle_test_metrics.csv"), index=False)

        # Add CCLE test mean metrics into combo_result so regression can select
        # the best combo by CCLE_test_MAE.
        mean_row = ccle_summary_df[ccle_summary_df["stat"] == "mean"]
        if not mean_row.empty:
            for col in mean_row.columns:
                if col == "stat":
                    continue
                value = mean_row.iloc[0][col]
                if pd.notna(value):
                    combo_result[f"CCLE_test_{col}"] = float(value)

    if fold_metric_rows:
        fold_metrics_df = pd.DataFrame(fold_metric_rows)
        fold_metrics_df = add_fold_metric_summary(fold_metrics_df)
        fold_metrics_df.to_csv(os.path.join(ft_dir, "fold_level_metrics.csv"), index=False)

    log_message(log_path, f"  finetune combo {ft_dir_name} completed")
    return combo_result


def choose_primary_metric(task_type):
    if task_type == "classification":
        # Classification combo selection keeps using TCGA_eval AUC.
        return "AUC", True

    # Regression combo selection uses CCLE independent test MAE.
    # Lower MAE is better.
    return "CCLE_test_MAE", False


def main():
    args = parse_args()
    # Fixed defaults by design for this workflow.
    args.n_splits = 5
    set_seed(args.random_seed)
    ensure_dir(args.output_dir)
    log_path = os.path.join(args.output_dir, "C_training_log.txt")
    log_message(log_path, f"device={device}")
    log_message(log_path, f"task_type={args.task_type}")
    log_message(
        log_path,
        f"REGRESSION_BINARY_THRESHOLD (-log2(0.5)): {REGRESSION_BINARY_THRESHOLD}",
    )
    log_message(
        log_path,
        "Regression mode: Stage 1 remains continuous regression; "
        "proto uses thresholded binary labels from neg_log2_auc.",
    )

    # --- Load drug latent (shared across all combos) ---
    drug_latent = validate_drug_latent_dict(load_pickle(args.drug_latent_pkl))
    log_message(log_path, f"drug_latent: {len(drug_latent)} drugs")

    pretrain_pairs = discover_all_pretrain_latent_pairs(args.pretrain_dir)
    log_message(log_path, f"discovered pretrain latent pairs: {len(pretrain_pairs)}")

    all_combo_results = []
    for idx, pair in enumerate(pretrain_pairs, start=1):
        pt_name = pair["pretrain_combo"]
        ccle_latent_pkl = pair["ccle_latent_pkl"]
        tcga_latent_pkl = pair["tcga_latent_pkl"]
        log_message(log_path, f"\n[{idx}/{len(pretrain_pairs)}] pretrain_combo={pt_name}")
        log_message(log_path, f"ccle_latent_pkl={ccle_latent_pkl}")
        log_message(log_path, f"tcga_latent_pkl={tcga_latent_pkl}")

        ccle_latent = load_pickle(ccle_latent_pkl)
        tcga_latent = normalize_tcga_latent_dict(load_pickle(tcga_latent_pkl))

        ccle_gt_df = prepare_gt(args)
        ccle_gt_df, missing_gt = validate_ground_truth(ccle_gt_df, args.task_type)

        combo_output_dir = os.path.join(args.output_dir, pt_name)
        ensure_dir(combo_output_dir)
        ccle_aligned = align_latents(
            ccle_gt_df, ccle_latent, tcga_latent, drug_latent,
            args.task_type, missing_gt,
            combo_output_dir,
        )
        ccle_aligned = ccle_aligned[ccle_aligned["domain"] == "CCLE"].reset_index(drop=True)
        if ccle_aligned.empty:
            log_message(log_path, f"SKIP {pt_name}: no aligned CCLE rows")
            continue
        ccle_cv_df, ccle_test_df = split_ccle_independent_test(ccle_aligned, args, log_path)
        log_message(
            log_path,
            f"{pt_name} CCLE split: cv_rows={len(ccle_cv_df)}, test_rows={len(ccle_test_df)}, test_size={args.test_size}",
        )
        if ccle_cv_df.empty:
            log_message(log_path, f"SKIP {pt_name}: empty CV split")
            continue
        input_dim = len(ccle_cv_df.iloc[0]["feature"])

        tcga_eval_df = prepare_tcga_eval_data(args, tcga_latent, drug_latent)
        tcga_proto_features = build_tcga_proto_features(tcga_eval_df)

        # Grid search fine-tune settings: ftlr x scheduler_flag
        finetune_grid = [
            {"ftlr": 0.01, "scheduler_flag": True},
            {"ftlr": 0.01, "scheduler_flag": False},
            {"ftlr": 0.001, "scheduler_flag": True},
            {"ftlr": 0.001, "scheduler_flag": False},
        ]
        for ft_param in finetune_grid:
            combo_result = run_one_finetune_combo(
                ft_param, ccle_cv_df, ccle_test_df, tcga_proto_features, tcga_eval_df,
                input_dim, args, combo_output_dir, log_path,
            )
            combo_result.update(
                {
                    "pretrain_combo": pt_name,
                    "combo_dir": combo_output_dir,
                    "ccle_latent_pkl": ccle_latent_pkl,
                    "tcga_latent_pkl": tcga_latent_pkl,
                    "ftlr": ft_param["ftlr"],
                    "scheduler_flag": ft_param["scheduler_flag"],
                }
            )
            all_combo_results.append(combo_result)

    if all_combo_results:
        primary_metric, higher_is_better = choose_primary_metric(args.task_type)
        log_message(
            log_path,
            f"best combo selection metric={primary_metric}, higher_is_better={higher_is_better}",
        )

        result_df = pd.DataFrame(all_combo_results)
        result_df.to_csv(os.path.join(args.output_dir, "all_combo_metrics.csv"), index=False)

        if primary_metric not in result_df.columns:
            log_message(
                log_path,
                f"WARNING: primary metric '{primary_metric}' not found in all_combo_metrics.csv. "
                "best_model_summary.csv will not be created."
            )
        else:
            valid_df = result_df.dropna(subset=[primary_metric]).copy()
            if valid_df.empty:
                log_message(
                    log_path,
                    f"WARNING: no valid values for primary metric '{primary_metric}'. "
                    "best_model_summary.csv will not be created."
                )
            else:
                valid_df = valid_df.sort_values(
                    primary_metric,
                    ascending=not higher_is_better,
                )
                best_row = valid_df.iloc[0].to_dict()
                best_row["selection_metric"] = primary_metric
                best_row["selection_higher_is_better"] = higher_is_better
                best_row["selection_task_type"] = args.task_type

                pd.DataFrame([best_row]).to_csv(
                    os.path.join(args.output_dir, "best_model_summary.csv"),
                    index=False,
                )
                log_message(log_path, f"best combo selected by {primary_metric}: {best_row.get(primary_metric)}")


if __name__ == "__main__":
    main()

