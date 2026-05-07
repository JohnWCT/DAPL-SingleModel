"""
A_pretrain.py
=============

用途
----
新版 sample pretrain 流程。此檔案保留原始 `pretrain.py` 的 shared/private VAE
與 domain deconfounding 訓練概念，但改為支援 Winnie 新版 input：

- CCLE source domain: `data_Winnie/CCLE_impact_hotspot.csv`
- TCGA target domain: `data_Winnie/TCGA_impact_hotspot.csv`

資料格式
--------
預設 `Sample_ID` 為 sample ID 欄位，其餘欄位都視為 sample expression / mutation
feature，會轉成 numeric。程式會自動取 CCLE 與 TCGA 的共同 feature 欄位，並以
讀入資料的 feature 數量設定 VAE input dimension：

    input_dim = ccle_tensor.shape[1]
    VAE(input_size=input_dim, output_size=input_dim, ...)

不得將 input dimension 寫死成原始的 1426。

Latent representation 定義
--------------------------
輸出的 sample latent representation 使用 shared VAE 的 latent `z`，也就是
`tools.model.VAE.forward()` 回傳的第二個值：

    re_x, z, mu, sigma = shared_vae(x)

輸出檔案
--------
`--output_dir` 內會產生：

- `CCLE_latent_representation.pkl`
- `TCGA_latent_representation.pkl`
- `deconfounding_metrics.csv`
- `kmeans_ari_metrics.csv`
- `after_traingan_shared_vae.pth`
- `A_training_log.txt`

其中 latent pkl 格式為：

    {
        "sample_id_1": [latent_value_1, latent_value_2, ...],
        "sample_id_2": [latent_value_1, latent_value_2, ...],
    }

Domain deconfounding 評估
-------------------------
`deconfounding_metrics.csv` 會輸出 CCLE vs TCGA latent distribution 的：

- FID
- MMD
- Wasserstein distance

`kmeans_ari_metrics.csv` 會使用 `CCLE_cancer_type.csv` 與
`TCGA_cancer_type.csv` 的 cancer type，分別計算：

- CCLE ARI
- TCGA ARI
- Average ARI
- CCLE/TCGA/Average NMI
- CCLE/TCGA/Average Silhouette
- CCLE/TCGA/Average Calinski-Harabasz
- CCLE/TCGA/Average Davies-Bouldin
- K-means k
- cancer type 類別數量

可用 `--run_metrics 0` 跳過上述評估輸出，以加速 smoke test。

Docker 執行範例
---------------
請在 Docker container `DAPL` 內執行，不要在本機 Python 環境安裝或修改套件：

    docker exec DAPL python3 /workspace/DAPL-master/A_pretrain.py \
      --ccle_exp_input /workspace/DAPL-master/data_Winnie/CCLE_impact_hotspot.csv \
      --tcga_exp_input /workspace/DAPL-master/data_Winnie/TCGA_impact_hotspot.csv \
      --sample_id_col Sample_ID \
      --ccle_cancer_type_input /workspace/DAPL-master/data_Winnie/CCLE_cancer_type.csv \
      --tcga_cancer_type_input /workspace/DAPL-master/data_Winnie/TCGA_cancer_type.csv \
      --cancer_type_col Cancer_type \
      --output_dir /workspace/DAPL-master/output_dir/A_pretrain \
      --random_seed 42

快速 smoke test 可只跑部分 grid（例如先改 `PARAMS_GRID`）：

    docker exec DAPL python3 /workspace/DAPL-master/A_pretrain.py \
      --output_dir /workspace/DAPL-master/output_dir/A_pretrain_smoke
"""

import argparse
import itertools
import os
import pickle
import random
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader, TensorDataset

from tools.latent_metrics import (
    calculate_fid as shared_calculate_fid,
    calculate_mmd as shared_calculate_mmd,
    calculate_wasserstein as shared_calculate_wasserstein,
)
from tools.model import Discriminator, VAE, vaeloss


device = "cuda" if torch.cuda.is_available() else "cpu"


LATENT_LAYER_NOTE = (
    "Latent representation uses the shared VAE latent z returned by "
    "tools.model.VAE.forward(): re_x, z, mu, sigma."
)


PARAMS_GRID = {
    "pretrain_num_epochs": [0, 100, 300],
    "pretrain_learning_rate": [0.001],
    "gan_learning_rate": [0.001],
    "train_num_epochs": [100, 300, 2000, 2500],
}


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


def tcga_patient_key(sample_id):
    parts = str(sample_id).split("-")
    if len(parts) >= 3 and str(sample_id).startswith("TCGA-"):
        return "-".join(parts[:3])
    return str(sample_id)


def read_expression(path, sample_id_col, domain_name):
    df = pd.read_csv(path)
    if sample_id_col not in df.columns:
        unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            sample_id_col = unnamed_cols[0]
            print(f"{domain_name}: using index column '{sample_id_col}' as sample ID from {path}")
        else:
            raise ValueError(f"{domain_name}: missing sample id column '{sample_id_col}' in {path}")
    sample_ids = df[sample_id_col].astype(str).tolist()
    if domain_name.upper() == "TCGA":
        sample_ids = [tcga_patient_key(v) for v in sample_ids]
    feature_df = df.drop(columns=[sample_id_col])
    if feature_df.empty:
        raise ValueError(f"{domain_name}: no expression feature columns after removing '{sample_id_col}'")
    converted = feature_df.apply(pd.to_numeric, errors="coerce")
    bad_cols = converted.columns[converted.isna().any()].tolist()
    if bad_cols:
        preview = ", ".join(bad_cols[:10])
        raise ValueError(f"{domain_name}: non-numeric or missing values in feature columns: {preview}")
    converted.index = sample_ids
    if domain_name.upper() == "TCGA" and converted.index.has_duplicates:
        before = len(converted)
        converted = converted[~converted.index.duplicated(keep="first")]
        print(f"TCGA: deduplicated patient IDs after normalization ({before} -> {len(converted)})")
    return converted.astype(np.float32)


def align_feature_columns(ccle_df, tcga_df):
    common_cols = [col for col in ccle_df.columns if col in set(tcga_df.columns)]
    if not common_cols:
        raise ValueError("CCLE and TCGA expression files do not share numeric feature columns")
    missing_ccle = len(tcga_df.columns) - len(common_cols)
    missing_tcga = len(ccle_df.columns) - len(common_cols)
    if missing_ccle or missing_tcga:
        print(
            f"[feature alignment] using {len(common_cols)} shared columns "
            f"(drop {missing_tcga} CCLE-only, {missing_ccle} TCGA-only)"
        )
    return ccle_df.loc[:, common_cols], tcga_df.loc[:, common_cols]


def make_loader(feature_df, batch_size):
    tensor = torch.from_numpy(feature_df.values.astype(np.float32)).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, tensor


def ortho_loss(shared_z, private_z):
    s_l2_norm = torch.norm(shared_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = shared_z.div(s_l2_norm.expand_as(shared_z) + 1e-6)
    p_l2_norm = torch.norm(private_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = private_z.div(p_l2_norm.expand_as(private_z) + 1e-6)
    return torch.mean((s_l2.t().mm(p_l2)).pow(2))


def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand((real_samples.shape[0], 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1), device=device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def next_cycled(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def train_vae_models(ccle_loader, tcga_loader, ccle_tensor, tcga_tensor, args, log_path):
    input_dim = ccle_tensor.shape[1]
    shared_vae = VAE(input_size=input_dim, output_size=input_dim, latent_size=args.latent_dim, hidden_size=args.hidden_dim).to(device)
    source_private_vae = VAE(input_size=input_dim, output_size=input_dim, latent_size=args.latent_dim, hidden_size=args.hidden_dim).to(device)
    target_private_vae = VAE(input_size=input_dim, output_size=input_dim, latent_size=args.latent_dim, hidden_size=args.hidden_dim).to(device)
    models = [shared_vae, source_private_vae, target_private_vae]
    optimizer = torch.optim.Adam(chain(*(m.parameters() for m in models)), lr=args.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.pretrain_epochs))
    best_loss = float("inf")
    best_states = [deepcopy(m.state_dict()) for m in models]
    tolerance = 0

    for epoch in range(args.pretrain_epochs):
        for model in models:
            model.train()
        tcga_iter = iter(tcga_loader)
        train_losses = []
        for ccle_batch in ccle_loader:
            tcga_batch, tcga_iter = next_cycled(tcga_iter, tcga_loader)
            ccle_x = ccle_batch[0]
            tcga_x = tcga_batch[0]
            optimizer.zero_grad()
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccle_x)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcga_x)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccle_x)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcga_x)
            loss = (
                vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccle_x)
                + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcga_x)
                + vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccle_x)
                + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcga_x)
                + ortho_loss(ccle_z, pccle_z)
                + ortho_loss(tcga_z, ptcga_z)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu().item()))

        for model in models:
            model.eval()
        with torch.no_grad():
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccle_tensor)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcga_tensor)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccle_tensor)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcga_tensor)
            eval_loss = (
                vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccle_tensor)
                + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcga_tensor)
                + vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccle_tensor)
                + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcga_tensor)
                + ortho_loss(ccle_z, pccle_z)
                + ortho_loss(tcga_z, ptcga_z)
            )
        eval_value = float(eval_loss.detach().cpu().item())
        log_message(log_path, {"stage": "vae_pretrain", "epoch": epoch, "train_loss": np.mean(train_losses), "eval_loss": eval_value})
        if eval_value < best_loss:
            best_loss = eval_value
            tolerance = 0
            best_states = [deepcopy(m.state_dict()) for m in models]
        else:
            tolerance += 1
        if tolerance >= args.pretrain_patience:
            log_message(log_path, f"VAE early stopping at epoch {epoch}")
            break

    for model, state in zip(models, best_states):
        model.load_state_dict(state)
    return shared_vae, source_private_vae, target_private_vae


def train_domain_gan(shared_vae, source_private_vae, target_private_vae, ccle_loader, tcga_loader, args, log_path):
    if args.gan_epochs <= 0:
        return shared_vae
    discrim = Discriminator(input_dim=args.latent_dim * 2).to(device)
    d_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=args.gan_lr)
    ae_optimizer = torch.optim.RMSprop(
        chain(shared_vae.parameters(), source_private_vae.parameters(), target_private_vae.parameters()),
        lr=args.gan_lr,
    )
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, max(1, args.gan_epochs))
    ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, max(1, args.gan_epochs))
    best_state = deepcopy(shared_vae.state_dict())
    best_loss = float("inf")
    tolerance = 0

    for epoch in range(args.gan_epochs):
        tcga_iter = iter(tcga_loader)
        d_losses = []
        g_losses = []
        for step, ccle_batch in enumerate(ccle_loader):
            tcga_batch, tcga_iter = next_cycled(tcga_iter, tcga_loader)
            ccle_x = ccle_batch[0]
            tcga_x = tcga_batch[0]
            # Keep source/target batch sizes identical for gradient penalty.
            if ccle_x.shape[0] != tcga_x.shape[0]:
                min_bs = min(ccle_x.shape[0], tcga_x.shape[0])
                ccle_x = ccle_x[:min_bs]
                tcga_x = tcga_x[:min_bs]

            shared_vae.eval()
            source_private_vae.eval()
            target_private_vae.eval()
            discrim.train()
            d_optimizer.zero_grad()
            with torch.no_grad():
                _, pzs, _, _ = source_private_vae(ccle_x)
                _, pzt, _, _ = target_private_vae(tcga_x)
                _, zs, _, _ = shared_vae(ccle_x)
                _, zt, _, _ = shared_vae(tcga_x)
            s = torch.cat((zs, pzs), dim=1)
            t = torch.cat((zt, pzt), dim=1)
            d_loss = torch.mean(t) - torch.mean(s) + 10 * compute_gradient_penalty(discrim, s, t)
            d_loss.backward()
            d_optimizer.step()
            d_scheduler.step()
            d_losses.append(float(d_loss.detach().cpu().item()))

            if (step + 1) % args.gan_generator_every == 0:
                shared_vae.train()
                source_private_vae.train()
                target_private_vae.train()
                discrim.eval()
                ae_optimizer.zero_grad()
                pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccle_x)
                ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcga_x)
                ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccle_x)
                tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcga_x)
                g_loss = -torch.mean(discrim(torch.cat((tcga_z, ptcga_z), dim=1)))
                loss = (
                    g_loss
                    + vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccle_x)
                    + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcga_x)
                    + vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccle_x)
                    + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcga_x)
                    + ortho_loss(ccle_z, pccle_z)
                    + ortho_loss(tcga_z, ptcga_z)
                )
                loss.backward()
                ae_optimizer.step()
                ae_scheduler.step()
                g_losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(d_losses) + (np.mean(g_losses) if g_losses else 0.0))
        log_message(log_path, {"stage": "domain_gan", "epoch": epoch, "d_loss": np.mean(d_losses), "g_loss": np.mean(g_losses) if g_losses else np.nan})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            tolerance = 0
            best_state = deepcopy(shared_vae.state_dict())
        else:
            tolerance += 1
        if tolerance >= args.gan_patience:
            log_message(log_path, f"GAN early stopping at epoch {epoch}")
            break

    shared_vae.load_state_dict(best_state)
    return shared_vae


def encode_latent_dict(model, feature_df, batch_size):
    model.eval()
    ids = feature_df.index.astype(str).tolist()
    x = torch.from_numpy(feature_df.values.astype(np.float32)).to(device)
    latents = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(len(ids), start + batch_size)
            _, z, _, _ = model(x[start:end])
            z_np = z.detach().cpu().numpy()
            for idx, sample_id in enumerate(ids[start:end]):
                latents[sample_id] = z_np[idx].tolist()
    return latents


def calculate_fid(source_latent, target_latent):
    return shared_calculate_fid(source_latent, target_latent)


def calculate_mmd(source_latent, target_latent, max_samples=1000, gamma=None):
    return shared_calculate_mmd(source_latent, target_latent, max_samples=max_samples, gamma=gamma)


def calculate_wasserstein(source_latent, target_latent):
    return shared_calculate_wasserstein(source_latent, target_latent)


def read_cancer_type(path, sample_id_col, cancer_type_col):
    df = pd.read_csv(path)
    col_lookup = {str(col).lower(): col for col in df.columns}
    sample_col = sample_id_col if sample_id_col in df.columns else col_lookup.get(str(sample_id_col).lower())
    cancer_col = cancer_type_col if cancer_type_col in df.columns else col_lookup.get(str(cancer_type_col).lower())
    if sample_col is None:
        # Common fallback for Winnie cancer type tables
        sample_col = col_lookup.get("sample_id")
    if cancer_col is None:
        # Common fallback for Winnie cancer type tables
        cancer_col = col_lookup.get("cancer_type")
    missing = []
    if sample_col is None:
        missing.append(sample_id_col)
    if cancer_col is None:
        missing.append(cancer_type_col)
    if missing:
        raise ValueError(f"missing cancer type columns in {path}: {missing}")
    df = df[[sample_col, cancer_col]].dropna()
    df[sample_col] = df[sample_col].astype(str)
    # Normalize TCGA sample IDs to patient-level key so they align with
    # pretrain TCGA expression / downstream Patient_id format.
    df[sample_col] = df[sample_col].apply(tcga_patient_key)
    df[cancer_col] = df[cancer_col].astype(str)
    if df[sample_col].duplicated().any():
        before = len(df)
        df = df.drop_duplicates(subset=[sample_col], keep="first")
        print(f"{path}: deduplicated cancer type IDs after normalization ({before} -> {len(df)})")
    return dict(zip(df[sample_col], df[cancer_col]))


def kmeans_cluster_metrics(latent_dict, cancer_type_map, k):
    ids = [sample_id for sample_id in latent_dict if sample_id in cancer_type_map]
    if len(ids) < 2:
        return {
            "k_eff": np.nan,
            "samples_used": len(ids),
            "ARI": np.nan,
            "NMI": np.nan,
            "Silhouette": np.nan,
            "Calinski_Harabasz": np.nan,
            "Davies_Bouldin": np.nan,
        }
    x = np.asarray([latent_dict[sample_id] for sample_id in ids], dtype=np.float32)
    labels = np.asarray([cancer_type_map[sample_id] for sample_id in ids])
    k_eff = int(max(2, min(k, len(np.unique(labels)), len(ids) - 1)))
    if k_eff < 2:
        return {
            "k_eff": np.nan,
            "samples_used": len(ids),
            "ARI": np.nan,
            "NMI": np.nan,
            "Silhouette": np.nan,
            "Calinski_Harabasz": np.nan,
            "Davies_Bouldin": np.nan,
        }
    pred = KMeans(n_clusters=k_eff, random_state=42, n_init=10).fit_predict(x)
    metrics = {
        "k_eff": float(k_eff),
        "samples_used": len(ids),
        "ARI": float(adjusted_rand_score(labels, pred)),
        "NMI": float(normalized_mutual_info_score(labels, pred)),
        "Silhouette": np.nan,
        "Calinski_Harabasz": np.nan,
        "Davies_Bouldin": np.nan,
    }
    try:
        metrics["Silhouette"] = float(silhouette_score(x, pred))
    except Exception:
        metrics["Silhouette"] = np.nan
    try:
        metrics["Calinski_Harabasz"] = float(calinski_harabasz_score(x, pred))
    except Exception:
        metrics["Calinski_Harabasz"] = np.nan
    try:
        metrics["Davies_Bouldin"] = float(davies_bouldin_score(x, pred))
    except Exception:
        metrics["Davies_Bouldin"] = np.nan
    return metrics


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def parse_args():
    parser = argparse.ArgumentParser("DAPL A_pretrain sample pretrain")
    parser.add_argument("--ccle_exp_input", default="data_Winnie/CCLE_impact_hotspot.csv")
    parser.add_argument("--tcga_exp_input", default="data_Winnie/TCGA_impact_hotspot.csv")
    parser.add_argument("--sample_id_col", default="Sample_ID")
    parser.add_argument("--ccle_cancer_type_input", default="data_Winnie/CCLE_cancer_type.csv")
    parser.add_argument("--tcga_cancer_type_input", default="data_Winnie/TCGA_cancer_type.csv")
    parser.add_argument("--cancer_type_col", default="Cancer_type")
    parser.add_argument("--output_dir", default="output_dir/A_pretrain")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # NOTE:
    # pretrain/gan epochs and learning rates are controlled by PARAMS_GRID.
    # They are intentionally not exposed as CLI arguments to avoid confusion.
    parser.add_argument("--pretrain_patience", type=int, default=20)
    parser.add_argument("--gan_patience", type=int, default=10)
    parser.add_argument("--gan_generator_every", type=int, default=5)
    parser.add_argument(
        "--run_metrics",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: compute deconfounding + K-means metrics; 0: skip metric files for faster smoke tests",
    )
    return parser.parse_args()


def make_set_dir_name(param):
    """Build subdirectory name matching original pretrain.py convention."""
    return (
        "pt_epochs_" + str(param["pretrain_num_epochs"])
        + ",t_epochs_" + str(param["train_num_epochs"])
        + ",Ptlr_" + str(param["pretrain_learning_rate"])
        + ",tlr" + str(param["gan_learning_rate"])
    )


def run_one_combo(param, ccle_df, tcga_df, ccle_loader, tcga_loader, ccle_tensor, tcga_tensor, args, parent_log):
    """Run pretrain + GAN + encode for a single parameter combination."""
    set_dir_name = make_set_dir_name(param)
    combo_dir = os.path.join(args.output_dir, set_dir_name)
    ensure_dir(combo_dir)
    log_path = os.path.join(combo_dir, "A_training_log.txt")
    log_message(log_path, f"param combo: {param}")
    log_message(parent_log, f"=== start combo: {set_dir_name} ===")

    # Skip if already completed
    vae_out = os.path.join(combo_dir, "after_traingan_shared_vae.pth")
    if (
        os.path.exists(vae_out)
        and os.path.exists(os.path.join(combo_dir, "CCLE_latent_representation.pkl"))
        and os.path.exists(os.path.join(combo_dir, "TCGA_latent_representation.pkl"))
    ):
        log_message(parent_log, f"combo {set_dir_name} already completed, skip")
        return

    # Override args with combo values
    combo_args = deepcopy(args)
    combo_args.pretrain_epochs = param["pretrain_num_epochs"]
    combo_args.pretrain_lr = param["pretrain_learning_rate"]
    combo_args.gan_lr = param["gan_learning_rate"]
    combo_args.gan_epochs = param["train_num_epochs"]

    shared_vae, source_private_vae, target_private_vae = train_vae_models(
        ccle_loader, tcga_loader, ccle_tensor, tcga_tensor, combo_args, log_path
    )
    shared_vae = train_domain_gan(
        shared_vae, source_private_vae, target_private_vae,
        ccle_loader, tcga_loader, combo_args, log_path
    )
    torch.save(shared_vae.state_dict(), vae_out)

    ccle_latent = encode_latent_dict(shared_vae, ccle_df, combo_args.batch_size)
    tcga_latent = encode_latent_dict(shared_vae, tcga_df, combo_args.batch_size)
    save_pickle(ccle_latent, os.path.join(combo_dir, "CCLE_latent_representation.pkl"))
    save_pickle(tcga_latent, os.path.join(combo_dir, "TCGA_latent_representation.pkl"))

    if combo_args.run_metrics == 1:
        ccle_arr = np.asarray(list(ccle_latent.values()), dtype=np.float32)
        tcga_arr = np.asarray(list(tcga_latent.values()), dtype=np.float32)
        pd.DataFrame(
            [
                {
                    "source_domain": "CCLE",
                    "target_domain": "TCGA",
                    "FID": calculate_fid(ccle_arr, tcga_arr),
                    "MMD": calculate_mmd(ccle_arr, tcga_arr),
                    "Wasserstein_distance": calculate_wasserstein(ccle_arr, tcga_arr),
                }
            ]
        ).to_csv(os.path.join(combo_dir, "deconfounding_metrics.csv"), index=False)

        ccle_ct = read_cancer_type(combo_args.ccle_cancer_type_input, combo_args.sample_id_col, combo_args.cancer_type_col)
        tcga_ct = read_cancer_type(combo_args.tcga_cancer_type_input, combo_args.sample_id_col, combo_args.cancer_type_col)
        cancer_types = sorted(set(ccle_ct.values()) | set(tcga_ct.values()))
        k = len(cancer_types)
        ccle_metrics = kmeans_cluster_metrics(ccle_latent, ccle_ct, k)
        tcga_metrics = kmeans_cluster_metrics(tcga_latent, tcga_ct, k)
        pd.DataFrame(
            [
                {
                    "CCLE_ARI": ccle_metrics["ARI"],
                    "TCGA_ARI": tcga_metrics["ARI"],
                    "Average_ARI": float(np.nanmean([ccle_metrics["ARI"], tcga_metrics["ARI"]])),
                    "CCLE_NMI": ccle_metrics["NMI"],
                    "TCGA_NMI": tcga_metrics["NMI"],
                    "Average_NMI": float(np.nanmean([ccle_metrics["NMI"], tcga_metrics["NMI"]])),
                    "CCLE_Silhouette": ccle_metrics["Silhouette"],
                    "TCGA_Silhouette": tcga_metrics["Silhouette"],
                    "Average_Silhouette": float(np.nanmean([ccle_metrics["Silhouette"], tcga_metrics["Silhouette"]])),
                    "CCLE_Calinski_Harabasz": ccle_metrics["Calinski_Harabasz"],
                    "TCGA_Calinski_Harabasz": tcga_metrics["Calinski_Harabasz"],
                    "Average_Calinski_Harabasz": float(
                        np.nanmean([ccle_metrics["Calinski_Harabasz"], tcga_metrics["Calinski_Harabasz"]])
                    ),
                    "CCLE_Davies_Bouldin": ccle_metrics["Davies_Bouldin"],
                    "TCGA_Davies_Bouldin": tcga_metrics["Davies_Bouldin"],
                    "Average_Davies_Bouldin": float(
                        np.nanmean([ccle_metrics["Davies_Bouldin"], tcga_metrics["Davies_Bouldin"]])
                    ),
                    "K-means_k": k,
                    "cancer_type_count": k,
                    "CCLE_samples_used": ccle_metrics["samples_used"],
                    "TCGA_samples_used": tcga_metrics["samples_used"],
                    "CCLE_k_eff": ccle_metrics["k_eff"],
                    "TCGA_k_eff": tcga_metrics["k_eff"],
                }
            ]
        ).to_csv(os.path.join(combo_dir, "kmeans_ari_metrics.csv"), index=False)
    else:
        log_message(log_path, "run_metrics=0: skip deconfounding_metrics.csv and kmeans_ari_metrics.csv")
    log_message(log_path, "combo completed")
    log_message(parent_log, f"=== done combo: {set_dir_name} ===")


def main():
    args = parse_args()
    set_seed(args.random_seed)
    ensure_dir(args.output_dir)
    parent_log = os.path.join(args.output_dir, "A_training_log.txt")
    log_message(parent_log, f"device={device}")
    log_message(parent_log, LATENT_LAYER_NOTE)

    # --- Load data once ---
    ccle_df = read_expression(args.ccle_exp_input, args.sample_id_col, "CCLE")
    tcga_df = read_expression(args.tcga_exp_input, args.sample_id_col, "TCGA")
    ccle_df, tcga_df = align_feature_columns(ccle_df, tcga_df)
    log_message(parent_log, f"CCLE samples={len(ccle_df)}, TCGA samples={len(tcga_df)}, input_dim={ccle_df.shape[1]}")
    ccle_loader, ccle_tensor = make_loader(ccle_df, args.batch_size)
    tcga_loader, tcga_tensor = make_loader(tcga_df, args.batch_size)

    # --- Generate parameter combinations ---
    keys, values = zip(*PARAMS_GRID.items())
    combo_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    log_message(parent_log, f"total param combos: {len(combo_list)}")
    log_message(parent_log, f"PARAMS_GRID: {PARAMS_GRID}")

    # --- Iterate over combos ---
    for combo_idx, param in enumerate(combo_list, start=1):
        log_message(parent_log, f"\n[{combo_idx}/{len(combo_list)}] {param}")
        run_one_combo(
            param, ccle_df, tcga_df,
            ccle_loader, tcga_loader, ccle_tensor, tcga_tensor,
            args, parent_log,
        )
    log_message(parent_log, "A_pretrain all combos completed")


if __name__ == "__main__":
    main()
