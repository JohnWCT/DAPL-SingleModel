"""
dapl_eval_outputs.py
====================

DAPL C 階段 SSDA/CODE-AE 對齊的評估輸出模組。

支援：
- source_test：CCLE 獨立測試（classification 或 regression）
- target_eval：TCGA 二元分類評估（可多個 suffix，如 _only / _DAPL）
- macro / weighted / overall 彙總 + per-drug 跨 fold mean/std
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = (
    "auroc",
    "aupr",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
)
REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson", "spearman")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _public_metric_name(name: str) -> str:
    return "auc" if name == "auroc" else name


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def normalize_classification_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "pred_label" not in out.columns:
        if "prediction_binary" in out.columns:
            out["pred_label"] = pd.to_numeric(out["prediction_binary"], errors="coerce").fillna(0).astype(int)
        else:
            raise ValueError("classification predictions 需含 prediction_binary 或 pred_label")
    if "pred_score" not in out.columns:
        if "prediction_probability" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction_probability"], errors="coerce")
        elif "prediction" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction"], errors="coerce")
        else:
            raise ValueError("classification predictions 需含 prediction_probability 或 prediction")
    out["ground_truth"] = pd.to_numeric(out["ground_truth"], errors="coerce")
    out = out.dropna(subset=["ground_truth", "pred_score", "pred_label"])
    out["ground_truth"] = out["ground_truth"].astype(int)
    out["pred_label"] = out["pred_label"].astype(int)
    return out


def normalize_regression_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "pred_score" not in out.columns:
        if "prediction_neg_log2_auc" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction_neg_log2_auc"], errors="coerce")
        elif "prediction" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction"], errors="coerce")
        else:
            raise ValueError("regression predictions 需含 prediction_neg_log2_auc 或 prediction")
    out["ground_truth"] = pd.to_numeric(out["ground_truth"], errors="coerce")
    out = out.dropna(subset=["ground_truth", "pred_score"])
    return out


def normalize_tcga_predictions(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """TCGA target eval 一律以二元 Label 評估；regression 模式用連續分數算 AUC/AUPR。"""
    if df.empty:
        return df
    out = df.copy()
    if "pred_label" not in out.columns:
        if "prediction_binary" in out.columns:
            out["pred_label"] = pd.to_numeric(out["prediction_binary"], errors="coerce").fillna(0).astype(int)
        else:
            raise ValueError("TCGA predictions 需含 prediction_binary 或 pred_label")
    if "pred_score" not in out.columns:
        if "prediction_probability" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction_probability"], errors="coerce")
        elif "prediction_neg_log2_auc" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction_neg_log2_auc"], errors="coerce")
        elif "prediction" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction"], errors="coerce")
        else:
            raise ValueError("TCGA predictions 需含 prediction_probability / prediction_neg_log2_auc / prediction")
    out["ground_truth"] = pd.to_numeric(out["ground_truth"], errors="coerce")
    out = out.dropna(subset=["ground_truth", "pred_score", "pred_label"])
    out["ground_truth"] = out["ground_truth"].astype(int)
    out["pred_label"] = out["pred_label"].astype(int)
    return out


def compute_classification_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in pred_df.groupby("drug_id", sort=True):
        y = g["ground_truth"].to_numpy(dtype=int)
        y_score = g["pred_score"].to_numpy(dtype=float)
        pred = g["pred_label"].to_numpy(dtype=int)
        rows.append(
            {
                "drug_id": drug_id,
                "n": len(g),
                "n_observed": len(g),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "auroc": _safe_auc(y, y_score),
                "aupr": _safe_aupr(y, y_score),
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            }
        )
    return pd.DataFrame(rows)


def compute_classification_metrics_overall(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {m: float("nan") for m in CLASSIFICATION_METRICS}
    y = pred_df["ground_truth"].to_numpy(dtype=int)
    y_score = pred_df["pred_score"].to_numpy(dtype=float)
    pred = pred_df["pred_label"].to_numpy(dtype=int)
    return {
        "auroc": _safe_auc(y, y_score),
        "aupr": _safe_aupr(y, y_score),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def compute_classification_metrics_summary(
    per_drug: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame:
    overall = compute_classification_metrics_overall(pred_df)
    rows: list[dict[str, object]] = []
    for m in CLASSIFICATION_METRICS:
        vals = per_drug[m].dropna() if m in per_drug.columns else pd.Series(dtype=float)
        macro_val = float(vals.mean()) if len(vals) else float("nan")
        w_col = "n_observed" if "n_observed" in per_drug.columns else "n"
        w = per_drug[w_col].values
        v = per_drug[m].values if m in per_drug.columns else np.array([])
        mask = ~np.isnan(v.astype(float))
        weighted = float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")
        for agg, val in (
            ("macro", macro_val),
            ("micro", overall.get(m, float("nan"))),
            ("weighted", weighted),
            ("overall", overall.get(m, float("nan"))),
        ):
            rows.append(
                {
                    "metric_name": f"{agg}_{m}",
                    "metric": m,
                    "aggregation": agg,
                    "metric_value": val,
                    "n_valid_drugs": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def compute_regression_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in pred_df.groupby("drug_id", sort=True):
        y = g["ground_truth"].to_numpy(dtype=float)
        p = g["pred_score"].to_numpy(dtype=float)
        mae = float(mean_absolute_error(y, p))
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_std = float(np.std(y.astype(np.float64)))
            pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
            sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")
        rows.append(
            {
                "drug_id": drug_id,
                "n": len(g),
                "n_observed": len(g),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson": float(pr) if pr == pr else float("nan"),
                "spearman": float(sp) if sp == sp else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_regression_metrics_overall(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {m: float("nan") for m in REGRESSION_METRICS}
    y = pred_df["ground_truth"].to_numpy(dtype=float)
    p = pred_df["pred_score"].to_numpy(dtype=float)
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_std = float(np.std(y.astype(np.float64)))
        pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
        sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": float(pr) if pr == pr else float("nan"),
        "spearman": float(sp) if sp == sp else float("nan"),
    }


def compute_regression_metrics_summary(
    per_drug: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame:
    overall = compute_regression_metrics_overall(pred_df)
    rows: list[dict[str, object]] = []
    for m in REGRESSION_METRICS:
        vals = per_drug[m].dropna() if m in per_drug.columns else pd.Series(dtype=float)
        macro_val = float(vals.mean()) if len(vals) else float("nan")
        w_col = "n_observed" if "n_observed" in per_drug.columns else "n"
        w = per_drug[w_col].values
        v = per_drug[m].values if m in per_drug.columns else np.array([])
        mask = ~np.isnan(v.astype(float))
        weighted = float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")
        for agg, val in (
            ("macro", macro_val),
            ("micro", overall.get(m, float("nan"))),
            ("weighted", weighted),
            ("overall", overall.get(m, float("nan"))),
        ):
            rows.append(
                {
                    "metric_name": f"{agg}_{m}",
                    "metric": m,
                    "aggregation": agg,
                    "metric_value": val,
                    "n_valid_drugs": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def summary_to_ssda_wide(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["metric", "macro", "weighted", "overall"])
    rows: list[dict[str, object]] = []
    for metric in summary["metric"].dropna().unique():
        sub = summary[summary["metric"] == metric]
        row: dict[str, object] = {"metric": _public_metric_name(str(metric))}
        for agg in ("macro", "weighted", "overall", "micro"):
            sel = sub[sub["aggregation"] == agg]
            if len(sel):
                row[agg] = float(sel["metric_value"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_value_columns(df: pd.DataFrame) -> list[str]:
    skip = {"drug_id", "n", "n_observed", "fold", "n_positive", "n_negative"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def aggregate_per_drug_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    if "drug_id" not in combined.columns:
        return pd.DataFrame()
    metric_cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for drug_id, grp in combined.groupby("drug_id"):
        row: dict[str, object] = {
            "drug_id": drug_id,
            "n_folds": int(grp["fold"].nunique()) if "fold" in grp.columns else len(grp),
        }
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for metric_name, grp in combined.groupby("metric"):
        row: dict[str, object] = {"metric": metric_name, "n_folds": len(grp)}
        for col in ("macro", "weighted", "overall"):
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_eval_summary(
    src_fold_frames: list[pd.DataFrame],
    tgt_fold_frames: list[pd.DataFrame],
    domain_suffix: str = "",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    src_domain = "source_test"
    tgt_domain = f"target_eval{domain_suffix}"
    if src_fold_frames:
        frames.append(aggregate_summary_metrics(src_fold_frames).assign(domain=src_domain))
    if tgt_fold_frames:
        frames.append(aggregate_summary_metrics(tgt_fold_frames).assign(domain=tgt_domain))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    front = ["domain", "metric", "n_folds"]
    rest = [c for c in combined.columns if c not in front]
    return combined[front + rest]


def _build_fold_summaries_classification(
    pred_df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    pred_df = normalize_classification_predictions(pred_df)
    if "fold" not in pred_df.columns:
        pred_df = pred_df.assign(fold=1)
    per_drug_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for fold_id, g in pred_df.groupby("fold", sort=True):
        per_drug = compute_classification_metrics_per_drug(g)
        summary_long = compute_classification_metrics_summary(per_drug, g)
        summary_wide = summary_to_ssda_wide(summary_long)
        fold_val = int(fold_id)
        per_drug_frames.append(per_drug.assign(fold=fold_val))
        summary_frames.append(summary_wide.assign(fold=fold_val))
    return per_drug_frames, summary_frames


def _build_fold_summaries_regression(
    pred_df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    pred_df = normalize_regression_predictions(pred_df)
    if "fold" not in pred_df.columns:
        pred_df = pred_df.assign(fold=1)
    per_drug_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for fold_id, g in pred_df.groupby("fold", sort=True):
        per_drug = compute_regression_metrics_per_drug(g)
        summary_long = compute_regression_metrics_summary(per_drug, g)
        summary_wide = summary_to_ssda_wide(summary_long)
        fold_val = int(fold_id)
        per_drug_frames.append(per_drug.assign(fold=fold_val))
        summary_frames.append(summary_wide.assign(fold=fold_val))
    return per_drug_frames, summary_frames


def _build_fold_summaries_target(
    pred_df: pd.DataFrame,
    task_type: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    pred_df = normalize_tcga_predictions(pred_df, task_type)
    if "fold" not in pred_df.columns:
        pred_df = pred_df.assign(fold=1)
    per_drug_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for fold_id, g in pred_df.groupby("fold", sort=True):
        per_drug = compute_classification_metrics_per_drug(g)
        summary_long = compute_classification_metrics_summary(per_drug, g)
        summary_wide = summary_to_ssda_wide(summary_long)
        fold_val = int(fold_id)
        per_drug_frames.append(per_drug.assign(fold=fold_val))
        summary_frames.append(summary_wide.assign(fold=fold_val))
    return per_drug_frames, summary_frames


def _summary_across_folds(summary_frames: list[pd.DataFrame]) -> pd.DataFrame:
    across_rows: list[pd.DataFrame] = []
    for summ in summary_frames:
        fold_id = int(summ["fold"].iloc[0])
        long_rows = []
        for _, row in summ.iterrows():
            for agg in ("macro", "weighted", "overall", "micro"):
                if agg in row and pd.notna(row[agg]):
                    long_rows.append({"metric": row["metric"], agg: float(row[agg]), "fold": fold_id})
        if long_rows:
            fold_long = pd.DataFrame(long_rows)
            fold_wide = fold_long.groupby(["metric", "fold"], as_index=False).first()
            across_rows.append(fold_wide)
    return pd.concat(across_rows, ignore_index=True) if across_rows else pd.DataFrame()


def write_domain_eval_outputs(
    pred_df: pd.DataFrame,
    output_dir: str,
    prefix: str,
    task_type: str,
    domain: str,
) -> dict[str, pd.DataFrame]:
    """寫入單一 domain 的 SSDA 風格評估表。"""
    ensure_dir(output_dir)
    if domain == "source":
        if task_type == "regression":
            per_drug_frames, summary_frames = _build_fold_summaries_regression(pred_df)
        else:
            per_drug_frames, summary_frames = _build_fold_summaries_classification(pred_df)
    else:
        per_drug_frames, summary_frames = _build_fold_summaries_target(pred_df, task_type)

    across_df = _summary_across_folds(summary_frames)
    fold_mean_std_df = aggregate_summary_metrics(summary_frames)
    per_drug_fold_df = aggregate_per_drug_metrics(per_drug_frames)

    across_path = os.path.join(output_dir, f"{prefix}_metrics_summary_across_folds.csv")
    summary_path = os.path.join(output_dir, f"{prefix}_metrics_summary_fold_mean_std.csv")
    per_drug_path = os.path.join(output_dir, f"{prefix}_metrics_per_drug_fold_mean_std.csv")
    across_df.to_csv(across_path, index=False)
    fold_mean_std_df.to_csv(summary_path, index=False)
    per_drug_fold_df.to_csv(per_drug_path, index=False)

    return {
        "across": across_df,
        "summary": fold_mean_std_df,
        "per_drug": per_drug_fold_df,
        "per_drug_frames": per_drug_frames,
        "summary_frames": summary_frames,
    }


def write_full_eval_outputs(
    output_dir: str,
    task_type: str,
    source_pred_df: pd.DataFrame | None = None,
    target_pred_dfs: dict[str, pd.DataFrame] | None = None,
    source_pred_path: str | None = None,
    target_pred_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    """
    一步到位寫入 source_test + 多個 target_eval 的 SSDA 風格輸出。

    target_pred_dfs / target_pred_paths 的 key 為 suffix：
      ""        -> target_eval
      "_only"   -> target_eval_only
      "_DAPL"   -> target_eval_DAPL
    """
    ensure_dir(output_dir)
    results: dict[str, object] = {}

    if source_pred_df is None and source_pred_path:
        source_pred_df = pd.read_csv(source_pred_path)
    if source_pred_df is not None and not source_pred_df.empty:
        results["source_test"] = write_domain_eval_outputs(
            source_pred_df, output_dir, "source_test", task_type, "source"
        )

    combined_parts: list[pd.DataFrame] = []
    if "source_test" in results:
        src_summary_frames = results["source_test"]["summary_frames"]
        if src_summary_frames:
            combined_parts.append(
                aggregate_summary_metrics(src_summary_frames).assign(domain="source_test")
            )

    if target_pred_dfs is None:
        target_pred_dfs = {}
    if target_pred_paths:
        for suffix, path in target_pred_paths.items():
            if os.path.isfile(path):
                target_pred_dfs[suffix] = pd.read_csv(path)

    for suffix, tgt_df in sorted(target_pred_dfs.items()):
        if tgt_df is None or tgt_df.empty:
            continue
        prefix = f"target_eval{suffix}"
        domain_result = write_domain_eval_outputs(
            tgt_df, output_dir, prefix, task_type, "target"
        )
        results[prefix] = domain_result
        tgt_summary_frames = domain_result["summary_frames"]
        if tgt_summary_frames:
            combined_parts.append(
                aggregate_summary_metrics(tgt_summary_frames).assign(domain=prefix)
            )

    if combined_parts:
        combined_df = pd.concat(combined_parts, ignore_index=True)
        front = ["domain", "metric", "n_folds"]
        rest = [c for c in combined_df.columns if c not in front]
        combined_df = combined_df[front + rest]
        combined_path = os.path.join(output_dir, "eval_metrics_summary_fold_mean_std.csv")
        combined_df.to_csv(combined_path, index=False)
        results["combined"] = combined_df

    return results


# 預設 TCGA target eval 資料集（主 + 次要）
DEFAULT_TCGA_EVAL_SOURCES = [
    {
        "suffix": "",
        "path": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
        "cancer_type_col": "cancers",
        "format": "tcga_dapl",
        "description": "GDSC intersect13 (main target eval)",
    },
    {
        "suffix": "_only",
        "path": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
        "cancer_type_col": "cancers",
        "format": "tcga_dapl",
        "description": "TCGA-only3 (secondary target eval)",
    },
    {
        "suffix": "_DAPL",
        "path": "data/TCGA/TCGA_drug_response_from_DAPL.csv",
        "cancer_type_col": "primary_disease",
        "format": "tcga_dapl",
        "description": "DAPL TCGA drug response (secondary target eval)",
    },
]
