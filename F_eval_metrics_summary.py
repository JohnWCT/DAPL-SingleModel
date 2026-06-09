"""
F_eval_metrics_summary.py
=========================

從 C 階段輸出的 predictions CSV，計算與 SSDA4Drug/CODE-AE 對齊的
macro / weighted / overall 分類指標，並跨 fold 彙總 mean/std。

輸出格式對齊：
  - source_test_metrics_summary_across_folds.csv
  - source_test_metrics_summary_fold_mean_std.csv
  - target_eval_metrics_summary_across_folds.csv
  - target_eval_metrics_summary_fold_mean_std.csv

用法（Docker 內）::

    docker exec DAPL python3 /workspace/DAPL-master/F_eval_metrics_summary.py \\
      --d_summary_dir /workspace/DAPL-master/output_dir/repro_MDSM_D_summary_classification_100epoch
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
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


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Map DAPL prediction columns to SSDA-style names."""
    if df.empty:
        return df
    out = df.copy()
    if "pred_label" not in out.columns:
        if "prediction_binary" in out.columns:
            out["pred_label"] = pd.to_numeric(out["prediction_binary"], errors="coerce").fillna(0).astype(int)
        else:
            raise ValueError("predictions 需含 prediction_binary 或 pred_label")
    if "pred_score" not in out.columns:
        if "prediction_probability" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction_probability"], errors="coerce")
        elif "prediction" in out.columns:
            out["pred_score"] = pd.to_numeric(out["prediction"], errors="coerce")
        else:
            raise ValueError("predictions 需含 prediction_probability 或 prediction")
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
        w = per_drug["n_observed"].values
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


def build_fold_summaries(pred_df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Return (per_drug_frames, summary_frames) with fold column."""
    pred_df = normalize_predictions(pred_df)
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


def write_domain_outputs(
    pred_path: str,
    output_dir: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(pred_path)
    if "drug_id" not in pred_df.columns:
        raise ValueError(f"{pred_path} 缺少 drug_id 欄位")
    per_drug_frames, summary_frames = build_fold_summaries(pred_df)

    across_rows: list[pd.DataFrame] = []
    for summ in summary_frames:
        fold_id = int(summ["fold"].iloc[0])
        long_rows = []
        for _, row in summ.iterrows():
            for agg in ("macro", "weighted", "overall", "micro"):
                if agg in row and pd.notna(row[agg]):
                    long_rows.append(
                        {
                            "metric": row["metric"],
                            agg: float(row[agg]),
                            "fold": fold_id,
                        }
                    )
        if long_rows:
            fold_long = pd.DataFrame(long_rows)
            fold_wide = fold_long.groupby(["metric", "fold"], as_index=False).first()
            across_rows.append(fold_wide)

    across_df = pd.concat(across_rows, ignore_index=True) if across_rows else pd.DataFrame()
    fold_mean_std_df = aggregate_summary_metrics(summary_frames)

    ensure_dir(output_dir)
    across_path = os.path.join(output_dir, f"{prefix}_metrics_summary_across_folds.csv")
    summary_path = os.path.join(output_dir, f"{prefix}_metrics_summary_fold_mean_std.csv")
    across_df.to_csv(across_path, index=False)
    fold_mean_std_df.to_csv(summary_path, index=False)
    return across_df, fold_mean_std_df


def run(
    d_summary_dir: str,
    ccle_rel: str,
    tcga_rel: str,
    output_dir: str | None,
) -> dict[str, pd.DataFrame]:
    out_dir = output_dir or d_summary_dir
    ccle_path = os.path.join(d_summary_dir, ccle_rel)
    tcga_path = os.path.join(d_summary_dir, tcga_rel)
    if not os.path.isfile(ccle_path):
        raise FileNotFoundError(f"找不到 CCLE 預測檔: {ccle_path}")
    if not os.path.isfile(tcga_path):
        raise FileNotFoundError(f"找不到 TCGA 預測檔: {tcga_path}")

    src_across, src_summary = write_domain_outputs(
        ccle_path, out_dir, prefix="source_test"
    )
    tgt_across, tgt_summary = write_domain_outputs(
        tcga_path, out_dir, prefix="target_eval"
    )

    combined_frames: list[pd.DataFrame] = []
    if not src_summary.empty:
        combined_frames.append(src_summary.assign(domain="source_test"))
    if not tgt_summary.empty:
        combined_frames.append(tgt_summary.assign(domain="target_eval"))
    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    if not combined_df.empty:
        front = ["domain", "metric", "n_folds"]
        rest = [c for c in combined_df.columns if c not in front]
        combined_df = combined_df[front + rest]
        combined_path = os.path.join(out_dir, "eval_metrics_summary_fold_mean_std.csv")
        combined_df.to_csv(combined_path, index=False)

    return {
        "source_test_across": src_across,
        "source_test_summary": src_summary,
        "target_eval_across": tgt_across,
        "target_eval_summary": tgt_summary,
        "combined_summary": combined_df,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從 CCLE/TCGA predictions 產出 SSDA 風格的 macro/weighted/overall 指標彙總表。"
    )
    p.add_argument(
        "--d_summary_dir",
        required=True,
        help="D_summary 輸出目錄（內含 copied_best_C_prototypical/）",
    )
    p.add_argument(
        "--ccle_rel_path",
        default=os.path.join("copied_best_C_prototypical", "ccle_test_predictions.csv"),
        help="相對 d_summary_dir 的 CCLE source test 預測檔",
    )
    p.add_argument(
        "--tcga_rel_path",
        default=os.path.join("copied_best_C_prototypical", "tcga_eval_predictions.csv"),
        help="相對 d_summary_dir 的 TCGA target eval 預測檔",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="輸出目錄；預設與 d_summary_dir 相同",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d_summary_dir = os.path.abspath(args.d_summary_dir)
    out_dir = os.path.abspath(args.output_dir or d_summary_dir)
    results = run(
        d_summary_dir=d_summary_dir,
        ccle_rel=args.ccle_rel_path,
        tcga_rel=args.tcga_rel_path,
        output_dir=out_dir,
    )
    for key in ("source_test_summary", "target_eval_summary"):
        df = results[key]
        print(f"[F_eval_metrics_summary] {key}: {len(df)} metrics")
        if not df.empty:
            print(df.to_string(index=False))
    print(f"[F_eval_metrics_summary] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
