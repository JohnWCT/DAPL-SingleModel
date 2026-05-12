"""
E_tcga_eval_by_drug.py
======================

接續 `D_summary.py` 產出的目錄，讀取複製後的 C 階段預測檔：

    <d_summary_dir>/copied_best_C_prototypical/tcga_eval_predictions.csv

依**個別藥物**（預設 `drug_id`）彙總 TCGA eval 表現：對每個 CV fold 先算一組
分類指標，再對各指標做跨 fold 的 mean / std。

輸出欄位含：`stat`（此處定義為該藥物在單一 fold 內的樣本數 n）、AUC、AUPR、
Accuracy、F1、Precision、Recall 的 `_mean` / `_std`。

用法（Docker 內路徑範例）::

    docker exec DAPL python3 /workspace/DAPL_git/E_tcga_eval_by_drug.py \\
      --d_summary_dir /workspace/DAPL_git/output_dir/repro_MDSM_D_summary_classification_100epoch \\
      --output_csv /workspace/DAPL_git/output_dir/repro_MDSM_D_summary_classification_100epoch/tcga_eval_by_drug_fold_mean_std.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_predictions_path(d_summary_dir: str, rel_path: str) -> str:
    path = os.path.join(d_summary_dir, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到預測檔: {path}")
    return path


def extract_pred_scores(df: pd.DataFrame) -> np.ndarray:
    if "prediction_probability" in df.columns:
        return pd.to_numeric(df["prediction_probability"], errors="coerce").to_numpy(dtype=np.float64)
    if "prediction" in df.columns:
        return pd.to_numeric(df["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    raise ValueError(
        "預測檔需含 prediction_probability（分類）或可轉成 float 的 prediction 欄位。"
    )


def extract_true_labels(df: pd.DataFrame) -> np.ndarray:
    if "ground_truth" not in df.columns:
        raise ValueError("預測檔缺少 ground_truth 欄位。")
    y = pd.to_numeric(df["ground_truth"], errors="coerce").to_numpy(dtype=np.float64)
    return y


def extract_binary_pred(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    if "prediction_binary" in df.columns:
        return pd.to_numeric(df["prediction_binary"], errors="coerce").fillna(0).astype(int).to_numpy()
    probs = extract_pred_scores(df)
    thr = 0.5
    if "threshold" in df.columns:
        t = pd.to_numeric(df["threshold"], errors="coerce").dropna()
        if len(t) > 0 and np.isfinite(t.iloc[0]):
            thr = float(t.iloc[0])
    return (probs >= thr).astype(int)


def compute_fold_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)
    n = int(len(y_true))
    out = {
        "stat": float(n),
        "AUC": np.nan,
        "AUPR": np.nan,
        "Accuracy": np.nan,
        "F1": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
    }
    if n == 0:
        return out
    valid = np.isfinite(y_score) & np.isfinite(y_true.astype(float))
    if not valid.all():
        y_true = y_true[valid]
        y_score = y_score[valid]
        y_pred_bin = y_pred_bin[valid]
        n = int(len(y_true))
        out["stat"] = float(n)
    if n == 0:
        return out
    labels = np.unique(y_true)
    out["Accuracy"] = float(accuracy_score(y_true, y_pred_bin))
    out["F1"] = float(f1_score(y_true, y_pred_bin, zero_division=0))
    out["Precision"] = float(precision_score(y_true, y_pred_bin, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred_bin, zero_division=0))
    if len(labels) > 1:
        try:
            out["AUC"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["AUC"] = np.nan
        try:
            out["AUPR"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out["AUPR"] = np.nan
    return out


def aggregate_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    if arr.size < 2:
        return mean, float("nan")
    std = float(np.nanstd(arr, ddof=1))
    return mean, std


def run(d_summary_dir: str, rel_predictions: str, drug_key: str, output_csv: str | None) -> pd.DataFrame:
    pred_path = resolve_predictions_path(d_summary_dir, rel_predictions)
    df = pd.read_csv(pred_path)
    if drug_key not in df.columns:
        raise ValueError(f"藥物鍵欄位 '{drug_key}' 不存在於 {pred_path}；可用欄位: {list(df.columns)}")
    if "fold" not in df.columns:
        df["_fold"] = 1
        fold_col = "_fold"
    else:
        fold_col = "fold"

    display_name_col = "original_drug_name" if "original_drug_name" in df.columns else drug_key

    metric_names = ["stat", "AUC", "AUPR", "Accuracy", "F1", "Precision", "Recall"]
    rows_out = []

    for drug_val, g_all in df.groupby(drug_key, sort=True):
        fold_metrics: dict[int, dict] = {}
        for fold_id, g in g_all.groupby(fold_col, sort=True):
            y_true = extract_true_labels(g)
            y_score = extract_pred_scores(g)
            y_bin = extract_binary_pred(g, y_true)
            fold_metrics[int(fold_id)] = compute_fold_metrics(y_true, y_score, y_bin)

        fold_ids = sorted(fold_metrics.keys())
        name_series = g_all[display_name_col].dropna().astype(str)
        if len(name_series) > 0:
            display_name = str(name_series.mode().iloc[0])
        else:
            display_name = str(drug_val)

        row = {
            drug_key: drug_val,
            display_name_col: display_name,
            "n_folds": len(fold_ids),
        }
        for m in metric_names:
            vals = [fold_metrics[f][m] for f in fold_ids]
            mean_v, std_v = aggregate_mean_std(vals)
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
        rows_out.append(row)

    out_df = pd.DataFrame(rows_out)
    out_path = output_csv
    if not out_path:
        out_path = os.path.join(d_summary_dir, "tcga_eval_by_drug_fold_mean_std.csv")
    ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
    out_df.to_csv(out_path, index=False)
    return out_df


def parse_args():
    p = argparse.ArgumentParser(
        description="依 D_summary 輸出目錄內的 tcga_eval_predictions.csv，對各藥物做跨 fold mean/std 彙總。"
    )
    p.add_argument(
        "--d_summary_dir",
        required=True,
        help="D_summary.py 的 --output_dir（內含 copied_best_C_prototypical/）",
    )
    p.add_argument(
        "--predictions_rel_path",
        default=os.path.join("copied_best_C_prototypical", "tcga_eval_predictions.csv"),
        help="相對於 d_summary_dir 的預測檔路徑",
    )
    p.add_argument(
        "--drug_key",
        default="drug_id",
        help="分組用的藥物欄位（預設 drug_id，可改 original_drug_name）",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="輸出 CSV 路徑；預設寫在 d_summary_dir/tcga_eval_by_drug_fold_mean_std.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    d_summary_dir = os.path.abspath(args.d_summary_dir)
    out_df = run(
        d_summary_dir=d_summary_dir,
        rel_predictions=args.predictions_rel_path,
        drug_key=args.drug_key,
        output_csv=args.output_csv,
    )
    out_path = args.output_csv or os.path.join(d_summary_dir, "tcga_eval_by_drug_fold_mean_std.csv")
    print(f"[E_tcga_eval_by_drug] wrote {out_path} ({len(out_df)} drugs)")


if __name__ == "__main__":
    main()
