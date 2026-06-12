"""
F_eval_metrics_summary.py
=========================

從 C 階段已輸出的 predictions CSV，補算或重算 SSDA 風格評估表。
核心邏輯在 dapl_eval_outputs.py；此腳本為薄封裝 CLI。

用法（Docker 內）::

    docker exec DAPL python3 /workspace/DAPL-master/F_eval_metrics_summary.py \\
      --d_summary_dir /workspace/DAPL-master/output_dir/repro_MDSM_D_summary_classification_100epoch
"""

from __future__ import annotations

import argparse
import os

from dapl_eval_outputs import DEFAULT_TCGA_EVAL_SOURCES, write_full_eval_outputs


def _target_pred_paths(d_summary_dir: str, rel_dir: str) -> dict[str, str]:
    base = os.path.join(d_summary_dir, rel_dir)
    paths: dict[str, str] = {}
    for src in DEFAULT_TCGA_EVAL_SOURCES:
        suffix = src["suffix"]
        name = f"target_eval_predictions{suffix}.csv" if suffix else "target_eval_predictions.csv"
        path = os.path.join(base, name)
        if os.path.isfile(path):
            paths[suffix] = path
    return paths


def run(
    d_summary_dir: str,
    rel_dir: str,
    task_type: str,
    output_dir: str | None,
) -> dict[str, object]:
    out_dir = output_dir or d_summary_dir
    base = os.path.join(d_summary_dir, rel_dir)
    source_path = os.path.join(base, "source_test_predictions.csv")
    if not os.path.isfile(source_path):
        source_path = os.path.join(base, "ccle_test_predictions.csv")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"找不到 CCLE 預測檔於 {base}")

    target_paths = _target_pred_paths(d_summary_dir, rel_dir)
    if not target_paths:
        raise FileNotFoundError(f"找不到任何 TCGA target 預測檔於 {base}")

    return write_full_eval_outputs(
        out_dir,
        task_type,
        source_pred_path=source_path,
        target_pred_paths=target_paths,
    )


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
        "--rel_dir",
        default=os.path.join("copied_best_C_prototypical"),
        help="相對 d_summary_dir 的 C 階段輸出子目錄",
    )
    p.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        default="classification",
        help="source_test 指標類型；target_eval 一律二元分類",
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
        rel_dir=args.rel_dir,
        task_type=args.task_type,
        output_dir=out_dir,
    )
    for key in ("source_test", "target_eval", "target_eval_only", "target_eval_DAPL"):
        if key in results:
            df = results[key]["summary"]
            print(f"[F_eval_metrics_summary] {key}: {len(df)} metrics")
            if not df.empty:
                print(df.to_string(index=False))
    print(f"[F_eval_metrics_summary] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
