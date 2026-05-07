"""
D_summary.py

根據 C_prototypical 的 best_model_summary.csv：
1) 找出最佳組合並複製對應 A/C 輸出資料夾
2) 使用最佳 A_pretrain latent 繪製含 cancer type 的 source/target t-SNE
"""

import argparse
import os
import pickle
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def tcga_patient_key(sample_id):
    sample_id = str(sample_id)
    parts = sample_id.split("-")
    if sample_id.startswith("TCGA-") and len(parts) >= 3:
        return "-".join(parts[:3])
    return sample_id


def read_cancer_map(path, sample_id_col, cancer_type_col):
    df = pd.read_csv(path)
    lower_map = {str(c).lower(): c for c in df.columns}
    sid_col = sample_id_col if sample_id_col in df.columns else lower_map.get(sample_id_col.lower())
    ctype_col = cancer_type_col if cancer_type_col in df.columns else lower_map.get(cancer_type_col.lower())
    if sid_col is None:
        sid_col = lower_map.get("sample_id")
    if ctype_col is None:
        ctype_col = lower_map.get("cancer_type")
    if sid_col is None or ctype_col is None:
        raise ValueError(f"Cannot find sample/cancer columns in {path}")
    df = df[[sid_col, ctype_col]].dropna().copy()
    df[sid_col] = df[sid_col].astype(str)
    if "tcga" in os.path.basename(path).lower():
        df[sid_col] = df[sid_col].map(tcga_patient_key)
    df[ctype_col] = df[ctype_col].astype(str)
    return dict(zip(df[sid_col], df[ctype_col]))


def sample_points(ids, feats, labels, max_points, random_seed):
    if len(ids) <= max_points:
        return ids, feats, labels
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(ids), size=max_points, replace=False)
    return [ids[i] for i in idx], feats[idx], [labels[i] for i in idx]


def plot_tsne_with_cancer_type(
    source_ids,
    source_z,
    source_labels,
    target_ids,
    target_z,
    target_labels,
    save_path,
):
    if len(source_z) == 0 or len(target_z) == 0:
        raise ValueError("source or target latent is empty")

    all_feats = np.vstack([source_z, target_z])
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, max(2, (len(all_feats) - 1) // 3)),
        init="random",
        learning_rate="auto",
    )
    out = tsne.fit_transform(all_feats)
    split = len(source_z)
    s2 = out[:split]
    t2 = out[split:]

    plt.figure(figsize=(9, 7))
    all_labels = np.unique(np.asarray(source_labels + target_labels, dtype=object))
    cmap = plt.cm.get_cmap("tab20", max(20, len(all_labels)))
    colors = {lab: cmap(i % cmap.N) for i, lab in enumerate(all_labels)}

    source_labels_arr = np.asarray(source_labels, dtype=object)
    target_labels_arr = np.asarray(target_labels, dtype=object)
    for lab in np.unique(source_labels_arr):
        idx = np.where(source_labels_arr == lab)[0]
        plt.scatter(
            s2[idx, 0],
            s2[idx, 1],
            c=[colors[lab]],
            s=14,
            alpha=0.85,
            marker="o",
            edgecolors="k",
            linewidths=0.3,
        )
    for lab in np.unique(target_labels_arr):
        idx = np.where(target_labels_arr == lab)[0]
        plt.scatter(
            t2[idx, 0],
            t2[idx, 1],
            c=[colors[lab]],
            s=12,
            alpha=0.5,
            marker="^",
            edgecolors="k",
            linewidths=0.3,
        )

    plt.title("Best A_pretrain Latent t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    handles = []
    for lab in all_labels:
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=str(lab), markerfacecolor=colors[lab], markersize=6)
        )
    plt.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser("DAPL summary script")
    p.add_argument("--a_output_dir", required=True, help="A_pretrain root output directory")
    p.add_argument("--c_output_dir", required=True, help="C_prototypical root output directory")
    p.add_argument("--best_summary_file", default="best_model_summary.csv", help="best summary filename under c_output_dir")
    p.add_argument("--ccle_cancer_type_input", default="data_Winnie/CCLE_cancer_type.csv")
    p.add_argument("--tcga_cancer_type_input", default="data_Winnie/TCGA_cancer_type.csv")
    p.add_argument("--sample_id_col", default="Sample_ID")
    p.add_argument("--cancer_type_col", default="Cancer_type")
    p.add_argument("--max_points_per_domain", type=int, default=3000)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", default="output_dir/D_summary")
    return p.parse_args()


def resolve_best_combo(c_output_dir, best_summary_file):
    best_path = os.path.join(c_output_dir, best_summary_file)
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Cannot find best summary: {best_path}")
    best_df = pd.read_csv(best_path)
    if best_df.empty:
        raise ValueError(f"Empty best summary file: {best_path}")
    row = best_df.iloc[0].to_dict()

    c_best_dir = None
    if "ft_dir" in row and isinstance(row["ft_dir"], str) and row["ft_dir"].strip():
        c_best_dir = row["ft_dir"]
    elif "combo_dir" in row and isinstance(row["combo_dir"], str) and row["combo_dir"].strip():
        c_best_dir = row["combo_dir"]
    else:
        c_best_dir = c_output_dir
    if not os.path.isabs(c_best_dir):
        c_best_dir = os.path.join(c_output_dir, c_best_dir)
    if not os.path.isdir(c_best_dir):
        raise FileNotFoundError(f"Resolved best C directory not found: {c_best_dir}")
    return best_path, row, c_best_dir


def find_latent_files(a_best_dir):
    ccle_path = os.path.join(a_best_dir, "CCLE_latent_representation.pkl")
    tcga_path = os.path.join(a_best_dir, "TCGA_latent_representation.pkl")
    if not os.path.exists(ccle_path) or not os.path.exists(tcga_path):
        raise FileNotFoundError(
            "Cannot find A_pretrain latent files in best combo dir: "
            f"{a_best_dir}"
        )
    return ccle_path, tcga_path


def parse_latent_paths_from_c_log(c_best_dir, c_output_dir):
    candidate_logs = [os.path.join(c_best_dir, "C_training_log.txt")]
    cur = os.path.abspath(c_best_dir)
    root = os.path.abspath(c_output_dir)
    while True:
        candidate_logs.append(os.path.join(cur, "C_training_log.txt"))
        if cur == root:
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    candidate_logs.append(os.path.join(root, "C_training_log.txt"))
    log_path = None
    for path in candidate_logs:
        if os.path.exists(path):
            log_path = path
            break
    if log_path is None:
        raise FileNotFoundError(
            "Cannot find C training log in best C dir or its parent directories: "
            + ", ".join(dict.fromkeys(candidate_logs))
        )
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    ccle_match = re.findall(r"ccle_latent_pkl=(.+)", content)
    tcga_match = re.findall(r"tcga_latent_pkl=(.+)", content)
    ccle_path = ccle_match[-1].strip() if ccle_match else None
    tcga_path = tcga_match[-1].strip() if tcga_match else None
    if not ccle_path or not tcga_path:
        raise ValueError("Cannot parse ccle_latent_pkl/tcga_latent_pkl from C_training_log.txt")
    if not os.path.exists(ccle_path) or not os.path.exists(tcga_path):
        raise FileNotFoundError("Parsed latent paths from C log do not exist")
    return ccle_path, tcga_path


def normalize_latent_dict(latent_dict):
    ids = []
    vecs = []
    for k, v in latent_dict.items():
        ids.append(str(k))
        arr = np.asarray(v, dtype=float).reshape(-1)
        vecs.append(arr)
    if not vecs:
        return [], np.zeros((0, 0), dtype=float)
    dims = {x.shape[0] for x in vecs}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent latent dimensions found: {sorted(dims)}")
    return ids, np.vstack(vecs)


def copy_dir(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    best_path, best_row, c_best_dir = resolve_best_combo(args.c_output_dir, args.best_summary_file)
    copy_dir(c_best_dir, os.path.join(args.output_dir, "copied_best_C_prototypical"))
    ccle_latent_path, tcga_latent_path = parse_latent_paths_from_c_log(c_best_dir, args.c_output_dir)
    a_best_dir = os.path.dirname(ccle_latent_path)
    copy_dir(a_best_dir, os.path.join(args.output_dir, "copied_best_A_pretrain"))
    ccle_latent = load_pickle(ccle_latent_path)
    tcga_latent = load_pickle(tcga_latent_path)

    ccle_ids, ccle_z = normalize_latent_dict(ccle_latent)
    tcga_ids, tcga_z = normalize_latent_dict(tcga_latent)

    ccle_ct_map = read_cancer_map(args.ccle_cancer_type_input, args.sample_id_col, args.cancer_type_col)
    tcga_ct_map = read_cancer_map(args.tcga_cancer_type_input, args.sample_id_col, args.cancer_type_col)

    ccle_labels = [ccle_ct_map.get(x, "Unknown") for x in ccle_ids]
    tcga_labels = [tcga_ct_map.get(x, "Unknown") for x in tcga_ids]

    ccle_ids, ccle_z, ccle_labels = sample_points(
        ccle_ids, ccle_z, ccle_labels, args.max_points_per_domain, args.random_seed
    )
    tcga_ids, tcga_z, tcga_labels = sample_points(
        tcga_ids, tcga_z, tcga_labels, args.max_points_per_domain, args.random_seed
    )

    plot_tsne_with_cancer_type(
        source_ids=ccle_ids,
        source_z=ccle_z,
        source_labels=ccle_labels,
        target_ids=tcga_ids,
        target_z=tcga_z,
        target_labels=tcga_labels,
        save_path=os.path.join(args.output_dir, "best_latent_tsne_with_cancer_type.png"),
    )
    report_path = os.path.join(args.output_dir, "D_summary_report.csv")
    pd.DataFrame(
        [
            {
                "best_summary_file": best_path,
                "best_c_dir": c_best_dir,
                "latent_source_dir": a_best_dir,
                "ccle_latent_pkl": ccle_latent_path,
                "tcga_latent_pkl": tcga_latent_path,
                "tsne_plot": os.path.join(args.output_dir, "best_latent_tsne_with_cancer_type.png"),
                "ccle_points": len(ccle_ids),
                "tcga_points": len(tcga_ids),
            }
        ]
    ).to_csv(report_path, index=False)

    print(f"[D_summary] best summary: {best_path}")
    print(f"[D_summary] best C dir: {c_best_dir}")
    print(f"[D_summary] latent source dir: {a_best_dir}")
    print(f"[D_summary] copied A dir -> {os.path.join(args.output_dir, 'copied_best_A_pretrain')}")
    print(f"[D_summary] copied C dir -> {os.path.join(args.output_dir, 'copied_best_C_prototypical')}")
    print(f"[D_summary] t-SNE plot -> {os.path.join(args.output_dir, 'best_latent_tsne_with_cancer_type.png')}")
    print(f"[D_summary] report -> {report_path}")


if __name__ == "__main__":
    main()
