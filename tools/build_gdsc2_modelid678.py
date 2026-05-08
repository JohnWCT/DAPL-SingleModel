import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter GDSC2 dose response rows by ModelID intersection "
            "with CCLE sample index IDs and pretrain CCLE IDs."
        )
    )
    parser.add_argument("--gdsc", type=Path, required=True, help="Input GDSC2 csv path")
    parser.add_argument("--ccle-info", type=Path, required=True, help="ccle_sample_info_df.csv path")
    parser.add_argument("--pretrain-ccle", type=Path, required=True, help="pretrain_ccle.csv path")
    parser.add_argument("--output", type=Path, required=True, help="Output csv path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gdsc_df = pd.read_csv(args.gdsc)
    ccle_info_df = pd.read_csv(args.ccle_info, index_col=0)
    pretrain_df = pd.read_csv(args.pretrain_ccle)

    if "ModelID" not in gdsc_df.columns:
        raise ValueError("Input GDSC2 file does not contain 'ModelID' column.")
    if pretrain_df.shape[1] == 0:
        raise ValueError("pretrain_ccle.csv has no columns.")

    gdsc_model_ids = set(gdsc_df["ModelID"].dropna().astype(str))
    ccle_index_ids = set(ccle_info_df.index.astype(str))
    pretrain_ids = set(pretrain_df.iloc[:, 0].dropna().astype(str))

    model_ids_678 = gdsc_model_ids & ccle_index_ids & pretrain_ids

    filtered_df = gdsc_df[gdsc_df["ModelID"].astype(str).isin(model_ids_678)].copy()
    filtered_model_count = filtered_df["ModelID"].nunique()
    filtered_row_count = len(filtered_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output, index=False)

    print(f"GDSC unique ModelID: {len(gdsc_model_ids)}")
    print(f"CCLE sample index IDs: {len(ccle_index_ids)}")
    print(f"pretrain CCLE IDs: {len(pretrain_ids)}")
    print(f"ModelID intersection count: {len(model_ids_678)}")
    print(f"Output unique ModelID: {filtered_model_count}")
    print(f"Output rows: {filtered_row_count}")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
