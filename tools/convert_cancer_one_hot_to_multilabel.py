import argparse
import csv
import os
import shutil
from pathlib import Path


DEFAULT_INPUTS = (
    Path("data_Winnie/CCLE_cancer_one_hot.csv"),
    Path("data_Winnie/TCGA_cancer_one_hot.csv"),
)


def is_positive_one_hot_value(value):
    text = str(value).strip().lower()
    if text in ("", "0", "0.0", "false", "nan", "none"):
        return False

    try:
        return float(text) != 0.0
    except ValueError:
        return True


def label_name_from_column(column_name, prefix):
    if prefix and column_name.startswith(prefix):
        return column_name[len(prefix):]
    return column_name


def default_output_path(input_path):
    input_path = Path(input_path)
    if input_path.name.endswith("_one_hot.csv"):
        return input_path.with_name(input_path.name.replace("_one_hot.csv", "_multi_label.csv"))
    return input_path.with_name(f"{input_path.stem}_multi_label{input_path.suffix}")


def backup_path_for(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}.one_hot_backup{input_path.suffix}")


def convert_one_hot_to_multilabel(
    input_path,
    output_path,
    id_column,
    label_prefix,
    output_label_column,
    multilabel_delimiter,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is empty or has no header")

        if id_column not in reader.fieldnames:
            raise ValueError(f"{input_path} does not contain id column: {id_column}")

        if reader.fieldnames == [id_column, output_label_column] and input_path == output_path:
            return {
                "input": str(input_path),
                "output": str(output_path),
                "rows": sum(1 for _ in reader),
                "label_columns": 1,
                "rows_without_label": 0,
                "rows_with_multiple_labels": 0,
                "max_labels_per_row": None,
                "already_multilabel": True,
            }

        label_columns = [
            column
            for column in reader.fieldnames
            if column != id_column
            and column != output_label_column
            and (not label_prefix or column.startswith(label_prefix))
        ]
        if not label_columns:
            raise ValueError(f"{input_path} does not contain label columns with prefix: {label_prefix}")

        temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        row_count = 0
        no_label_count = 0
        multi_label_count = 0
        max_labels_per_row = 0

        with temp_output_path.open("w", newline="") as target_file:
            writer = csv.DictWriter(target_file, fieldnames=[id_column, output_label_column])
            writer.writeheader()

            for row in reader:
                labels = [
                    label_name_from_column(column, label_prefix)
                    for column in label_columns
                    if is_positive_one_hot_value(row.get(column, ""))
                ]
                row_count += 1
                no_label_count += int(len(labels) == 0)
                multi_label_count += int(len(labels) > 1)
                max_labels_per_row = max(max_labels_per_row, len(labels))

                writer.writerow(
                    {
                        id_column: row[id_column],
                        output_label_column: multilabel_delimiter.join(labels),
                    }
                )

        os.replace(temp_output_path, output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": row_count,
        "label_columns": len(label_columns),
        "rows_without_label": no_label_count,
        "rows_with_multiple_labels": multi_label_count,
        "max_labels_per_row": max_labels_per_row,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert cancer one-hot CSV files into two-column multi-label CSV files: "
            "Sample_ID plus one Cancer_type column."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Input one-hot CSV files. Defaults to the CCLE and TCGA data_Winnie files.",
    )
    parser.add_argument("--id-column", default="Sample_ID", help="Sample identifier column name.")
    parser.add_argument("--label-prefix", default="Cancer_type", help="Prefix used by one-hot label columns.")
    parser.add_argument("--output-label-column", default="Cancer_type", help="Name of the output label column.")
    parser.add_argument(
        "--delimiter",
        default=";",
        help="Delimiter used when one sample has multiple positive cancer labels.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file instead of writing *_multi_label.csv files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create *.one_hot_backup.csv before in-place overwrite.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for input_path in args.inputs:
        output_path = input_path if args.in_place else default_output_path(input_path)

        if args.in_place and not args.no_backup:
            backup_path = backup_path_for(input_path)
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)

        summary = convert_one_hot_to_multilabel(
            input_path=input_path,
            output_path=output_path,
            id_column=args.id_column,
            label_prefix=args.label_prefix,
            output_label_column=args.output_label_column,
            multilabel_delimiter=args.delimiter,
        )
        print(summary)


if __name__ == "__main__":
    main()
