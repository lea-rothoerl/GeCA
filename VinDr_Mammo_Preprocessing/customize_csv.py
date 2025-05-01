import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
import ast

def filter_csv(input_csv, output_csv, columns, conditions, findings_flag, label_column, keep_split):
    """
    Filters a CSV file based on user-specified columns and conditions.
    
    Args:
        input-csv (str): Path to the input CSV file.
        output-csv (str): Path to save the filtered CSV file.
        --columns (list): List of column names to include in the output. All if not specified.
        --conditions (list): List of conditions to filter rows (e.g., "laterality=R").
        --findings (bool): Whether to generate finding-based filenames.
        --label-column (str): Column name for labels.
        --keep-split (bool): Whether to keep the original split in the CSV.
    """
        
    df = pd.read_csv(input_csv)

    # default to all columns if none are specified
    if columns is None:
        columns = list(df.columns)
    else:
        # ensure required columns are always included (but not duplicated)
        required = ['image_id', 'split']
        for req_col in required:
            if req_col not in columns:
                columns.append(req_col)

    # check for column existence
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns do not exist in the CSV: {missing_cols}")
        return

    # apply filtering conditions
    for condition in conditions:
        column, value = condition.split("=")
        column = column.strip()
        value = value.strip()

        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)

        if column == "finding_categories":
            df = df[df[column].astype(str).str.contains(value, regex=False, na=False)]
        else:
            df = df[df[column] == value]

    # only keep requested columns (including forced ones)
    df = df[columns]

    # add fold column
    df["fold"] = float("nan")

    # take care of rare labels
    label_series = df[label_column].dropna().astype(str)
    is_multilabel = label_series.str.contains(",").any()

    if is_multilabel:
        split_labels = label_series.str.split(",")
        flat_labels = [label.strip() for sublist in split_labels for label in sublist]
        label_counts = Counter(flat_labels)

        valid_labels = {label for label, count in label_counts.items() if count > 1}

        def row_has_only_valid_labels(row):
            labels = {label.strip() for label in str(row).split(",")}
            return all(label in valid_labels for label in labels)

        original_len = len(df)
        df = df[df[label_column].apply(row_has_only_valid_labels)]
        print(f"Removed {original_len - len(df)} rows with rare labels.")

    else:
        label_counts = label_series.value_counts()
        valid_labels = label_counts[label_counts > 1].index
        original_len = len(df)
        df = df[df[label_column].isin(valid_labels)]
        print(f"Removed {original_len - len(df)} rows with rare labels.")

    # train/test split if not keeping original split
    if not keep_split:
        if is_multilabel:
            print("Skipping stratified train/test split (multi-label case).")
            train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=0.25,
                stratify=df[label_column],
                random_state=42
            )

        train_df["split"] = "training"
        test_df["split"] = "test"
        df = pd.concat([train_df, test_df], ignore_index=True)

    # 5-fold cross-validation
    train_df = df[df["split"] == "training"].copy()
    train_df["fold"] = float("nan")

    if is_multilabel:
        print("Skipping stratified KFold (multi-label case), using random folds instead.")
        shuffled_idx = train_df.sample(frac=1, random_state=42).index
        fold_sizes = len(shuffled_idx) // 5
        for i in range(5):
            fold_idx = shuffled_idx[i * fold_sizes:] if i == 4 else shuffled_idx[i * fold_sizes:(i + 1) * fold_sizes]
            train_df.loc[fold_idx, "fold"] = i
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df[label_column])):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold_idx

    # merge everything back
    df = df.merge(train_df[["image_id", "split", "fold"]], on="image_id", how="left", suffixes=("", "_new"))
    df["split"] = df["split_new"].combine_first(df["split"])
    df["fold"] = df["fold_new"].combine_first(df["fold"])
    df.drop(columns=["split_new", "fold_new"], inplace=True)

    # hold out 5% as validation set
    val_subset = train_df.sample(frac=0.05, random_state=42)
    train_df.loc[val_subset.index, "split"] = "val"
    train_df.loc[val_subset.index, "fold"] = float("nan")

    # merge back updated training info
    df = df.merge(train_df[["image_id", "split", "fold"]], on="image_id", how="left", suffixes=("", "_new"))
    df["split"] = df["split_new"].combine_first(df["split"])
    df["fold"] = df["fold_new"].combine_first(df["fold"])
    df.drop(columns=["split_new", "fold_new"], inplace=True)

    # format image_id for findings if specified
    if findings_flag:
        df["image_id"] = df.apply(lambda row: f"{row['image_id']}_finding_{row['finding_idx']}.png", axis=1)
    else:
        df["image_id"] = df["image_id"] + ".png"

    # save to output file
    df.to_csv(output_csv, index=False)
    print(f"Customized CSV saved as {output_csv}")

    # print summary
    print("\nImage counts per split:")
    split_counts = df["split"].value_counts()
    for split, count in split_counts.items():
        print(f"  {split.capitalize()}: {count} images")

    # print number of unique labels
    if label_column in df.columns:
        labels_series = df[label_column].dropna().astype(str)

        all_labels = set()

        for entry in labels_series:
            try:
                parsed = ast.literal_eval(entry)
                if isinstance(parsed, list):
                    cleaned_labels = [label.strip().strip("'\"") for label in parsed]
                    all_labels.update(cleaned_labels)
                else:
                    # if not a list, add the entry as is
                    all_labels.add(str(parsed).strip().strip("'\""))
            except (ValueError, SyntaxError):
                split_labels = [label.strip().strip("'\"[]") for label in entry.split(",")]
                all_labels.update(split_labels)

        print(f"\nNumber of unique labels in '{label_column}': {len(all_labels)}")
        print("Unique labels:")
        for label in sorted(all_labels):
            print(f"  - {label}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom CSV from annotations.")
    parser.add_argument("input_csv", help="Path to the input CSV file (annotations.csv).")
    parser.add_argument("output_csv", help="Path to save the filtered CSV file.")
    parser.add_argument("--columns", nargs="+", help="List of columns to include.")
    parser.add_argument("--conditions", nargs="+", default=[], help="Filtering conditions (e.g., laterality=R view_position=CC).")
    parser.add_argument("--findings", action="store_true", help="Generate filenames for findings.")
    parser.add_argument("--label-column", default="finding_categories", help="Column name for labels.")
    parser.add_argument("--keep-split", action="store_true", help="Keep the original split in the CSV.")

    args = parser.parse_args()

    filter_csv(args.input_csv, args.output_csv, args.columns, args.conditions, args.findings, args.label_column, args.keep_split)
