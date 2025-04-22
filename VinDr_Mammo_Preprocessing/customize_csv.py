import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

def filter_csv(input_csv, output_csv, columns, conditions, findings_flag):
    """
    Filters a CSV file based on user-specified columns and conditions.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered CSV file.
        columns (list): List of column names to include in the output. All if not specified.
        conditions (list): List of conditions to filter rows (e.g., "laterality=R").
        findings (bool): Whether to generate finding-based filenames.
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

    # stratified 5-fold split
    train_df = df[df["split"] == "training"].copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df[args.label_column])):
        train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold_idx

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
    if args.label_column in df.columns:
        labels_series = df[args.label_column].dropna().astype(str)

        # comma-separated multi-label entries
        split_labels = labels_series.str.split(",")
        flat_labels = [label.strip() for sublist in split_labels for label in sublist]
        unique_labels = set(flat_labels)

        print(f"\nNumber of unique labels in '{args.label_column}': {len(unique_labels)}")
        print(f"Unique labels: {sorted(unique_labels)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom CSV from annotations.")
    parser.add_argument("input_csv", help="Path to the input CSV file (annotations.csv).")
    parser.add_argument("output_csv", help="Path to save the filtered CSV file.")
    parser.add_argument("--columns", nargs="+", help="List of columns to include.")
    parser.add_argument("--conditions", nargs="+", default=[], help="Filtering conditions (e.g., laterality=R view_position=CC).")
    parser.add_argument("--findings", action="store_true", help="Generate filenames for findings.")
    parser.add_argument("--label-column", default="finding_categories", help="Column name for labels.")

    args = parser.parse_args()

    filter_csv(args.input_csv, args.output_csv, args.columns, args.conditions, args.findings)

#python3 ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv ../findings/Mammomat_Mass.csv --columns image_id finding_categories xmin ymin xmax ymax finding_idx model split --conditions model=Mammomat\ Inspiration finding_categories=Mass --findings