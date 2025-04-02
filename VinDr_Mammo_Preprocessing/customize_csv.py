import pandas as pd
import argparse
import os

def filter_csv(input_csv, output_csv, columns, conditions, findings_flag):
    """
    Filters a CSV file based on user-specified columns and conditions.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered CSV file.
        columns (list): List of column names to include in the output.
        conditions (list): List of conditions to filter rows (e.g., "laterality=R").
        findings_flag (bool): Whether to generate lesion-based filenames.
    """
    df = pd.read_csv(input_csv)

    # ensure the required columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns do not exist in the CSV: {missing_cols}")
        return
    
    # filter on conditions
    for condition in conditions:
        column, value = condition.split("=")
        column = column.strip()
        value = value.strip()

        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        elif value.startswith("[") and value.endswith("]"):
            df = df[df[column].str.contains(value.strip("[]"), regex=False, na=False)]
        else:
            df = df[df[column] == value]

    # keep only requested columns
    df = df[columns]

    # if --findings flag is set, modify filenames
    if findings_flag:
        lesion_filenames = []
        for _, row in df.iterrows():
            image_id = row["image_id"]
            finding_idx = row["finding_idx"]
            lesion_filenames.extend(f"{image_id}_lesion_{finding_idx}.png")

        lesion_df = pd.DataFrame({"lesion_filename": lesion_filenames})
        df = pd.concat([df.reset_index(drop=True), lesion_df], axis=1)

    # save customized CSV
    df.to_csv(output_csv, index=False)
    print(f"Customized CSV saved as {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom CSV from annotations.")
    parser.add_argument("input_csv", help="Path to the input CSV file (annotations.csv).")
    parser.add_argument("output_csv", help="Path to save the filtered CSV file.")
    parser.add_argument("--columns", nargs="+", help="List of columns to include.")
    parser.add_argument("--conditions", nargs="+", default=[], help="Filtering conditions (e.g., laterality=R view_position=CC).")
    parser.add_argument("--findings", action="store_true", help="Generate filenames for findings.")

    args = parser.parse_args()

    filter_csv(args.input_csv, args.output_csv, args.columns, args.conditions, args.findings)
