import pandas as pd
import argparse

def filter_csv(input_csv, output_csv, columns, conditions):
    """
    Filters a CSV file based on user-specified columns and conditions.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered CSV file.
        columns (list): List of column names to include in the output.
        conditions (list): List of conditions to filter rows (e.g., "laterality=R").
    """
    df = pd.read_csv(input_csv)

    # keep requested cols
    if columns:
        df = df[columns]

    # filter on conditions
    for condition in conditions:
        column, value = condition.split("=")
        column = column.strip()
        value = value.strip()

        # numeric to avoid issues
        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        if column == "finding_categories":
            df = df[df[column].astype(str).str.contains(value, regex=False, na=False)]
        else:
            df = df[df[column] == value]

    # save result
    df.to_csv(output_csv, index=False)
    print(f"Customized CSV saved as {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom CSV from annotations.")
    parser.add_argument("input_csv", help="Path to the input CSV file (annotations.csv).")
    parser.add_argument("output_csv", help="Path to save the filtered CSV file.")
    parser.add_argument("--columns", nargs="+", help="List of columns to include.")
    parser.add_argument("--conditions", nargs="+", help="List of filtering conditions (e.g., laterality=R view_position=CC).")

    args = parser.parse_args()

    filter_csv(args.input_csv, args.output_csv, args.columns, args.conditions if args.conditions else [])
