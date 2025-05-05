import pandas as pd
import ast
import argparse

def count_labels_per_split(csv_path, label_column="finding_categories"):
    df = pd.read_csv(csv_path)

    splits = ["training", "test", "val"]

    if label_column not in df.columns or "split" not in df.columns:
        print(f"Error: CSV must have '{label_column}' and 'split' columns.")
        return

    labels_series = df[label_column].dropna().astype(str)

    # collect all unique labels
    all_labels = set()
    for entry in labels_series:
        try:
            parsed = ast.literal_eval(entry)
            if isinstance(parsed, list):
                cleaned_labels = [label.strip().strip("'\"") for label in parsed]
                all_labels.update(cleaned_labels)
            else:
                all_labels.add(str(parsed).strip().strip("'\""))
        except (ValueError, SyntaxError):
            split_labels = [label.strip().strip("'\"[]") for label in entry.split(",")]
            all_labels.update(split_labels)

    print("\nImage counts per split per label:")
    for label in sorted(all_labels):
        counts = {}
        for split in splits:
            split_df = df[df["split"] == split]

            if split_df.empty:
                counts[split] = 0
                continue

            label_count = 0
            for _, row in split_df.iterrows():
                try:
                    parsed = ast.literal_eval(str(row[label_column]))
                    if isinstance(parsed, list):
                        cleaned_labels = [l.strip().strip("'\"") for l in parsed]
                    else:
                        cleaned_labels = [str(parsed).strip().strip("'\"")]
                except (ValueError, SyntaxError):
                    cleaned_labels = [l.strip().strip("'\"[]") for l in str(row[label_column]).split(",")]

                if label in cleaned_labels:
                    label_count += 1

            counts[split] = label_count

        total = sum(counts.values())
        print(f"  {label}: Training: {counts.get('training', 0)}, Test: {counts.get('test', 0)}, "
              f"Val: {counts.get('val', 0)}, Total: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count images per split per label from CSV.")
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument("--label-column", default="finding_categories", help="Column name for labels.")
    args = parser.parse_args()

    count_labels_per_split(args.csv_path, args.label_column)
