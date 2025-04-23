import pandas as pd
import argparse
import os
import random

def build_mixed_csv(reference_csv, 
                    synth_annotation_csv, 
                    output_csv, 
                    mode="extend", 
                    label_column="finding_categories"
                    ):
    
    assert mode in ["extend", "replace"], "Mode must be 'extend' or 'replace'"

    # load reference CSV
    ref_df = pd.read_csv(reference_csv)
    # DEBUG
    print(f"Loaded reference CSV with {len(ref_df)} entries.")

    # load synthetic annotations
    synth_df = pd.read_csv(synth_annotation_csv)
    # DEBUG
    print(f"Loaded synthetic annotation CSV with {len(synth_df)} entries.")

    # identify label columns
    known_cols = {"client_id", "filename", "patient_name", "multi_class_label"}
    label_cols = [col for col in synth_df.columns if col not in known_cols]
    # DEBUG
    print(f"Detected label columns: {label_cols}")

    # convert one-hot encoded labels to label name
    def get_label(row):
        active_labels = [label for label in label_cols if row[label] == 1]
        if len(active_labels) == 0:
            return "No_Label"
        elif len(active_labels) == 1:
            return active_labels[0]
        else:
            return ",".join(active_labels)  
        
    synth_df[label_column] = synth_df.apply(get_label, axis=1)

    # prepare synthetic samples DataFrame
    synth_samples = pd.DataFrame({
        "image_id": synth_df["filename"].apply(lambda x: os.path.basename(x)),
        "model": "Synthetic",
        label_column: synth_df[label_column],
        "split": "training",
        "fold": -1 
    })

    if mode == "extend":
        mixed_df = pd.concat([ref_df, synth_samples], ignore_index=True)
        print(f"Extended training set by {len(synth_samples)} synthetic samples.")
    elif mode == "replace":
        real_train_idx = ref_df[ref_df["split"] == "training"].index.tolist()
        n_replace = min(len(real_train_idx), len(synth_samples))
        replace_idx = random.sample(real_train_idx, n_replace)

        # drop selected real samples
        ref_df.drop(index=replace_idx, inplace=True)
        print(f"Replaced {n_replace} real training samples with synthetic ones.")

        # add synthetic samples
        mixed_df = pd.concat([ref_df, synth_samples.iloc[:n_replace]], ignore_index=True)

    # save result
    mixed_df.to_csv(output_csv, index=False)
    print(f"Mixed CSV saved to {output_csv}")
    print("\nImage counts per split:")
    print(mixed_df["split"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixed CSV dataset combining real and synthetic images.")
    parser.add_argument("--reference-csv", required=True, help="Path to the reference CSV file with real images.")
    parser.add_argument("--synth-annotation-csv", required=True, help="Path to synthetic images annotation CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to save the mixed CSV file.")
    parser.add_argument("--mode", choices=["extend", "replace"], default="extend", help="Whether to extend or replace real training samples.")
    parser.add_argument("--label-column", type=str, default="finding_categories")

    args = parser.parse_args()

    build_mixed_csv(
        reference_csv=args.reference_csv,
        synth_annotation_csv=args.synth_annotation_csv,
        output_csv=args.output_csv,
        mode=args.mode,
        label_column=args.label_column
    )
