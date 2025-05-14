import pandas as pd
import argparse
import os
import random

def build_mixed_csv(reference_csv, 
                    synth_annotation_csv, 
                    output_csv, 
                    label_column="finding_categories",
                    ratio=None):

    # load reference CSV
    ref_df = pd.read_csv(reference_csv)
    print(f"Loaded reference CSV with {len(ref_df)} entries.")

    # load synthetic annotations
    synth_df = pd.read_csv(synth_annotation_csv)
    synth_df = synth_df.loc[:, ~synth_df.columns.str.contains('^Unnamed')]
    print(f"Loaded synthetic annotation CSV with {len(synth_df)} entries.")

    synth_base_path = os.path.dirname(os.path.abspath(synth_annotation_csv))

    # identify label columns
    known_cols = {"client_id", "filename", "patient_name", "multi_class_label"}
    label_cols = [col for col in synth_df.columns if col not in known_cols]

    # convert one-hot encoded labels to label name
    def get_label(row):
        active_labels = [label for label in label_cols if row[label] == 1]
        if label_column == "finding_categories":
            # For finding_categories: make it a stringified list like ['Mass']
            if len(active_labels) == 0:
                return "['No Finding']"
            else:
                return str(active_labels)
        else:
            # For other label columns: return single label or comma-separated string
            if len(active_labels) == 0:
                return "No Finding"
            elif len(active_labels) == 1:
                return active_labels[0]
            else:
                return ",".join(active_labels) 
        
    synth_df[label_column] = synth_df.apply(get_label, axis=1)

    # prepare synthetic samples DataFrame
    synth_samples = pd.DataFrame({
        "image_id": synth_df["filename"].apply(lambda x: os.path.join(synth_base_path, x)),
        "model": "Synthetic",
        label_column: synth_df[label_column],
        "split": "training",
        "fold": -1 
    })

    # calculate number of synthetic samples based on ratio if given
    if ratio is None:
        num_synth_needed = len(synth_samples)
    else:
        num_real_images = len(ref_df)
        num_synth_needed = int(num_real_images * ratio)

    # verify number of images and sample
    available_synth = len(synth_samples)
    if num_synth_needed > available_synth:
        print(f"WARNING: Requested {num_synth_needed} synthetic images, but only {available_synth} available.")
        num_synth_needed = available_synth

    synth_samples = synth_samples.sample(n=num_synth_needed, random_state=42)

    mixed_df = pd.concat([ref_df, synth_samples], ignore_index=True)
    print(f"Extended training set by {len(synth_samples)} synthetic samples.")

    # save result
    mixed_df.to_csv(output_csv, index=False)
    print(f"Mixed CSV saved to {output_csv}")
    print("\nImage counts per split:")
    print(mixed_df["split"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", required=True, help="Path to the reference CSV file with real images.")
    parser.add_argument("--synth-annotation-csv", required=True, help="Path to synthetic images annotation CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to save the mixed CSV file.")
    parser.add_argument("--label-column", type=str, default="breast_density")
    parser.add_argument("--ratio", type=float, default=None, help="Ratio of synthetic to real images in the mixed dataset. If none, all synthetic images will be used.")

    args = parser.parse_args()

    build_mixed_csv(
        reference_csv=args.reference_csv,
        synth_annotation_csv=args.synth_annotation_csv,
        output_csv=args.output_csv,
        label_column=args.label_column,
        ratio=args.ratio
    )
