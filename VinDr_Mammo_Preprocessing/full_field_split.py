import os
import pandas as pd
import shutil
import argparse

parser = argparse.ArgumentParser(description="Sort full field images into subfolders based on CSV annotations.")
parser.add_argument("png_dir", help="Path to the PNG directory.")
parser.add_argument("meta_path", help="Path to the metadata CSV file.")
parser.add_argument("breastlevel_path", help="Path to the breast level annotation CSV file.")
parser.add_argument("output_dir", help="Path to the output directory where sorted images will be stored.")

args = parser.parse_args()

png_dir = args.png_dir
meta_path = args.meta_path
breastlevel_path = args.breastlevel_path
output_dir = args.output_dir

# load CSV files
meta_df = pd.read_csv(meta_path)
breastlevel_df = pd.read_csv(breastlevel_path)

# dictionary mapping image_id to manufacturer model name
meta_dict = dict(zip(meta_df["SOP Instance UID"], meta_df["Manufacturer's Model Name"]))

# dictionary mapping image_id to laterality, view position, and split type
breastlevel_dict = breastlevel_df.set_index("image_id")[["laterality", "view_position", "split"]].to_dict(orient="index")

# create directories and subfolders
for model in meta_df["Manufacturer's Model Name"].unique():
    model_dir = os.path.join(output_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    for _, row in breastlevel_df.iterrows():
        laterality = row["laterality"]
        view_position = row["view_position"]
        subfolder = f"{laterality}_{view_position}"

        subfolder_path = os.path.join(model_dir, subfolder)
        os.makedirs(os.path.join(subfolder_path, "training"), exist_ok=True)
        os.makedirs(os.path.join(subfolder_path, "test"), exist_ok=True)

# move images to their folders
for filename in os.listdir(png_dir):
    if filename.endswith(".png"):
        image_id = filename.split(".png")[0]

        if image_id in meta_dict and image_id in breastlevel_dict:
            model_name = meta_dict[image_id]
            laterality = breastlevel_dict[image_id]["laterality"]
            view_position = breastlevel_dict[image_id]["view_position"]
            split = breastlevel_dict[image_id]["split"]

            dest_folder = os.path.join(output_dir, model_name, f"{laterality}_{view_position}", split)
            os.makedirs(dest_folder, exist_ok=True)

            src_path = os.path.join(png_dir, filename)
            dest_path = os.path.join(dest_folder, filename)

            shutil.move(src_path, dest_path)
            print(f"Moved {filename} to {dest_folder}/")

print("Finished sorting full field images.")
