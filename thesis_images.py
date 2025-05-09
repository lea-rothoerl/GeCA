import os
import random
import argparse
import pandas as pd
from PIL import Image

def create_sample_grid(input_folder, output_path, spacing=4, csv_path=None):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]

    if csv_path:
        df = pd.read_csv(csv_path)
        csv_image_ids = set(df["image_id"])
        image_files = [f for f in image_files if f in csv_image_ids]

    sample_files = random.sample(image_files, 4)
    print("Selected images:", sample_files)

    images = [Image.open(os.path.join(input_folder, f)) for f in sample_files]

    img_size = 128
    num_images = 4

    combined_width = img_size * num_images + spacing * (num_images - 1)
    combined_height = img_size
    combined_image = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))

    for i, img in enumerate(images):
        x_offset = i * (img_size + spacing)
        combined_image.paste(img, (x_offset, 0))

    combined_image.save(output_path)
    print(f"Saved sample grid as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a horizontal sample grid of 4 random 128x128 images with optional CSV filtering.")
    parser.add_argument("input_folder", help="Folder containing PNG images.")
    parser.add_argument("output_path", help="Output PNG file path.")
    parser.add_argument("--csv", help="Optional CSV file with 'image_id' column to filter images.", default=None)
    args = parser.parse_args()

    create_sample_grid(args.input_folder, args.output_path, csv_path=args.csv)
