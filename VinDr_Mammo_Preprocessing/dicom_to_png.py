import pydicom
import numpy as np
from PIL import Image
import os
import shutil
import argparse
import pandas as pd

def crop_borders(image_array, threshold=10):
    """
    Crop the image to the smallest rectangle containing all pixels above a threshold,
    ignoring corners containing patient informartion according to documentation.
    
    Parameters:
    - image_array: The normalized image array.
    - threshold: Minimum intensity to be considered non-black.
    
    Returns:
    - Cropped image array.
    """
    white_threshold = 150
    
    # black out info
    rem_info = image_array.copy()
    rem_info[rem_info > white_threshold] = 0

    # binary mask for above and below threshold
    mask = rem_info > threshold
    
    # catch problems with empty mask
    if not mask.any():
        return image_array

    # collect coordinates of the non-black pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # crop to these boundaries
    return image_array[y0:y1+1, x0:x1+1]

def resize_with_padding(image, target_size=(256, 256)):
    """
    Resize image and pad to target_size.
    
    Parameters:
    - image: A PIL Image.
    - target_size: Tuple (width, height) for desired output size.
    
    Returns:
    - A new padded PIL Image of size target_size.
    """
    original_size = image.size  # (width, height)
    target_width, target_height = target_size

    # get scale factor and new size to fit
    scale = min(target_width / original_size[0], target_height / original_size[1])
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    
    # resize
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # create image with desired size
    new_image = Image.new("L", target_size, 0)
    
    # add cropped image to padding "passpartout"
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    new_image.paste(resized_image, paste_position)
    
    return new_image

def dicom_to_png(dicom_path, output_path, target_size=(256, 256), apply_resize=True):
    """
    Convert DICOM image to PNG and crop it to a desired target size.
    
    If apply_resize is True, the cropped image is padded to fit the target_size.
    Otherwise, the cropped image is saved as is.
    """
    try:
        dicom_image = pydicom.dcmread(dicom_path)

        # handle problematic images
        if not hasattr(dicom_image, "pixel_array"):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return

        image_array = dicom_image.pixel_array

        # normalize pixel values to 0-255
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
        image_array = image_array.astype(np.uint8)

        # crop the image 
        cropped_array = crop_borders(image_array)

        # convert to a PIL
        image = Image.fromarray(cropped_array)

        # if desired, resize with black padding to target size
        if apply_resize:
            image = resize_with_padding(image, target_size=target_size)

        # save the output PNG
        image_id = os.path.splitext(os.path.basename(dicom_path))[0]
        filename = f"{image_id}.png"
        out_path = os.path.join(output_path, filename)
        image.save(out_path)
        print(f"Processed: {dicom_path} → {out_path}")

    # catch exceptions
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def extract_findings(dicom_path, annotations_df, output_path, target_size=(256, 256), apply_resize=True):
    """
    Extract finding regions from a DICOM image based on bounding boxes provided in annotations_df.
    
    The CSV (finding_annotations.csv) must include at least the following columns:
      image_id, study_id, xmin, ymin, xmax, ymax
    Each finding is saved as a separate PNG in output_path/<study_id>/findings/.
    """
    try:
        dicom_image = pydicom.dcmread(dicom_path)
        if not hasattr(dicom_image, "pixel_array"):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return

        # normalization
        image_array = dicom_image.pixel_array
        image_array = (image_array - np.min(image_array)) / (np.ptp(image_array)) * 255.0
        image_array = image_array.astype(np.uint8)

        # get image ID from the filename
        image_id = os.path.splitext(os.path.basename(dicom_path))[0]
        # get annotations for respective ID
        finding_rows = annotations_df[annotations_df['image_id'] == image_id]

        # process each single finding
        for idx, row in finding_rows.iterrows():
            # handle images without findings
            if pd.isnull(row[['xmin', 'ymin', 'xmax', 'ymax']]).any():
                continue 
        
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])

            # extract findings using annotation bounding boxes
            finding_region = image_array[ymin:ymax, xmin:xmax]
            finding_img = Image.fromarray(finding_region)

            if apply_resize:
                finding_img = resize_with_padding(finding_img, target_size=target_size)

            finding_filename = f"{image_id}_finding_{idx}.png"
            finding_out_path = os.path.join(output_path, finding_filename)
            finding_img.save(finding_out_path)
            print(f"Extracted finding: {finding_out_path}")

    # catch exceptions
    except Exception as e:
        print(f"Error extracting findings from {dicom_path}: {e}")

def process_dicom_folder(input_root, output_path, target_size=(256, 256), apply_resize=False, findings_flag=False, annotations_df=None):
    """
    Process all DICOM images in subfolders, converting them to PNG.
    
    If apply_resize is True, each image is resized (with padding) to target_size.
    If findings_flag is False, converts full DICOM images to PNG (cropped/resized as specified).
    If True, extracts finding regions based on bounding boxes from annotations_df.

    Also copies index.html files.
    """
    for subdir, _, files in os.walk(input_root):
        for file in files:
            input_file_path = os.path.join(subdir, file)
            if file.lower().endswith(".dicom"):
                if findings_flag:
                    extract_findings(input_file_path, annotations_df, output_path, target_size=target_size, apply_resize=apply_resize)
                else: 
                    dicom_to_png(input_file_path, output_path, target_size=target_size, apply_resize=apply_resize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DICOM images to PNG with cropping/resizing."
    )
    parser.add_argument("in_folder", help="Path to the input folder containing DICOM images.")
    parser.add_argument("out_folder", help="Path to the output folder for PNG images.")
    parser.add_argument("--resize", action="store_true", 
                        help="Apply resizing with padding to a uniform target size.")
    parser.add_argument("--findings", action="store_true", 
                        help="Extract findings based on finding_annotations.csv.")    
    parser.add_argument("--annotations", type=str, default=None, 
                        help="Path to the CSV file for finding annotations.")
    parser.add_argument("--target-size", type=str, default=(256, 256),
                        help="Target size for resizing images (width, height).")
    
    args = parser.parse_args()

    # load annotation CSV for finding extraction
    annotations_df = None
    if args.findings:
        try:
            annotations_df = pd.read_csv(args.annotations) 
            print("Loaded finding_annotations.csv")
        except Exception as e:
            print(f"Error loading finding_annotations.csv: {e}")
            exit(1)

    process_dicom_folder(args.in_folder, args.out_folder,
                         apply_resize=args.resize,
                         findings_flag=args.findings,
                         annotations_df=annotations_df)

    print("Done!")
