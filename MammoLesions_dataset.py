import os
from glob import glob
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class MammoLesionsDataset(Dataset):
    def __init__(self, root: str, mode: str = "train", transform=None, annotation_path="../../shared_data/VinDr_Mammo/finding_annotations.csv"):
        super(MammoLesionsDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.mode = mode

        # Get all PNG files in the folder
        print(f"Looking for images in: {os.path.join(root, mode, '*.png')}")
        self.image_paths = glob(os.path.join(root, mode, '*.png'))
        print(f"Found {len(self.image_paths)} images.")

        # Load the annotations
        self.annotations = pd.read_csv(annotation_path)

        # Create a mapping of image_id to finding_categories
        self.label_dict = self._create_label_mapping()
        
        # Create a list of all unique labels
        self.all_labels = sorted(list(set([label for sublist in self.label_dict.values() for label in sublist])))
        self.label_to_index = {label: idx for idx, label in enumerate(self.all_labels)}  # Label to index mapping

        print(f"All unique labels: {self.all_labels}")

    def _create_label_mapping(self):
        """
        Creates a dictionary mapping image IDs to their corresponding labels.
        """
        label_dict = {}
        
        for _, row in self.annotations.iterrows():
            image_id = row["image_id"]  # Extract image ID
            labels = eval(row["finding_categories"])  # Convert string list to Python list

            if isinstance(labels, list) and len(labels) > 0:
                label_dict[image_id] = labels  # Store labels as a list

        return label_dict
    
    def get_mapped_labels(self):
        """
        Returns a list of labels corresponding to the images in the dataset.
        """
        labels = []
        for img_path in self.image_paths:
            # Extract the image_id by removing "_lesion_{idx}" suffix from filename
            img_id = Path(img_path).stem.split('_lesion')[0]  
            
            # Get the labels for the image_id
            label_strings = self.label_dict.get(img_id, [])
            
            # Create the one-hot encoded label vector
            one_hot_label = [0] * len(self.all_labels)  # Initialize a vector of zeros
            for label in label_strings:
                if label in self.label_to_index:  # Ensure label is in our defined set
                    one_hot_label[self.label_to_index[label]] = 1  # Set the corresponding index to 1
            
            labels.append(one_hot_label)
        
        return labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # Extract the image_id by removing "_lesion_{idx}" suffix from filename
        img_id = Path(img_path).stem.split('_lesion')[0]  

        # Get the labels for the image_id
        label_strings = self.label_dict.get(img_id, [])
        
        # Create the one-hot encoded label vector
        one_hot_label = [0] * len(self.all_labels)  # Initialize a vector of zeros
        for label in label_strings:
            if label in self.label_to_index:  # Ensure label is in our defined set
                one_hot_label[self.label_to_index[label]] = 1  # Set the corresponding index to 1

        # DEBUG: Print image ID and label vector
        print(f"Image ID: {img_id}, One-hot Label: {one_hot_label}")

        return img, torch.tensor(one_hot_label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)
