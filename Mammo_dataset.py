import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class MammoDataset(Dataset):
    def __init__(self, root: str, annotation_csv: str, transform=None):
        """
        Args:
            root (str): Path to the folder containing images.
            annotation_csv (str): Path to the CSV file containing image metadata and labels.
            transform: Optional image transformations.
        """
        super(MammoDataset, self).__init__()
        self.root = root
        self.transform = transform

        # get annotation csv
        self.annotations = pd.read_csv(annotation_csv)

        # create full image paths based on CSV image_id column
        self.image_paths = [
            os.path.join(root, f"{image_id}.png") for image_id in self.annotations["image_id"]
        ]
        
        # some debugging
        self.annotations = self.annotations[self.annotations["image_id"].apply(lambda x: os.path.exists(os.path.join(root, f"{x}.png")))]
        self.image_paths = [os.path.join(root, f"{image_id}.png") for image_id in self.annotations["image_id"]]
        
        # mapping labels
        self.label_dict = self._create_label_mapping()
        self.all_labels = sorted(set(label for sublist in self.label_dict.values() for label in sublist))
        self.label_to_index = {label: idx for idx, label in enumerate(self.all_labels)}

        print(f"Total images: {len(self.image_paths)}")
        print(f"All unique labels: {self.all_labels}")

    def _create_label_mapping(self):
        """Maps image IDs to their corresponding labels."""
        label_dict = {}
        for _, row in self.annotations.iterrows():
            image_id = row["image_id"]
            labels = eval(row["finding_categories"]) if isinstance(row["finding_categories"], str) else []
            label_dict[image_id] = labels
        return label_dict

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # get image ID
        img_id = Path(img_path).stem

        # get one-hot encoded labels
        label_strings = self.label_dict.get(img_id, [])
        one_hot_label = [0] * len(self.all_labels)
        for label in label_strings:
            if label in self.label_to_index:
                one_hot_label[self.label_to_index[label]] = 1  

        return img, torch.tensor(one_hot_label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)
