import os
from glob import glob
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class MammoFullFieldDataset(Dataset):
    def __init__(self, root: str, mode: str = "train", transform=None, annotation_path="../../shared_data/VinDr_Mammo/finding_annotations.csv"):
        super(MammoFullFieldDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.mode = mode

        # get all PNG files in the folder
        print(f"Looking for images in: {os.path.join(root, mode, '*.png')}")
        self.image_paths = glob(os.path.join(root, mode, '*.png'))
        print(f"Found {len(self.image_paths)} images.")

        # load annotation file
        self.annotations = pd.read_csv(annotation_path)

        # create mapping of image_id to finding_categories
        self.label_dict = self._create_label_mapping()
        
        # create list of all unique labels
        self.all_labels = sorted(list(set([label for sublist in self.label_dict.values() for label in sublist])))
        self.label_to_index = {label: idx for idx, label in enumerate(self.all_labels)} 

        print(f"All unique labels: {self.all_labels}")

    def _create_label_mapping(self):
        """
        Creates a dictionary mapping image IDs to their corresponding labels.
        """
        label_dict = {}
        
        for _, row in self.annotations.iterrows():
            image_id = row["image_id"]  
            labels = eval(row["finding_categories"])  

            if isinstance(labels, list) and len(labels) > 0:
                label_dict[image_id] = labels  

        return label_dict
    
    def get_mapped_labels(self):
        """
        Returns a list of labels corresponding to the images in the dataset.
        """
        labels = []
        for img_path in self.image_paths:
            # extract image_id by removing ".png" from filename
            img_id = Path(img_path).stem.split('.png')[0]  
            
            # get the labels for image_id
            label_strings = self.label_dict.get(img_id, [])
            
            # create one-hot encoded label vector
            one_hot_label = [0] * len(self.all_labels) 
            for label in label_strings:
                if label in self.label_to_index: 
                    one_hot_label[self.label_to_index[label]] = 1  
            
            labels.append(one_hot_label)
        
        return labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # extract image ID from file name
        img_id = Path(img_path).stem

        # get the labels for the image_id
        label_strings = self.label_dict.get(img_id, [])
        
        # create the one-hot encoded label vector
        one_hot_label = [0] * len(self.all_labels)  
        for label in label_strings:
            if label in self.label_to_index: 
                one_hot_label[self.label_to_index[label]] = 1  

        # DEBUG: Print image ID and label vector
        #print(f"Image ID: {img_id}, One-hot Label: {one_hot_label}")

        return img, torch.tensor(one_hot_label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)
