import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class MammoDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 annotation_path: str, 
                 transform=None, 
                 mode='training', 
                 label_column='finding_categories',
                 single_label=False):
        """
        Args:
            root (str): Path to the folder containing images.
            annotation_path (str): Path to the CSV file containing image metadata and labels.
            transform: Optional image transformations.
            mode (str): The dataset mode ('training', 'test', or 'val'). Filters data accordingly.
            label_column (str): Name of the column to be used for labels (default: 'finding_categories').
            single_label (bool): If True, assumes each image has a single label. If False, assumes multiple labels.
        """
        super(MammoDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.label_column = label_column
        self.mode = mode
        self.single_label = single_label

        # get annotation csv while checking if images are synthetic (path as ID)
        def resolve_path(image_id):
            if os.path.isabs(image_id):
                return image_id
            else:
                return os.path.join(root, image_id)

        self.annotations = pd.read_csv(annotation_path)
        self.annotations["resolved_path"] = self.annotations["image_id"].apply(resolve_path)
        self.annotations = self.annotations[self.annotations["resolved_path"].apply(os.path.exists)]
        self.image_paths = self.annotations["resolved_path"].tolist()

        # filter dataset by mode (split)
        if mode not in ["training", "test", "val"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'training', 'test', or 'val'.")
    
        self.annotations["split"] = self.annotations["split"].astype(str).str.lower().str.strip()
        self.annotations = self.annotations[self.annotations["split"] == mode.lower()]
        
        # some debugging
        self.annotations = self.annotations[self.annotations["image_id"].apply(lambda x: os.path.exists(os.path.join(root, f"{x}")))]
        self.image_paths = [os.path.join(root, f"{image_id}") for image_id in self.annotations["image_id"]]
        
        # mapping labels
        self.label_dict = self._create_label_mapping()
        self.all_labels = sorted(list(set([label for sublist in self.label_dict.values() for label in sublist])))
        self.label_to_index = {label: idx for idx, label in enumerate(self.all_labels)}

        # DEBUG 
        missing_files = self.annotations[self.annotations["resolved_path"].apply(lambda p: not os.path.exists(p))]
        if not missing_files.empty:
            print(f"Warning: {len(missing_files)} files listed in CSV not found on disk.")

        print(f"Total images: {len(self.image_paths)}")
        print(f"Using label column: {self.label_column}")
        print(f"All unique labels: {self.all_labels}")

    def _create_label_mapping(self):
        """Maps image IDs to their corresponding labels."""
        label_dict = {}
        for _, row in self.annotations.iterrows():
            #image_id = row["image_id"]
            # DEBUG
            image_id = Path(row["image_id"]).name


            #labels = eval(row["finding_categories"]) 
            
            #if isinstance(labels, list) and len(labels) > 0:
            #    label_dict[image_id] = labels

            raw_label = row.get(self.label_column)

            if isinstance(raw_label, str):
                try:
                    labels = eval(raw_label) if raw_label.startswith("[") else [raw_label]
                except:
                    labels = [raw_label]
            elif pd.isna(raw_label):
                labels = []
            else:
                labels = [str(raw_label)]

            label_dict[image_id] = labels
            
            # DEBUG
            #print(label_dict[image_id])
            
        return label_dict
    
    def get_one_hot_label(self, img_path):
        img_id = Path(img_path).name
        label_strings = self.label_dict.get(img_id, [])

        one_hot_label = [0] * len(self.all_labels)
        for label in label_strings:
            if label in self.label_to_index:
                one_hot_label[self.label_to_index[label]] = 1

        return one_hot_label

    def get_class_index_label(self, img_path):
        img_id = Path(img_path).name
        label_strings = self.label_dict.get(img_id, [])

        if len(label_strings) == 0:
            raise ValueError(f"No label found for image {img_id}")

        label = label_strings[0]
        if label not in self.label_to_index:
            raise ValueError(f"Label '{label}' not found in label_to_index mapping.")

        return self.label_to_index[label]


    def get_mapped_labels(self):
        return [self.get_one_hot_label(img_path) for img_path in self.image_paths]


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        if self.single_label:
            label = self.get_class_index_label(img_path)
            return img, torch.tensor(label, dtype=torch.long)
        else:
            one_hot_label = self.get_one_hot_label(img_path)
            return img, torch.tensor(one_hot_label, dtype=torch.float32)


    def __len__(self):
        return len(self.image_paths)
