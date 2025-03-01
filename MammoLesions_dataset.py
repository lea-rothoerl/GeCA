class MammoLesionsDataset(Dataset):
    def __init__(self, root: str, mode: str = "train", transform=None):
        super(MammoLesionsDataset, self).__init__()
        self.root = root
        self.transform = transform
        
        # get all PNG files in the folder
        self.image_paths = glob(os.path.join(root, mode, '*.png'))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.image_paths)
