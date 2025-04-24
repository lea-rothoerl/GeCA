import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from Mammo_dataset import MammoDataset
import argparse
import torchvision.transforms as tf
import wandb

class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--label-column", type=str, default="finding_categories")

    args = parser.parse_args()

    wandb.init(
        project="mammo-multilabel-cnn",
        config={
            "epochs": 5,
            "batch_size": 16,
            "label_column": args.label_column
        }
    )

    transform = tf.Compose([
        tf.ToPILImage(),
        tf.Grayscale(num_output_channels=1),
        tf.Resize((128, 128)),
        tf.ToTensor(),
    ])

    dataset_train = MammoDataset(
        root=args.image_root,
        annotation_path=args.annotation_path,
        transform=transform,
        mode="training",
        label_column=args.label_column,
    )

    dataset_test = MammoDataset(
        root=args.image_root,
        annotation_path=args.annotation_path,
        transform=transform,
        mode="val", 
        label_column=args.label_column,
    )

    dataset_test.all_labels = dataset_train.all_labels

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset_train.all_labels)
    model = MultiLabelCNN(num_classes=num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        wandb.log({"train_loss": running_loss / len(train_loader), "epoch": epoch+1})

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.numpy()

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds = (probs > 0.5).astype(int)

            all_preds.append(preds)
            all_targets.append(labels)

    # concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # F1 score (macro average)
    f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Macro F1 Score: {f1:.4f}")
    wandb.log({"macro_f1": f1})


