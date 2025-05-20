import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from Mammo_dataset import MammoDataset
import argparse
import torchvision.transforms as tf
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    
class EarlyStopping:
    def __init__(self, path):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.path = path

    def __call__(self, val_score, model):
        score = -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score:
            self.counter += 1
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        torch.save(model.state_dict(), self.path)
        self.val_score_min = val_score    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--label-column", type=str, default="finding_categories")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()

    wandb.init(
        project="mammo-density-cnn",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "label_column": args.label_column
        }
    )

    early_stopping = EarlyStopping(path='best_model.pt')

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
        single_label=True
    )

    dataset_test = MammoDataset(
        root=args.image_root,
        annotation_path=args.annotation_path,
        transform=transform,
        mode="val", 
        label_column=args.label_column,
        single_label=True
    )

    dataset_test.all_labels = dataset_train.all_labels

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset_train.all_labels)
    model = MultiLabelCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        wandb.log({"train_loss": running_loss / len(train_loader), "epoch": epoch+1})

        # validation loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_preds, val_targets, average="macro")
        print(f"Epoch {epoch+1}, Validation Macro F1: {val_f1:.4f}")
        wandb.log({"val_f1": val_f1, "epoch": epoch+1})

        # early stopping check 
        early_stopping(val_f1, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels_np = labels.numpy()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels_np)

    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)
    print(f"\nMacro F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=dataset_train.all_labels, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4) 
    print(report_df)

    # log report to wandb as a table
    wandb.log({"classification_report_table": wandb.Table(dataframe=report_df)})
    # log macro F1 score
    wandb.log({"macro_f1": f1})

    # map 4-class labels into binary groups: 0 (A+B), 1 (C+D)
    binary_targets = [0 if t in [0,1] else 1 for t in all_targets]
    binary_preds   = [0 if p in [0,1] else 1 for p in all_preds]

    accuracy = accuracy_score(binary_targets, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_targets, binary_preds, average="binary")

    print("\nBinary Classification (A+B vs C+D):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    wandb.log({
        "binary_accuracy": accuracy,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1
    })

    torch.save(model.state_dict(), "mammo_density_classifier.pt")
    print("Model saved.")
    
    # heatmap for confusion matrix
    short_labels = [label.split()[-1] for label in dataset_train.all_labels]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=short_labels, 
                yticklabels=short_labels, 
                cmap="Blues",
                annot_kws={"size": 22}) 
    plt.xlabel("Predicted label", fontsize=22)
    plt.ylabel("Actual label", fontsize=22)
    plt.title("Confusion Matrix", fontsize=28)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    wandb.log({"confusion_matrix_image": wandb.Image(plt)})
    plt.close()

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.numpy()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(labels)

    # concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    wandb.log({"macro_f1": f1})
