import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from Mammo_dataset import MammoDataset

# EfficientNet classifier
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

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

def main(args):
    wandb.init(project="mammo-density-classifier", config=vars(args))
    early_stopping = EarlyStopping(path='best_model.pt')

    # transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load datasets
    train_dataset = MammoDataset(
        root=args.image_root,
        annotation_path=args.annotation_path,
        transform=train_transform,
        mode="training",
        label_column=args.label_column,
        single_label=True
    )

    test_dataset = MammoDataset(
        root=args.image_root,
        annotation_path=args.annotation_path,
        transform=test_transform,
        mode="test",
        label_column=args.label_column,
        single_label=True
    )

    test_dataset.all_labels = train_dataset.all_labels
    test_dataset.label_to_index = train_dataset.label_to_index

    # Testing sth
    train_labels = [train_dataset.get_class_index_label(p) for p in train_dataset.image_paths]
    class_sample_counts = np.bincount(train_labels)
    weights = 1. / class_sample_counts
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # end

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(train_dataset.all_labels)
    model = EfficientNetClassifier(num_classes).to(device)

    # Testing sth
    total_layers = len(list(model.base_model.features.children()))
    freeze_until = total_layers // 2

    for i, child in enumerate(model.base_model.features.children()):
        for param in child.parameters():
            param.requires_grad = i >= freeze_until
    # end
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.all_labels), y=train_dataset.all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch+1})

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

        scheduler.step()

    # validation
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
    print(classification_report(all_targets, all_preds, target_names=train_dataset.all_labels))

    # Map 4-class labels into 2 groups: 0 (A+B), 1 (C+D)
    binary_targets = [0 if t in [0,1] else 1 for t in all_targets]
    binary_preds   = [0 if p in [0,1] else 1 for p in all_preds]

    wandb.log({"macro_f1": f1})
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=all_targets,
                                                               preds=all_preds,
                                                               class_names=train_dataset.all_labels)})

    # Compute metrics
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

# args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--label-column", type=str, default="breast-density")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)
    