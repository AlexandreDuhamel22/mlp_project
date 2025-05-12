# I’ve added inline comments to highlight all the key changes:
# Removed all torchplus/pyoverload imports, using only native libraries.
# Replaced the Init utility with manual device selection.
# Introduced our own ClassificationAccuracy class.
# Streamlined data loading in PreproDataset.
# Centralized hyperparameter and path setup in main().
# Switched from TensorBoard logging to simple print statements.
# Updated the dataset directory to your actual path.
# Simplified checkpoint saving with .pth files.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Utility for classification accuracy
class ClassificationAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def accumulate(self, label, predict):
        self.correct += (predict == label).sum().item()
        self.total += label.size(0)

    def get(self):
        return self.correct / self.total if self.total > 0 else 0

# Preprocess dataset: load all images into memory as tensors
def PreproDataset(root: str, transform):
    ds = ImageFolder(root, transform=transform)
    loader = DataLoader(ds, batch_size=128, num_workers=2)
    images, labels = [], []
    for imgs, lbls in tqdm(loader, desc="Preprocessing dataset"):
        images.append(imgs)
        labels.append(lbls)
    images = torch.cat(images)
    labels = torch.cat(labels)
    return TensorDataset(images, labels)

# Define small linear network
class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.regression = nn.Linear(in_features=input_features, out_features=output_features)

    def forward(self, x):
        return self.regression(x)

# Main training routine
def main():
    # Hyperparameters and paths
    batch_size = 8
    train_epochs = 50
    log_epoch = 2
    class_num = 40
    
    base_dir = os.getcwd()
    root_dir    = os.path.join(base_dir, "models/OG")
    dataset_dir = os.path.join(base_dir, "datasets", "at&t_face_database")

    os.makedirs(root_dir, exist_ok=True)

    # Image dimensions
    h, w = 112, 92

    # Prepare dataset
    transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
    ds = PreproDataset(dataset_dir, transform)
    ds_len = len(ds)
    train_len = ds_len * 7 // 10
    train_ds, test_ds = random_split(ds, [train_len, ds_len - train_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model, optimizer, loss
    model = Net(h * w, class_num).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, train_epochs + 1):
        model.train()
        train_acc = ClassificationAccuracy()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            bs = imgs.size(0)
            optimizer.zero_grad()
            outputs = model(imgs.view(bs, -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            train_acc.accumulate(labels, preds)

        avg_loss = total_loss / len(train_loader)
        acc_train = train_acc.get()
        print(f"Epoch [{epoch}/{train_epochs}] Loss: {avg_loss:.4f}, Train Acc: {acc_train:.4f}")

        # Evaluation and model saving every log_epoch
        if epoch % log_epoch == 0:
            model.eval()
            test_acc = ClassificationAccuracy()
            test_loss = 0.0
            with torch.no_grad():
                for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch} Testing"):
                    imgs, labels = imgs.to(device), labels.to(device)
                    bs = imgs.size(0)
                    outputs = model(imgs.view(bs, -1))
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    preds = outputs.argmax(dim=-1)
                    test_acc.accumulate(labels, preds)

            avg_test_loss = test_loss / len(test_loader)
            acc_test = test_acc.get()
            print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {acc_test:.4f}")

            checkpoint_path = os.path.join(root_dir,f"mynet_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    main()