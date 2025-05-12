import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)

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

# Simple linear network matching training model
class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.regression = nn.Linear(in_features=input_features, out_features=output_features)

    def forward(self, x):
        return self.regression(x)

# Normalize image flatten between 0 and 1
def normalize_flatten(x):
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-12)

# Attack procedure: invert model for each target label
def attack_model(model, target_label, h, w, alpha, beta, gamma, lr, momentum, save_dir):
    model.eval()
    # start from zero image
    x = torch.zeros(1, h * w, device=device, requires_grad=True)
    v = torch.zeros_like(x)
    prev_cost = float('inf')
    stagnation = 0

    for i in tqdm(range(alpha), desc=f"Attack label {target_label}"):
        out = model(x)
        loss = F.cross_entropy(out, torch.tensor([target_label], device=device))
        model.zero_grad()
        loss.backward()
        grad = x.grad
        # SGD update
        v = momentum * v + grad
        x = x - lr * v
        x = normalize_flatten(x)
        x = x.detach().clamp(0, 1).requires_grad_(True)

        # early stopping on stagnation or low cost
        if loss.item() >= prev_cost:
            stagnation += 1
            if stagnation > beta:
                break
        else:
            stagnation = 0
        prev_cost = loss.item()
        if loss.item() < gamma:
            break

    # Final output and save
    out = model(x.detach())
    pred = out.softmax(dim=-1).argmax(dim=-1).item()
    print(f"Label {target_label} attacked, final pred: {pred}")
    # Save inverted image
    img = x.view(1, h, w).cpu()
    os.makedirs(save_dir, exist_ok=True)
    save_image(img, os.path.join(save_dir, f"inverted_{target_label}.png"))

# Main attack script
if __name__ == "__main__":
    # Parameters
    class_num = 40
    h, w = 112, 92
    alpha = 50000
    beta = 1000
    gamma = 1e-3
    lr = 0.1
    momentum = 0.9

    base_dir   = os.getcwd()
    root_dir   = os.path.join(base_dir, "output/OG")
    model_path = os.path.join(base_dir, "models/OG/mynet_epoch50.pth")
    os.makedirs(root_dir, exist_ok=True)

    # Load model
    model = Net(h * w, class_num).to(device)
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Run attacks
    for label in range(class_num):
        attack_model(model, label, h, w, alpha, beta, gamma, lr, momentum, root_dir)

    print("All attacks completed.")
