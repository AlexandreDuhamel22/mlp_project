#!/usr/bin/env python3
"""
train.py — train Softmax, MLP and DAE on either AT&T or Olivetti faces.
If --out-dir is not provided, models will be saved under models/att/ or models/oliv/.
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm

base_dir = os.getcwd()

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Train Softmax/MLP/DAE on face data")
p.add_argument("--dataset", choices=["att","olivetti"], default="att",
               help="Which dataset to use: 'att' (AT&T) or 'olivetti'.")
p.add_argument("--att-dir", type=str,
               default=os.path.join(base_dir, "datasets/at&t_face_database"),
               help="Path to AT&T face database (only used if --dataset att).")
p.add_argument("--out-dir", type=str, default=None,
               help="Where to save trained model files.  Defaults to models/att/ or models/oliv/.")
args = p.parse_args()

# if user did not set --out-dir, build it from the dataset name
if args.out_dir is None:
    subdir = "att" if args.dataset == "att" else "oliv"
    args.out_dir = os.path.join("models", subdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
print(f"[INFO] Models will be saved under: {args.out_dir}")

# ── DATA LOADING ─────────────────────────────────────────────────────────────
if args.dataset == "att":
    dataset_dir  = args.att_dir
    model_suffix = ""
    assert os.path.isdir(dataset_dir), f"cannot find AT&T dataset at {dataset_dir}"
    subjects = sorted(glob.glob(os.path.join(dataset_dir, "s*")))
    print(f"[INFO] AT&T: found {len(subjects)} subject folders")

    X_train, y_train, X_val, y_val = [], [], [], []
    for subj_path in subjects:
        label = int(os.path.basename(subj_path).lstrip("s")) - 1
        files = sorted(glob.glob(os.path.join(subj_path, "*.pgm")),
                       key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        if len(files) < 10: continue
        for img_file in files[:7]:
            arr = np.array(Image.open(img_file).convert("L"), np.float32)/255.0
            X_train.append(arr.flatten());  y_train.append(label)
        for img_file in files[7:10]:
            arr = np.array(Image.open(img_file).convert("L"), np.float32)/255.0
            X_val.append(arr.flatten());    y_val.append(label)

    X_train = torch.from_numpy(np.stack(X_train)).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val   = torch.from_numpy(np.stack(X_val  )).to(device)
    y_val   = torch.tensor(y_val,   dtype=torch.long, device=device)
    input_dim = X_train.size(1)   # 112*92 = 10304

else:
    # Olivetti loader
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces()
    X, y = data.data.astype(np.float32), data.target
    print(f"[INFO] Olivetti: {X.shape[0]} images of size "
          f"{int(np.sqrt(X.shape[1]))}×{int(np.sqrt(X.shape[1]))}")

    X_train, y_train, X_val, y_val = [], [], [], []
    for cls in range(40):
        idx = np.where(y == cls)[0]
        X_train.append(X[idx[:7]])
        y_train += [cls]*7
        X_val.append(  X[idx[7:10]])
        y_val   += [cls]*3

    X_train = torch.from_numpy(np.concatenate(X_train,axis=0)).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val   = torch.from_numpy(np.concatenate(X_val  ,axis=0)).to(device)
    y_val   = torch.tensor(y_val,   dtype=torch.long, device=device)
    input_dim   = X_train.size(1)   # 64*64 = 4096
    model_suffix = "_oliv"

print(f"[INFO] Loaded {X_train.shape[0]} train, {X_val.shape[0]} val  (dim={input_dim})")
num_classes = int(y_train.max().item()) + 1

# ── MODELS ────────────────────────────────────────────────────────────────────
class SoftmaxNet(nn.Module):
    def __init__(self,D,C): super().__init__(); self.fc=nn.Linear(D,C)
    def forward(self,x): return self.fc(x.view(x.size(0),-1))

class MLPNet(nn.Module):
    def __init__(self,D,C):
        super().__init__()
        self.fc1=nn.Linear(D,3000)
        self.fc2=nn.Linear(3000,C)
    def forward(self,x):
        h=torch.sigmoid(self.fc1(x.view(x.size(0),-1)))
        return self.fc2(h)

class DAEClassifier(nn.Module):
    def __init__(self,D,h1,h2,C):
        super().__init__()
        self.fc1=nn.Linear(D,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,C)
    def forward(self,x):
        x=x.view(x.size(0),-1)
        h1=torch.sigmoid(self.fc1(x))
        h2=torch.sigmoid(self.fc2(h1))
        return self.out(h2)

class Autoencoder1(nn.Module):
    def __init__(self,D,h1):
        super().__init__()
        self.enc=nn.Linear(D,h1)
        self.dec=nn.Linear(h1,D)
    def forward(self,x):
        z = torch.sigmoid(self.enc(x))
        return torch.sigmoid(self.dec(z))

class Autoencoder2(nn.Module):
    def __init__(self,h1,h2):
        super().__init__()
        self.enc=nn.Linear(h1,h2)
        self.dec=nn.Linear(h2,h1)
    def forward(self,h1):
        z2 = torch.sigmoid(self.enc(h1))
        return torch.sigmoid(self.dec(z2))

# ── PRETRAIN DAE ─────────────────────────────────────────────────────────────
def corrupt(x,p=0.3): return x * (torch.rand_like(x)>p).float()
mse = nn.MSELoss()

hidden1, hidden2 = 1000, 300
ae1 = Autoencoder1(input_dim, hidden1).to(device).train()
opt1 = optim.SGD(ae1.parameters(), lr=0.1, momentum=0.9)
best1,noimp1 = float('inf'), 0
for epoch in range(1,501):
    Xc   = corrupt(X_train,0.1)
    recon= ae1(Xc)
    loss1= mse(recon, X_train)
    opt1.zero_grad(); loss1.backward(); opt1.step()
    if loss1.item() < best1 - 1e-6:
        best1,noimp1 = loss1.item(),0
    else:
        noimp1 += 1
    if noimp1 >= 100:
        print(f"[AE1] stop @{epoch}")
        break
ae1.eval(); print(f"[INFO] AE1 MSE={best1:.6f}")

with torch.no_grad():
    H1 = torch.sigmoid(ae1.enc(X_train)).detach()
ae2  = Autoencoder2(hidden1,hidden2).to(device).train()
opt2 = optim.SGD(ae2.parameters(), lr=0.1, momentum=0.9)
best2,noimp2 = float('inf'), 0
for epoch in range(1,501):
    Hc    = corrupt(H1,0.1)
    Hrec  = ae2(Hc)
    loss2 = mse(Hrec, H1)
    opt2.zero_grad(); loss2.backward(); opt2.step()
    if loss2.item() < best2 - 1e-6:
        best2,noimp2 = loss2.item(),0
    else:
        noimp2 += 1
    if noimp2 >= 100:
        print(f"[AE2] stop @{epoch}")
        break
ae2.eval(); print(f"[INFO] AE2 MSE={best2:.6f}")

# ── BUILD CLASSIFIERS & WARM START DAE ──────────────────────────────────────
softmax = SoftmaxNet(input_dim,num_classes).to(device)
mlp      = MLPNet(input_dim,num_classes).to(device)
dae      = DAEClassifier(input_dim,hidden1,hidden2,num_classes).to(device)

with torch.no_grad():
    dae.fc1.weight.copy_(ae1.enc.weight)
    dae.fc1.bias  .copy_(ae1.enc.bias)
    dae.fc2.weight.copy_(ae2.enc.weight)
    dae.fc2.bias  .copy_(ae2.enc.bias)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_model(model, X_tr, y_tr, X_va, y_va,
                lr=0.1, patience=100, use_val_stop=False, val_patience=20):
    print(f"[TRAIN] {model.__class__.__name__}  lr={lr}")
    opt  = optim.SGD(model.parameters(), lr=lr,
                     momentum=(0.9 if not use_val_stop else 0.0))
    crit = nn.CrossEntropyLoss()
    best_tr, stall_tr = 0, 0

    if use_val_stop:
        best_va, stall_va, best_state = 0, 0, None

    for epoch in range(1,10001):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 16):
            idx = perm[i:i+16]
            xb, yb = X_tr[idx], y_tr[idx]
            logits = model(xb)
            loss   = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            tr_acc = (model(X_tr).argmax(1)==y_tr).float().mean().item()

        if tr_acc > best_tr + 1e-6:
            best_tr, stall_tr = tr_acc, 0
        else:
            stall_tr += 1

        if use_val_stop:
            with torch.no_grad():
                va_acc = (model(X_va).argmax(1)==y_va).float().mean().item()
            if va_acc > best_va + 1e-6:
                best_va, stall_va, best_state = va_acc, 0, model.state_dict()
            else:
                stall_va += 1

        if epoch % 10 == 0:
            msg = f"  Ep{epoch:04d} Train={tr_acc*100:.2f}%"
            if use_val_stop:
                msg += f" Val={va_acc*100:.2f}%"
            print(msg)

        if use_val_stop and stall_va >= val_patience:
            print(f"  → stop VAL@{epoch} best={best_va*100:.2f}%")
            model.load_state_dict(best_state)
            break

        if not use_val_stop and stall_tr >= patience:
            print(f"  → stop TR@{epoch} best={best_tr*100:.2f}%")
            break

    with torch.no_grad():
        final_va = (model(X_va).argmax(1)==y_va).float().mean().item()
    print(f"[RESULT] {model.__class__.__name__} "
          f"Train={best_tr*100:.2f}% Val={final_va*100:.2f}%\n")
    return model

# ── SAVE MODELS ─────────────────────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
sf_name = f"softmax{model_suffix}.pth"
ml_name = f"mlp{model_suffix}.pth"
da_name = f"dae{model_suffix}.pth"

softmax = train_model(
    softmax, X_train, y_train, X_val, y_val,
    lr=0.1, patience=100, use_val_stop=True, val_patience=20)
mlp      = train_model(
    mlp,      X_train, y_train, X_val, y_val,
    lr=0.01, patience=100, use_val_stop=False)
dae      = train_model(
    dae,      X_train, y_train, X_val, y_val,
    lr=0.01, patience=100, use_val_stop=False)

torch.save(softmax.state_dict(), os.path.join(args.out_dir, sf_name))
torch.save(mlp.state_dict(),      os.path.join(args.out_dir, ml_name))
torch.save({
    "cls":   dae.state_dict(),
    "dec1_w": ae1.dec.weight, "dec1_b": ae1.dec.bias,
    "dec2_w": ae2.dec.weight, "dec2_b": ae2.dec.bias
}, os.path.join(args.out_dir, da_name))

print("[ALL DONE] Models saved under", args.out_dir)