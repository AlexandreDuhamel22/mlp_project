#!/usr/bin/env python3
"""
Black-box inversion + rounding or noise defenses on AT&T or Olivetti
"""

import os
import argparse
import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(
    description="Black-box inversion + rounding or noise defenses on AT&T or Olivetti"
)
p.add_argument("--dataset",    choices=["att","oliv"], required=True,
               help="Which dataset to use")
p.add_argument("--models",     nargs="+", choices=["softmax","mlp","dae"],
               required=True, help="Which model(s) to attack")
p.add_argument("-r","--rounds", nargs="+",
               default=["None","0.001","0.005","0.01","0.05"],
               help="Quantization levels ('None' for no rounding)")
p.add_argument("--noise",      nargs="+", default=None,
               help="Gaussian‐noise σ levels; if given, supersedes rounding")
p.add_argument("--max_labels", type=int, default=None,
               help="Reconstruct labels 0..max_labels-1 (if --labels not given)")
p.add_argument("--labels",     nargs="+", type=int, default=None,
               help="Explicit list of labels to reconstruct (overrides --max_labels)")
p.add_argument("--spsa_samples",  type=int,   default=16)
p.add_argument("--gamma",         type=float, default=1e-2)
p.add_argument("--fd_epsilon",    type=float, default=None,
               help="(unused) kept for backward compatibility")
p.add_argument("--spsa_delta",    type=float, default=0.05)
p.add_argument("--iters_softmax", type=int,   default=5000)
p.add_argument("--iters_mlp",     type=int,   default=500)
p.add_argument("--iters_dae",     type=int,   default=500)
p.add_argument("--lr_softmax",    type=float, default=0.1)
p.add_argument("--lr_mlp",        type=float, default=0.1)
p.add_argument("--lr_dae",        type=float, default=0.05)
p.add_argument("--patience",      type=int,   default=100)
cfg = p.parse_args()

# ── normalize defenses ────────────────────────────────────────────────────────
rounds = [None if t.lower() in ("none","n","noround") else float(t)
          for t in cfg.rounds]

if cfg.noise is None:
    noises = None
else:
    noises = []
    for t in cfg.noise:
        tl = str(t).lower()
        if tl in ("none","n","noround"):
            noises.append(None)
        else:
            ft = float(t)
            noises.append(None if ft == 0.0 else ft)

use_noises = (noises is not None and any(n is not None for n in noises))
def_vals   = noises if use_noises else rounds

# ── label list ─────────────────────────────────────────────────────────────────
if cfg.labels is not None:
    labels = cfg.labels
else:
    if cfg.max_labels is None:
        raise ValueError("Must specify either --labels or --max_labels")
    labels = list(range(cfg.max_labels))

# ── helpers ───────────────────────────────────────────────────────────────────
def clamp01(x): return np.clip(x, 0.0, 1.0)

def display_comparison(model_name, lbl, defs, base, losses):
    import matplotlib.pyplot as plt
    ncols = 1 + len(defs)
    fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    plt.subplots_adjust(wspace=0.05)
    orig = origs_att[lbl] if cfg.dataset=="att" else origs_oliv[lbl]
    axs[0].imshow(orig, cmap="gray", aspect="equal")
    axs[0].set_title(f"Original\n(lbl={lbl})")
    axs[0].axis("off")
    label_char = 'σ' if use_noises else 'r'
    for i, d in enumerate(defs, start=1):
        ax = axs[i]
        tag = d if d is not None else "None"
        fn  = os.path.join(f"{base}_r{tag}", model_name,
                           f"{model_name}_inv_{lbl}.png")
        if os.path.exists(fn):
            ax.imshow(plt.imread(fn), cmap="gray")
            ax.set_title(f"{label_char}={d}\nconf={losses[d]:.3f}")
        else:
            ax.text(0.5,0.5,"(missing)",ha="center",va="center")
            ax.set_title(f"{label_char}={d}")
        ax.axis("off")
    plt.show()

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ── dataset‐specific setup ─────────────────────────────────────────────────────
if cfg.dataset=="att":
    ORL = os.path.join("datasets","at&t_face_database")
    D, SH, SUFF = 112*92, (112,92), ""
    origs_att = {}
    for lbl in labels:
        p = os.path.join(ORL, f"s{lbl+1}", "1.pgm")
        origs_att[lbl] = np.array(Image.open(p).convert("L"))
else:
    D, SH, SUFF = 64*64, (64,64), "_oliv"
    data = fetch_olivetti_faces()
    X, y  = data.data, data.target
    train_idx,val_idx = [],[]
    for c in range(40):
        idxs = np.where(y==c)[0]
        train_idx += idxs[:7].tolist()
        val_idx   += idxs[7:].tolist()
    Xtr, Ytr = X[train_idx], y[train_idx]
    origs_oliv = {}
    for lbl in labels:
        i0 = next(i for i in val_idx if y[i]==lbl)
        origs_oliv[lbl] = X[i0].reshape(*SH)

# ── network definitions ───────────────────────────────────────────────────────
class SoftmaxNet(nn.Module):
    def __init__(self,D,C): super().__init__(); self.fc=nn.Linear(D,C)
    def forward(self,x): return self.fc(x.view(x.size(0),-1))

class MLPNet(nn.Module):
    def __init__(self,D,C):
        super().__init__()
        self.fc1=nn.Linear(D,3000); self.fc2=nn.Linear(3000,C)
    def forward(self,x):
        h = F.relu(self.fc1(x.view(x.size(0),-1))); return self.fc2(h)

class DAENet(nn.Module):
    def __init__(self,D,C):
        super().__init__()
        self.enc1=nn.Linear(D,1000); self.enc2=nn.Linear(1000,300)
        self.dec2=nn.Linear(300,1000); self.dec1=nn.Linear(1000,D)
        self.out =nn.Linear(300,C)
    def forward(self,x):
        h1=F.relu(self.enc1(x.view(x.size(0),-1)))
        z =F.relu(self.enc2(h1)); return self.out(z)

def wrap_att(net):
    def f(x_flat):
        t = torch.from_numpy(x_flat.astype(np.float32))\
                 .to(device).unsqueeze(0)
        with torch.no_grad():
            return F.softmax(net(t),dim=1).cpu().numpy()[0]
    return f

# ── SPSA gradient estimator ───────────────────────────────────────────────────
def spsa_grad(fn,x,delta,samples):
    g = np.zeros_like(x)
    for _ in range(samples):
        r = np.random.choice([1.,-1.],size=x.shape)
        g += (fn(clamp01(x+delta*r)) - fn(clamp01(x-delta*r)))\
             / (2*delta) * r
    return g/samples

# ── inversion core ────────────────────────────────────────────────────────────
def invert_bb(name, predictor, lbl, defs, out_root, its, lr):
    rng    = default_rng()
    losses = {}
    for d in defs:
        tag = d if d is not None else "None"
        od  = os.path.join(f"{out_root}_r{tag}", name)
        os.makedirs(od, exist_ok=True)

        # save original
        savep = os.path.join(od, "orig.png")
        if not os.path.exists(savep) and d is None:
            orig_img = origs_att[lbl] if cfg.dataset=="att" else origs_oliv[lbl]
            arr = orig_img if cfg.dataset=="att" else (orig_img*255).astype(np.uint8)
            Image.fromarray(arr,'L').save(savep)

        x, best, xb, stall = clamp01(0.5 + rng.standard_normal(D)*0.01), 1e6, None, 0

        def cost(xv):
            p = predictor(xv).copy()

            if use_noises and d not in (None, 0.0):
                p += rng.normal(0, d, size=p.shape)
                p = np.clip(p, 0.0, 1.0)

            elif (not use_noises) and d is not None:
                p = np.round(p / d) * d
                p = np.clip(p, 0.0, 1.0)

            s = p.sum()
            if s > 0:
                p /= s
            else:
                p = predictor(xv)

            return 1.0 - p[lbl]

        for _ in tqdm(range(its), desc=f"{name} lbl{lbl} d={d}"):
            L = cost(x)
            if L < best:
                best, xb, stall = L, x.copy(), 0
            else:
                stall += 1
            if best <= cfg.gamma or stall >= cfg.patience:
                break
            g = spsa_grad(cost, x,
                          max(cfg.spsa_delta, (d or 0))/2,
                          cfg.spsa_samples)
            x = clamp01(x - lr*g)

        losses[d] = best
        im = (xb.reshape(*SH)*255).astype(np.uint8)
        Image.fromarray(im,'L').save(os.path.join(od, f"{name}_inv_{lbl}.png"))
        print(f"[{name}] lbl={lbl} d={d} loss={best:.4f}")

    return losses

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    mstr  = "_".join(cfg.models)
    dstr  = "_".join("none" if v is None else str(v) for v in def_vals)
    param = (f"{mstr}_{'noise' if use_noises else 'round'}_{dstr}"
             f"_spsa{cfg.spsa_samples}_γ{cfg.gamma}_δ{cfg.spsa_delta}"
             f"_iS{cfg.iters_softmax}_lrS{cfg.lr_softmax}_pat{cfg.patience}"
            ).replace(".","p")

    out_base = os.path.join("output", cfg.dataset, param)
    os.makedirs(out_base, exist_ok=True)

    for mn in cfg.models:
        if cfg.dataset=="att":
            root = "models/att";    ckpt = f"{mn}{SUFF}.pth"
        else:
            root = "models/oliv";  ckpt = f"{mn}_olivetti.pth"

        net  = {"softmax":SoftmaxNet,
                "mlp":    MLPNet,
                "dae":    DAENet}[mn](D,40).to(device)
        sd   = torch.load(os.path.join(root,ckpt), map_location=device)
        net.load_state_dict(sd.get("state_dict",sd))
        net.eval()
        pred = wrap_att(net)

        its = getattr(cfg, f"iters_{mn}")
        lr  = getattr(cfg, f"lr_{mn}")

        for lbl in labels:
            losses = invert_bb(mn, pred, lbl,
                               def_vals, out_base, its, lr)
            display_comparison(mn, lbl, def_vals, out_base, losses)

if __name__=="__main__":
    main()
