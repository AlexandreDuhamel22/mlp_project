#!/usr/bin/env python3
"""
Black-box inversion with two lightweight defenses:
  1) Rounding confidences (--rounding)
  2) Gaussian noise on confidences (--noise)

Supports AT&T (112×92) or Olivetti (64×64) faces.
"""
import os
import argparse
import numpy as np
from numpy.random import default_rng
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Black-box inversion with rounding or Gaussian-noise defenses"
)
parser.add_argument("--model", choices=["softmax","mlp","dae"], default="softmax",
    help="Which model to attack: softmax, mlp, or dae (PCA+softmax).")
parser.add_argument("-r","--rounding", nargs="+", default=["None"],
    help="Rounding levels (e.g. 0.001 0.01), or 'None' for no rounding.")
parser.add_argument("--noise", nargs="+", type=float, default=None,
    help="List of Gaussian σ to add to confidences (supersedes rounding).")
parser.add_argument("--labels", nargs="+", type=int, default=None,
    help="One or more class labels to reconstruct (0–39).")
parser.add_argument("--max_labels", type=int, default=None,
    help="If --labels is omitted, reconstruct 0..max_labels-1")
parser.add_argument("--max_iter", type=int, default=5000,
    help="Max GD iterations.")
parser.add_argument("--patience", type=int, default=100,
    help="Early stop if no improvement over this many iterations.")
parser.add_argument("--gamma", type=float, default=1e-2,
    help="Early-stop loss threshold (best_loss ≤ γ).")
parser.add_argument("--learning_rate", type=float, default=0.1,
    help="GD step size.")
parser.add_argument("--spsa_samples", type=int, default=1,
    help="Number of SPSA perturbations per step.")
parser.add_argument("--perturbation", type=float, default=None,
    help="SPSA perturbation size (default = half the rounding step or 0.01).")
parser.add_argument("--output_prefix", type=str, default="recon",
    help="Prefix for saving reconstructed PNGs")
args = parser.parse_args()

# ── determine labels to attack ────────────────────────────────────────────────
if args.labels is not None:
    labels = args.labels
elif args.max_labels is not None:
    labels = list(range(args.max_labels))
else:
    parser.error("must specify either --labels or --max_labels")

# ── parse rounding levels ─────────────────────────────────────────────────────
roundings = []
for t in args.rounding:
    if t.lower() in ("none","noround","n","null"):
        roundings.append(None)
    else:
        try:
            roundings.append(float(t))
        except:
            parser.error(f"Invalid rounding value: {t}")

# ── parse noises, treat exactly 0 as “no noise” ────────────────────────────────
if args.noise is None:
    noises = None
else:
    noises = []
    for t in args.noise:
        if t == 0:
            noises.append(None)
        else:
            noises.append(t)

use_noises = (noises is not None)
def_vals   = noises if use_noises else roundings

# ── RNG ───────────────────────────────────────────────────────────────────────
rng = default_rng()

# ── Load & split Olivetti ──────────────────────────────────────────────────────
data       = fetch_olivetti_faces()
X_all, y_all = data.data, data.target  # (400,4096)
train_idx, val_idx = [], []
for c in range(40):
    inds = np.where(y_all == c)[0]
    train_idx.extend(inds[:7])
    val_idx.extend(inds[7:])
train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)
X_train, y_train = X_all[train_idx], y_all[train_idx]

# ── Train / load model ────────────────────────────────────────────────────────
if args.model == "softmax":
    model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000)
    model.fit(X_train, y_train)
    def predict_proba(x_flat):
        return model.predict_proba(x_flat.reshape(1,-1))[0]

elif args.model == "mlp":
    model = MLPClassifier(
        hidden_layer_sizes=(300,),
        activation="logistic",
        solver="adam",
        max_iter=500,
        learning_rate_init=0.01,
        early_stopping=True,
        n_iter_no_change=20
    )
    model.fit(X_train, y_train)
    def predict_proba(x_flat):
        return model.predict_proba(x_flat.reshape(1,-1))[0]

else:  # dae via PCA + softmax
    pca = PCA(n_components=300)
    X_lat = pca.fit_transform(X_train)
    clf   = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000)
    clf.fit(X_lat, y_train)
    def predict_proba(x_flat):
        lat = pca.transform(x_flat.reshape(1,-1))
        return clf.predict_proba(lat)[0]

# ── Build unique output folder ────────────────────────────────────────────────
def sanitize(s):
    return s.replace(" ", "")\
            .replace("[","")\
            .replace("]","")\
            .replace(",","-")
param_parts = [
    f"model={args.model}",
    f"labels={'-'.join(map(str,labels))}",
    f"round=" + ("-".join(str(r) for r in roundings)),
    f"noise=" + ("-".join(str(n) for n in (args.noise or [])) or "None"),
    f"iters={args.max_iter}",
    f"lr={args.learning_rate}",
    f"spsa={args.spsa_samples}",
    f"perturb={args.perturbation or 0}",
    f"γ={args.gamma}"
]
run_name = sanitize("_".join(param_parts))
out_root  = os.path.join("output","ext",run_name)
os.makedirs(out_root, exist_ok=True)

# ── Inversion routine ─────────────────────────────────────────────────────────
def invert_with_defense(r=None, noise=None, label=None):
    D = X_all.shape[1]
    x = np.clip(0.5 + 0.01 * rng.standard_normal(D), 0, 1)
    best_x, best_loss = x.copy(), float("inf")
    delta = (
        args.perturbation
        if args.perturbation is not None
        else (0.5*r if (r and isinstance(r,float)) else 0.01)
    )
    stall = 0

    def cost_and_probs(xv):
        """
        Returns (loss, p) where
          loss = 1 - p[label]
          p    = post-processed confidence vector
        """
        p = predict_proba(xv)

        # Gaussian-noise defense
        if noise is not None:
            p_noisy = p + rng.normal(0, noise, size=p.shape)
            p = np.clip(p_noisy, 0, 1)
            if p.sum() > 0:
                p /= p.sum()
            else:
                p = predict_proba(xv)

        # Rounding defense
        elif r is not None:
            q = np.round(p / r) * r
            if q.sum() > 0:
                p = q / q.sum()
            else:
                p = predict_proba(xv)

        loss = 1 - p[label]
        return loss, p

    # initial loss
    loss, _ = cost_and_probs(x)
    best_loss = loss

    for _ in range(args.max_iter):
        grad = np.zeros_like(x)
        for __ in range(args.spsa_samples):
            rv  = rng.choice([-1,1], size=D)
            xp  = np.clip(x + delta*rv, 0,1)
            xm  = np.clip(x - delta*rv, 0,1)
            cp, _  = cost_and_probs(xp)
            cm, _  = cost_and_probs(xm)
            grad += (cp - cm)/(2*delta) * rv
        grad /= args.spsa_samples

        x = np.clip(x - args.learning_rate * grad, 0,1)
        loss, _ = cost_and_probs(x)

        if loss < best_loss:
            best_loss, best_x, stall = loss, x.copy(), 0
        else:
            stall += 1

        # early stop on patience OR on γ threshold
        if stall >= args.patience or best_loss <= args.gamma:
            break

    return best_x.reshape(64,64), best_loss

# ── Run inversion + save + display ────────────────────────────────────────────
for lbl in labels:
    # pick one validation image for original
    orig_idxs = val_idx[y_all[val_idx] == lbl]
    if not len(orig_idxs):
        raise ValueError(f"No validation image for label {lbl}")
    orig_img = X_all[orig_idxs[0]].reshape(64,64)

    results = {}
    kind = "noise" if use_noises else "round"

    sweep = noises   if use_noises else roundings
    for v in sweep:
        recon, loss = invert_with_defense(
            r=None if use_noises else v,
            noise=v    if use_noises else None,
            label=lbl
        )
        tag = f"{kind}={v}" if v is not None else f"{kind}=None"
        results[tag] = (recon, loss)

        fname = os.path.join(
            out_root,
            f"{args.output_prefix}_lbl{lbl}_{tag}.png"
        )
        Image.fromarray((recon*255).astype(np.uint8), "L").save(fname)
        print(f"[lbl{lbl}] {tag} → conf={loss:.4f}, saved {fname}")

    # plot original + all reconstructions
    n = 1 + len(results)
    fig, axs = plt.subplots(1, n, figsize=(4*n,4))
    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(orig_img, cmap="gray", aspect="equal")
    axs[0].set_title(f"Original (lbl={lbl})")
    axs[0].axis("off")

    for i,(tag,(im,ls)) in enumerate(results.items(), start=1):
        axs[i].imshow(im, cmap="gray", aspect="equal")
        axs[i].set_title(f"{tag}\nconf={ls:.3f}")
        axs[i].axis("off")

    plt.show()
