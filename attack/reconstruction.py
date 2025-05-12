# reconstruction.py

import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageFilter

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="MI-Face inversion: Softmax, MLP, DAE (+ DAE without Process-DAE)"
)
parser.add_argument("--softmax",     action="store_true", help="Run Softmax inversion")
parser.add_argument("--mlp",         action="store_true", help="Run MLP inversion")
parser.add_argument("--dae",         action="store_true", help="Run DAE with Process-DAE")
parser.add_argument("--dae_without", action="store_true", help="Run DAE without Process-DAE")
parser.add_argument("--max_pic", type=int, default=40, help="Max number of labels to attack (default = all)")
args = parser.parse_args()

# If no flags provided, default to full run
if not (args.softmax or args.mlp or args.dae or args.dae_without):
    args.softmax = args.mlp = args.dae = True

# ── Device & seed ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Device: {device}")
torch.manual_seed(0)

# -----------------------------------------------------------------------------
# 0) Helpers
# -----------------------------------------------------------------------------
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
    dx = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    return dx + dy

def normalize_flatten(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)

def process_dae(x_img: torch.Tensor) -> torch.Tensor:
    """NLMeans denoise + sharpen for Process-DAE."""
    arr = (x_img.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(arr, None, h=10,
                                    templateWindowSize=7,
                                    searchWindowSize=21)
    pil = Image.fromarray(den).filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
    )
    proc = np.array(pil).astype(np.float32) / 255.0
    proc = np.clip(proc, 0.0, 1.0)
    return torch.from_numpy(proc).unsqueeze(0).to(device)

def _save_inv(img: torch.Tensor, loss: float, out_root: str, name: str, label: int):
    folder = os.path.join(out_root, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_inv_{label}.png")
    save_image(img, path)
    print(f"[{name}] lb{label} done. loss={loss:.4f} → {path}")
    npimg = img.squeeze().cpu().clamp(0,1).numpy()
    plt.figure(figsize=(3,3))
    plt.imshow(npimg, cmap="gray")
    plt.title(f"{name} inv {label} (loss={loss:.4f})")
    plt.axis("off")
    plt.show()

# -----------------------------------------------------------------------------
# 1) Model definitions
# -----------------------------------------------------------------------------
class SoftmaxNet(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.fc = nn.Linear(D, C)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MLPNet(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.fc1 = nn.Linear(D, 3000)
        self.fc2 = nn.Linear(3000, C)
    def forward(self, x):
        h = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(h)

class DAENet(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.enc1 = nn.Linear(D, 1000)
        self.enc2 = nn.Linear(1000, 300)
        self.dec2 = nn.Linear(300, 1000)
        self.dec1 = nn.Linear(1000, D)
        self.out  = nn.Linear(300, C)
    def forward(self, x):
        h1 = torch.sigmoid(self.enc1(x.view(x.size(0), -1)))
        z  = torch.sigmoid(self.enc2(h1))
        return self.out(z)
    def encode(self, x):
        h1 = torch.sigmoid(self.enc1(x))
        return torch.sigmoid(self.enc2(h1))
    def decode(self, z):
        h2 = torch.sigmoid(self.dec2(z))
        return torch.sigmoid(self.dec1(h2))

# -----------------------------------------------------------------------------
# 2) Pixel‐space inversion for Softmax & MLP (unchanged)
# -----------------------------------------------------------------------------
def invert_softmax(model, label, D, out_root, name,
                   α=50000, β=1000, γ=1e-4, λ=0.05, μ=0.95):
    model.eval()
    x = torch.zeros(1, D, device=device, requires_grad=True)
    v = torch.zeros_like(x)
    best, best_loss = x.clone().detach(), float("inf")
    prev, stagn = float("inf"), 0
    for _ in tqdm(range(α), desc=f"{name} invert lb{label}"):
        logits = model(x)
        loss   = F.cross_entropy(logits, torch.tensor([label], device=device))
        model.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        g = x.grad.detach()
        v = μ*v + g
        x = x - λ*v
        x = normalize_flatten(x).detach().clamp(0,1).requires_grad_()
        if loss.item() < best_loss:
            best_loss, best = loss.item(), x.clone().detach()
        if loss.item() >= prev:
            stagn += 1
            if stagn > β: break
        else:
            stagn = 0
        prev = loss.item()
        if loss.item() < γ: break
    _save_inv(best.view(1,112,92), best_loss, out_root, name, label)

def invert_mlp(model, label, D, out_root, name,
               α=50000, β=1000, γ=1e-4, λ=0.05, μ=0.95):
    model.eval()
    x = torch.zeros(1, D, device=device, requires_grad=True)
    v = torch.zeros_like(x)
    best, best_loss = x.clone().detach(), float("inf")
    prev_conf, stagn = -1.0, 0
    for _ in tqdm(range(α), desc=f"{name} invert lb{label}"):
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        conf   = probs[0, label]
        loss   = 1.0 - conf
        model.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        g = x.grad.detach()
        v = μ*v + g
        x = x - λ*v
        x = normalize_flatten(x).detach().clamp(0,1).requires_grad_()
        if loss.item() < best_loss:
            best_loss, best = loss.item(), x.clone().detach()
        if conf.item() <= prev_conf:
            stagn += 1
            if stagn > β: break
        else:
            stagn = 0
        prev_conf = conf.item()
        if loss.item() < γ: break
    _save_inv(best.view(1,112,92), best_loss, out_root, name, label)

# -----------------------------------------------------------------------------
# 3) Latent‐space inversion for DAE (mise à jour hyperparams d’après l’article)
# -----------------------------------------------------------------------------
def invert_dae(model, label, D, out_root, name,
               α=5000, β=100, γ=10e-3, λ=0.1, μ=0.9,
               reproj_every=512, min_iters=500, use_process_dae=True):
    model.eval()
    # init z0 = encode(0)
    with torch.no_grad():
        z = model.encode(torch.zeros(1, D, device=device))
    z = z.clone().detach().requires_grad_(True)
    v_z = torch.zeros_like(z)
    best_z, best_loss = z.clone().detach(), float("inf")
    prev, stagn = float("inf"), 0

    print(f"[{name}] Start label={label}, Process-DAE={'ON' if use_process_dae else 'OFF'}")
    for i in range(1, α+1):
        logits = model.out(z)
        prob   = F.softmax(logits, dim=1)[0, label]
        x_dec  = model.decode(z).view(1,112,92)
        # coût principal + TV + L2
        loss   = (1.0 - prob) + 1e-4 * tv_loss(x_dec) + 1e-3 * (z.norm()**2)

        model.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        g_z = z.grad.detach()
        v_z = μ*v_z + g_z
        z   = (z - λ * v_z).detach().requires_grad_()

        # Process-DAE projection toutes les reproj_every
        if use_process_dae and (i % reproj_every == 0):
            with torch.no_grad():
                x_img = model.decode(z).view(1,112,92)
                x_pr  = process_dae(x_img)
                z     = model.encode(x_pr.view(1,-1)).detach().requires_grad_()
            v_z.zero_()

        if i % 500 == 0:
            print(f"[{name}] iter {i} loss={loss.item():.4f}")

        # track best
        if loss.item() < best_loss:
            best_loss, best_z = loss.item(), z.clone().detach()

        # only start early‐stopping after min_iters
        if i >= min_iters:
            if loss.item() >= prev:
                stagn += 1
                if stagn > β:
                    print(f"[{name}] stop (stagnation) at iter={i}")
                    break
            else:
                stagn = 0
            if loss.item() < γ:
                print(f"[{name}] stop (loss<{γ}) at iter={i}")
                break

        prev = loss.item()

    with torch.no_grad():
        final = model.decode(best_z).view(1,112,92)
        pred  = model.out(best_z).softmax(dim=1).argmax().item()
    print(f"[{name}] lb{label} final pred={pred}, best_loss={best_loss:.4f}")
    _save_inv(final, best_loss, out_root, name, label)

# -----------------------------------------------------------------------------
# 4) Orchestrator
# -----------------------------------------------------------------------------
def attack_all(D, args):
    n_labels = args.max_pic

    out_root   = "output/att"
    os.makedirs(out_root, exist_ok=True)
    model_root = "models/att"

    specs = []
    if args.softmax:
        specs.append(("softmax",    SoftmaxNet, "softmax.pth",    invert_softmax, {}))
    if args.mlp:
        specs.append(("mlp",        MLPNet,     "mlp.pth",        invert_mlp,     {}))
    if args.dae:
        specs.append(("dae",        DAENet,     "dae.pth",        invert_dae,     {"use_process_dae": True}))
    if args.dae_without:
        specs.append(("dae_wo",     DAENet,     "dae.pth",        invert_dae,     {"use_process_dae": False}))

    for name, NetC, ckpt, fn, kwargs in specs:
        print(f"\n=== Attacking {name} ===")
        net  = NetC(D, 40).to(device)
        path = os.path.join(model_root, ckpt)
        assert os.path.exists(path), f"Missing {path}"

        if name in ("softmax", "mlp"):
            net.load_state_dict(torch.load(path, map_location=device))
        else:
            data = torch.load(path, map_location=device)
            cls  = data["cls"]
            net.enc1.weight.data.copy_(cls["fc1.weight"])
            net.enc1.bias.data.copy_(cls["fc1.bias"])
            net.enc2.weight.data.copy_(cls["fc2.weight"])
            net.enc2.bias.data.copy_(cls["fc2.bias"])
            net.out.weight.data.copy_(cls["out.weight"])
            net.out.bias.data.copy_(cls["out.bias"])
            net.dec1.weight.data.copy_(data["dec1_w"])
            net.dec1.bias.data.copy_(data["dec1_b"])
            net.dec2.weight.data.copy_(data["dec2_w"])
            net.dec2.bias.data.copy_(data["dec2_b"])

        net.eval()
        for lbl in range(n_labels):
            fn(net, lbl, D, out_root, name, **kwargs)

    print(f"\nAll attacks completed → {out_root}/")

if __name__ == "__main__":
    attack_all(112*92, args)
