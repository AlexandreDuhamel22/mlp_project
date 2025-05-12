# plot_reconstructions.py

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot original images alongside reconstruction results"
    )
    p.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Root of AT&T face folders (s1, s2, …)"
    )
    p.add_argument(
        "--recon-root", "-r",
        required=True,
        help="Root of reconstruction outputs (subfolders softmax, mlp, dae, dae_without)"
    )
    p.add_argument(
        "--softmax", action="store_true", help="Include Softmax reconstructions"
    )
    p.add_argument(
        "--mlp",     action="store_true", help="Include MLP reconstructions"
    )
    p.add_argument(
        "--dae",     action="store_true", help="Include DAE (with Process-DAE)"
    )
    p.add_argument(
        "--dae_without", action="store_true",
        help="Include DAE_without (no Process-DAE)"
    )
    p.add_argument(
        "--max", type=int, default=6,
        help="Max number of subjects to display (default: 6)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Build list of models to plot, in the order given
    MODELS = []
    if args.softmax:     MODELS.append("softmax")
    if args.mlp:         MODELS.append("mlp")
    if args.dae:         MODELS.append("dae")
    if args.dae_without: MODELS.append("dae_wo")

    # If none specified, default to all three
    if not MODELS:
        MODELS = ["softmax", "mlp", "dae"]

    # 1) discover subject folders
    subs = sorted(
        [d for d in os.listdir(args.dataset_dir)
         if d.lower().startswith("s") and
            os.path.isdir(os.path.join(args.dataset_dir, d))],
        key=lambda d: int(d[1:])
    )[: args.max]

    for idx, subj in enumerate(subs):
        subj_dir = os.path.join(args.dataset_dir, subj)

        # pick first image as original
        pgms = sorted(
            [f for f in os.listdir(subj_dir) if f.lower().endswith(".pgm")],
            key=lambda fn: int(os.path.splitext(fn)[0])
        )
        if not pgms:
            print(f"[!] no .pgm in {subj_dir}, skipping")
            continue
        orig_path = os.path.join(subj_dir, pgms[0])

        # collect recon paths
        recon_paths = []
        for m in MODELS:
            p = os.path.join(args.recon_root, m, f"{m}_inv_{idx}.png")
            recon_paths.append(p if os.path.exists(p) else None)
            if recon_paths[-1] is None:
                print(f"[!] missing {m} for {subj}: {p}")

        # plot
        ncols = 1 + len(MODELS)

        fig, axs = plt.subplots(1, ncols, figsize=(3 * ncols, 3))

        plt.subplots_adjust(wspace=0.1, top=0.88)

        # original
        orig = Image.open(orig_path).convert("L")
        axs[0].imshow(orig, cmap="gray")
        axs[0].set_title("Original")
        axs[0].axis("off")

        # reconstructions
        for j, model_name in enumerate(MODELS, start=1):
            ax = axs[j]
            rp = recon_paths[j-1]
            if rp:
                img = Image.open(rp).convert("L")
                ax.imshow(img, cmap="gray")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            ax.set_title(model_name.replace("_"," ").capitalize())
            ax.axis("off")

        plt.show()

if __name__ == "__main__":
    main()
