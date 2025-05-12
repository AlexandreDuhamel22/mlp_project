import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Visualize original vs inverted faces")
parser.add_argument(
    "--num-subj", "-n",
    type=int,
    default=6,
    help="How many subjects to display (default: 6)"
)
args = parser.parse_args()

base_dir    = os.getcwd()
attack_dir  = os.path.join(base_dir, "output/OG")
dataset_dir = os.path.join(base_dir, "datasets/at&t_face_database")

subjects = sorted(
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
)

for idx, subj in enumerate(subjects[: args.num_subj ]):
    subdir = os.path.join(dataset_dir, subj)
    imgname = sorted(os.listdir(subdir))[0]
    orig = Image.open(os.path.join(subdir, imgname)).convert("L")

    inv = Image.open(os.path.join(attack_dir, f"inverted_{idx}.png")).convert("L")

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(orig, cmap="gray")
    axs[0].set_title(f"Original ({subj})")
    axs[0].axis("off")
    axs[1].imshow(inv, cmap="gray")
    axs[1].set_title(f"Inverted ({idx})")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
