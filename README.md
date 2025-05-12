MIA-Main: Model Inversion Attack Toolkit

This repository contains our end-to-end implementation of model inversion attacks (white-box & black-box) on facial-recognition and speaker-recognition models. It extends the MIA toolkit of Zhang et al. with additional architectures (MLP, DAE) and countermeasures (Gaussian noise & confidence rounding).

---
Project Structure:
MIA-main/
├── datasets/                   # Raw datasets (AT&T “ORL” & Olivetti faces)
├── models/                     # Trained model checkpoints
│   ├── att/softmax/…           
│   ├── att/mlp/…               
│   └── att/dae/…               
├── train/                      # Training scripts
│   ├── OG_train.py             # Original MIA training (Softmax & MLP)
│   ├── train.py                # Unified train interface (–dataset att/olivetti)
│   └── ext_train.py            # Extended Softmax-only train for refined pipeline
├── attack/                     # Inversion-attack scripts
│   ├── OG_reconstruction.py    # Original white-box reconstructions
│   ├── reconstruction.py       # White-box for all three models
│   ├── prop-black-box.py       # Black-box (AT&T & Olivetti) with noise/rounding
│   └── ext_black-box.py        # Refined Olivetti black-box pipeline
├── utils/                      # Plotting & helper functions
│   ├── plot_OG_reconstruction.py
│   └── plot_reconstruction.py
├── output/                     # Saved inversion outputs & figures
├── logs/                       # Training & attack logs
├── MIA_notebook.ipynb          # Colab notebook: setup + end-to-end demo
├── .gitignore
└── .DS_Store

Installation & Setup:
1. Clone & enter
   git clone https://github.com/your-org/MIA-main.git
   cd MIA-main

2. Create environment
   python3 -m venv venv && source venv/bin/activate
   pip install --upgrade pip

3. Install dependencies
   pip install torch==2.1.2 torchvision==0.16.2 tqdm matplotlib
   pip uninstall -y torchplus && pip install torchplus
   pip uninstall -y tensorflow jax tensorboard jaxlib
   pip install git+https://github.com/zhangzp9970/torchplus.git@master

4. Verify GPU
   import torch
   print("CUDA available:", torch.cuda.is_available())

Usage:

1. White-Box Attacks:
   python train/OG_train.py
   python attack/OG_reconstruction.py
   python utils/plot_OG_reconstruction.py --num-subj 10
   python train/train.py --dataset att
   python attack/reconstruction.py --softmax --mlp --dae --dae_without --max_pic 5
   python utils/plot_reconstruction.py --dataset-dir datasets/at&t_face_database --recon-root output/att --softmax --mlp --dae_without --dae --max 5

2. Black-Box Attacks (AT&T):
   python attack/prop-black-box.py --dataset att --models softmax --labels 0 5 --noise 0 0.001 0.005 0.01 0.02 0.05 --spsa_samples 128 --lr_softmax 0.01 --gamma 0.01 --spsa_delta 0.01 --iters_softmax 1000
   python attack/prop-black-box.py --dataset att --models softmax --labels 0 5 --round None 0.001 0.005 0.01 0.02 0.05 --spsa_samples 128 --lr_softmax 0.01 --gamma 0.01 --spsa_delta 0.01 --iters_softmax 1000

3. Black-Box Attacks (Olivetti):
   python train/train.py --dataset olivetti
   python attack/prop-black-box.py --dataset oliv --models softmax --labels 1 5 --round None 1e-5 5e-5 1e-3 5e-3 1e-2 --iters_softmax 1000 --patience 500 --lr_softmax 0.1 --spsa_samples 16 --gamma 1e-4 --spsa_delta 0.05
   python attack/prop-black-box.py --dataset oliv --models softmax --labels 1 5 --noise 0 1e-5 5e-5 1e-3 5e-3 1e-2 --iters_softmax 1000 --patience 500 --lr_softmax 0.1 --spsa_samples 16 --gamma 1e-4 --spsa_delta 0.05

4. Refined Olivetti Pipeline:
   python train/ext_train.py --model softmax
   python attack/ext_black-box.py --model softmax --label 1 --max_iter 5000 --patience 100 --learning_rate 0.05 --spsa_samples 64 --rounding None 1e-5 5e-5 1e-3 5e-3 1e-2 --output_prefix recon_sm_lbl1
   python attack/ext_black-box.py --model softmax --label 1 --max_iter 5000 --patience 100 --learning_rate 0.05 --spsa_samples 64 --noise 0 1e-5 5e-5 1e-3 5e-3 1e-2 --output_prefix recon_sm_lbl1

Colab Notebook:
See MIA_notebook.ipynb for a guided demo.

References:
- Fredrikson et al., “Model inversion attacks that exploit confidence information and basic countermeasures,” CCS ’15.
- Zhang et al., MIA toolkit GitHub.
- Pizzi et al., “Model Inversion Attacks on Automatic Speaker Recognition,” arXiv:2301.03206.
- Shokri et al., “Exploring Model Inversion Attacks in the Black‐box Setting,” PETS ’23.
- ORL face dataset.
- SincNet paper.
- TIMIT Corpus.
- PyTorch docs.
