# MIA-Main: Model Inversion Attack Toolkit

**Model inversion attacks** reconstruct input data from model outputs. This toolkit implements end-to-end inversion attacks—both **white-box** and **black-box**—on facial-recognition and speaker-recognition models. It extends the original MIA toolkit of Zhang *et al.* with additional architectures (MLP, DAE) and lightweight defenses (Gaussian noise & confidence rounding).

---

## 📂 Project Structure

```
MIA-main/
├── datasets/                   # Raw datasets (AT&T “ORL” & Olivetti faces)
├── models/                     # Trained model checkpoints
│   ├── att/softmax/…           
│   ├── att/mlp/…               
│   └── att/dae/…               
├── train/                      # Training scripts
│   ├── OG_train.py             # Original MIA training (Softmax & MLP)
│   ├── train.py                # Unified interface (–dataset att/olivetti)
│   └── ext_train.py            # Softmax-only train for refined pipeline
├── attack/                     # Inversion-attack scripts
│   ├── OG_reconstruction.py    # Original white-box reconstructions
│   ├── reconstruction.py       # White-box for Softmax/MLP/DAE
│   ├── prop-black-box.py       # Black-box (AT&T & Olivetti) with defenses
│   └── ext_black-box.py        # Refined Olivetti black-box pipeline
├── utils/                      # Plotting & helper functions
│   ├── plot_OG_reconstruction.py
│   └── plot_reconstruction.py
├── output/                     # Saved inversion outputs & figures
├── logs/                       # Training & attack logs
├── MIA_notebook.ipynb          # Colab notebook: setup + demo
├── .gitignore
└── .DS_Store
```

---

## ⚙️ Installation & Setup

1. **Clone & cd**  
   ```bash
   git clone https://github.com/your-org/MIA-main.git
   cd MIA-main
   ```

2. **Create virtualenv**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install dependencies**  
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 tqdm matplotlib
   pip uninstall -y torchplus && pip install torchplus
   pip uninstall -y tensorflow jax tensorboard jaxlib
   pip install git+https://github.com/zhangzp9970/torchplus.git@master
   ```

4. **Verify GPU**  
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   ```

---

## ▶️ Usage

### 1. White-Box Attacks

```bash
# Original MIA training & inversion
python train/OG_train.py
python attack/OG_reconstruction.py
python utils/plot_OG_reconstruction.py --num-subj 10

# Train all models
python train/train.py --dataset att

# Perform white-box inversions
python attack/reconstruction.py   --softmax --mlp --dae --dae_without   --max_pic 5

# Plot results
python utils/plot_reconstruction.py   --dataset-dir datasets/at&t_face_database   --recon-root output/att   --softmax --mlp --dae_without --dae   --max 5
```

### 2. Black-Box Attacks (AT&T)

```bash
# Gaussian noise defense
python attack/prop-black-box.py   --dataset att --models softmax --labels 0 5   --noise 0 0.001 0.005 0.01 0.02 0.05   --spsa_samples 128 --lr_softmax 0.01   --gamma 0.01 --spsa_delta 0.01   --iters_softmax 1000

# Confidence rounding defense
python attack/prop-black-box.py   --dataset att --models softmax --labels 0 5   --round None 0.001 0.005 0.01 0.02 0.05   --spsa_samples 128 --lr_softmax 0.01   --gamma 0.01 --spsa_delta 0.01   --iters_softmax 1000
```

### 3. Black-Box Attacks (Olivetti)

```bash
# Train Olivetti Softmax
python train/train.py --dataset olivetti

# Rounding defense
python attack/prop-black-box.py   --dataset oliv --models softmax --labels 1 5   --round None 1e-5 5e-5 1e-3 5e-3 1e-2   --iters_softmax 1000 --patience 500   --lr_softmax 0.1 --spsa_samples 16   --gamma 1e-4 --spsa_delta 0.05

# Gaussian noise defense
python attack/prop-black-box.py   --dataset oliv --models softmax --labels 1 5   --noise 0 1e-5 5e-5 1e-3 5e-3 1e-2   --iters_softmax 1000 --patience 500   --lr_softmax 0.1 --spsa_samples 16   --gamma 1e-4 --spsa_delta 0.05
```

### 4. Refined Olivetti Pipeline

```bash
# Train refined Softmax
python train/ext_train.py --model softmax

# Rounding defense
python attack/ext_black-box.py   --model softmax --label 1   --max_iter 5000 --patience 100   --learning_rate 0.05 --spsa_samples 64   --rounding None 1e-5 5e-5 1e-3 5e-3 1e-2   --output_prefix recon_sm_lbl1

# Gaussian noise defense
python attack/ext_black-box.py   --model softmax --label 1   --max_iter 5000 --patience 100   --learning_rate 0.05 --spsa_samples 64   --noise 0 1e-5 5e-5 1e-3 5e-3 1e-2   --output_prefix recon_sm_lbl1
```

---

## 📖 Colab Notebook

See **MIA_notebook.ipynb** for a step-by-step Google Colab demo:
- Mount Drive
- Install dependencies
- Run training & attacks
- Plot and inspect results

---

## 📚 References

- **Fredrikson et al.**, *CCS ’15*: Model inversion attacks & defenses.  
- **Zhang et al.**, *MIA Toolkit*: GitHub repository.  
- **Pizzi et al.**, arXiv:2301.03206: Speaker-recognition inversion.  
- **Shokri et al.**, PETS ’23: Black-box inversion analysis.  
- **ORL Face Database**: AT&T “ORL” dataset.  
- **PETS 2023**: Reproducibility guidelines for privacy research.  
- **PyTorch**, **scikit-learn**, et al.: Framework docs.  
