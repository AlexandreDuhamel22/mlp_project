import numpy as np
from numpy.random import default_rng
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from PIL import Image
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Black-box model inversion attack demo")
parser.add_argument("--model", choices=["softmax", "mlp", "dae"], default="softmax",
                    help="Type of target model to attack: 'softmax', 'mlp', or 'dae'.")
parser.add_argument("--label", type=int, default=0,
                    help="Target class label to reconstruct (0-39 for Olivetti faces).")
parser.add_argument("--rounding", type=float, default=None,
                    help="Rounding granularity for output defense.")
parser.add_argument("--max_iter", type=int, default=5000,
                    help="Maximum gradient descent steps for the attack.")
parser.add_argument("--patience", type=int, default=100,
                    help="Patience for stopping.")
parser.add_argument("--learning_rate", type=float, default=0.1,
                    help="Gradient descent step size.")
parser.add_argument("--spsa_samples", type=int, default=1,
                    help="Number of SPSA samples for gradient estimation.")
parser.add_argument("--perturbation", type=float, default=None,
                    help="Perturbation size for SPSA.")
parser.add_argument("--output_image", type=str, default="reconstruction.png",
                    help="Filename for the reconstructed image.")
args = parser.parse_args()

# Load Olivetti Faces dataset
data = fetch_olivetti_faces()
X_all = data.data
y_all = data.target

# Split into train/test (7 train + 3 test per class)
train_idx, test_idx = [], []
for cls in range(40):
    idx = np.where(y_all == cls)[0]
    train_idx.extend(idx[:7])
    test_idx.extend(idx[7:])
train_idx, test_idx = np.array(train_idx), np.array(test_idx)
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# Train the target model
if args.model == "softmax":
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"[Softmax] Training accuracy: {train_acc*100:.2f}%")
    print(f"[Softmax] Testing accuracy: {test_acc*100:.2f}%")

elif args.model == "mlp":
    model = MLPClassifier(hidden_layer_sizes=(300,), activation='logistic', max_iter=500, solver='adam',
                          learning_rate_init=0.01, early_stopping=True, n_iter_no_change=20)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"[MLP] Training accuracy: {train_acc*100:.2f}%")
    print(f"[MLP] Testing accuracy: {test_acc*100:.2f}%")

else:
    pca = PCA(n_components=300)
    X_train_latent = pca.fit_transform(X_train)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train_latent, y_train)
    train_acc = model.score(X_train_latent, y_train)

    X_test_latent = pca.transform(X_test)
    test_acc = model.score(X_test_latent, y_test)

    print(f"[DAE (PCA)] Training accuracy (latent space): {train_acc*100:.2f}%")
    print(f"[DAE (PCA)] Testing accuracy (latent space): {test_acc*100:.2f}%")

    def model_predict_proba_pixels(x):
        X_flat = x.reshape(1, -1)
        X_latent = pca.transform(X_flat)
        probs = model.predict_proba(X_latent)[0]
        return probs

# Final summary log (for LaTeX integration)
print("\n====================================")
print(f"[INFO] Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%")
print("====================================")
