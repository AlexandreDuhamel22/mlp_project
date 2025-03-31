import numpy as np

def fgsm_attack(model, X, y, epsilon):
    """
    Perform FGSM attack on the input data.

    Args:
        model: The target model (must support gradient computation).
        X: Input data (numpy array).
        y: True labels (numpy array).
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial examples (numpy array).
    """
    # Compute gradients of the loss with respect to the input
    gradients = compute_gradients(model, X, y)  # Replace with actual gradient computation
    perturbation = epsilon * np.sign(gradients)
    X_adv = X + perturbation
    return np.clip(X_adv, 0, 1)  # Ensure values are in valid range