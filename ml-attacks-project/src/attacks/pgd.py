def pgd_attack(model, data, labels, epsilon=0.3, alpha=0.01, num_iter=40):
    """
    Perform the Projected Gradient Descent (PGD) attack on the input data.

    Parameters:
    - model: The machine learning model to attack.
    - data: The input data to be attacked.
    - labels: The true labels of the input data.
    - epsilon: The maximum perturbation allowed.
    - alpha: The step size for each iteration.
    - num_iter: The number of iterations to perform.

    Returns:
    - adversarial_examples: The generated adversarial examples.
    """
    # Clone the data to create adversarial examples
    adversarial_examples = data.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(adversarial_examples)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the adversarial examples
        with torch.no_grad():
            adversarial_examples = adversarial_examples + alpha * adversarial_examples.grad.sign()
            # Project the adversarial examples to the epsilon ball
            adversarial_examples = torch.clamp(adversarial_examples, data - epsilon, data + epsilon)
            adversarial_examples = torch.clamp(adversarial_examples, 0, 1)  # Ensure valid pixel range

    return adversarial_examples.detach()