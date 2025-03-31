# filepath: /ml-attacks-project/ml-attacks-project/src/main.py

import numpy as np
from models.model1 import Model1
from models.model2 import Model2
from utils.data_loader import load_data
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.cw import CWAttack

def evaluate(predictions, y_test):
    # Function to evaluate predictions against true labels
    accuracy = np.mean(predictions == y_test)
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Initialize and train the model
    model = Model1()
    model.train(X_train, y_train)

    # Perform FGSM attack
    epsilon = 0.1  # Magnitude of the perturbation
    X_test_adv = fgsm_attack(model, X_test, y_test, epsilon)

    # Evaluate the model on adversarial examples
    predictions = model.predict(X_test_adv)
    evaluate(predictions, y_test)

    # Initialize models
    model1 = Model1()
    model2 = Model2()

    # Train models
    model1.train(X_train, y_train)
    model2.train(X_train, y_train)

    # Evaluate models on clean data
    accuracy_model1 = model1.evaluate(X_test, y_test)
    accuracy_model2 = model2.evaluate(X_test, y_test)

    print(f'Accuracy of Model 1 on clean data: {accuracy_model1:.2f}')
    print(f'Accuracy of Model 2 on clean data: {accuracy_model2:.2f}')

    # Generate adversarial examples using FGSM
    X_test_fgsm = fgsm_attack(model1, X_test, y_test)
    accuracy_fgsm = model1.evaluate(X_test_fgsm, y_test)
    print(f'Accuracy of Model 1 on FGSM adversarial examples: {accuracy_fgsm:.2f}')

    # Generate adversarial examples using PGD
    X_test_pgd = pgd_attack(model2, X_test, y_test)
    accuracy_pgd = model2.evaluate(X_test_pgd, y_test)
    print(f'Accuracy of Model 2 on PGD adversarial examples: {accuracy_pgd:.2f}')

    # Generate adversarial examples using CW
    cw_attack = CWAttack(model1)
    X_test_cw = cw_attack.attack(X_test, y_test)
    accuracy_cw = model1.evaluate(X_test_cw, y_test)
    print(f'Accuracy of Model 1 on CW adversarial examples: {accuracy_cw:.2f}')

if __name__ == "__main__":
    main()