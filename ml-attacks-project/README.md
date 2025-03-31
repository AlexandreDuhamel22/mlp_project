# ML Attacks Project

Ce projet implémente diverses attaques sur des modèles de machine learning. Les attaques incluent des méthodes bien connues telles que FGSM, PGD et CW, qui sont utilisées pour générer des exemples adversariaux afin d'évaluer la robustesse des modèles.

## Structure du projet

```
ml-attacks-project
├── src
│   ├── attacks
│   │   ├── fgsm.py        # Implémentation de l'attaque FGSM
│   │   ├── pgd.py         # Implémentation de l'attaque PGD
│   │   └── cw.py          # Implémentation de l'attaque CW
│   ├── models
│   │   ├── model1.py      # Définition du modèle 1
│   │   ├── model2.py      # Définition du modèle 2
│   │   └── __init__.py    # Importation des modèles
│   ├── utils
│   │   ├── data_loader.py  # Chargement et prétraitement des données
│   │   ├── metrics.py      # Évaluation des performances des modèles
│   │   └── __init__.py     # Importation des utilitaires
│   └── main.py             # Point d'entrée de l'application
├── requirements.txt         # Dépendances nécessaires
├── README.md                # Documentation du projet
└── .gitignore               # Fichiers à ignorer par Git
```

## Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```
pip install -r requirements.txt
```

## Utilisation

1. **Charger les données** : Utilisez les fonctions dans `src/utils/data_loader.py` pour charger vos ensembles de données.
2. **Configurer les modèles** : Modifiez `src/models/model1.py` et `src/models/model2.py` pour définir vos modèles de machine learning.
3. **Exécuter les attaques** : Utilisez les fonctions dans `src/attacks/fgsm.py`, `src/attacks/pgd.py`, et `src/attacks/cw.py` pour générer des exemples adversariaux.
4. **Évaluer les performances** : Utilisez les fonctions dans `src/utils/metrics.py` pour évaluer la robustesse de vos modèles face aux attaques.

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à soumettre des demandes de tirage pour améliorer le projet.

## License

Ce projet est sous licence MIT. Veuillez consulter le fichier LICENSE pour plus de détails.