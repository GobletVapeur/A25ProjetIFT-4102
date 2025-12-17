# Projet — Apprentissage d’heuristiques pour Pacman (IFT-4102)

## Architecture — Différences TP1 vs Projet

### learnedHeuristic.py
Nouveau module dédié à l’apprentissage :
- extraction des caractéristiques (features)
- génération du jeu de données (X, y)
- entraînement du régresseur (scikit-learn)
- sauvegarde et chargement du modèle (joblib)

Ce fichier est indépendant de l’exécution de Pacman.

---

### trainFoodHeuristic.py
Script d’entraînement hors ligne :
- génère des instances aléatoires de FoodSearchProblem
- calcule les coûts optimaux à l’aide de A*
- entraîne un modèle de régression linéaire
- sauvegarde le modèle appris sur disque

Ce script ne lance pas de partie Pacman.

---

### Modifications dans searchAgents.py
Ajout de l’agent :
- LearnedAStarFoodAgent

Cet agent :
- charge le modèle appris
- définit une heuristique apprise h(n)
- utilise A* sur FoodSearchProblem

---

## Procédure d’exécution

### 1. Entraîner l’heuristique
À exécuter une seule fois (ou après modification des features) :

```python trainFoodHeuristic.py```

La commande produit le fichier suivant :
food_model.joblib

Ce fichier contient le modèle de régression linéaire entraîné à partir des données générées par A*.

---

### Lancer Pacman avec l’heuristique apprise

Une fois le modèle entraîné et sauvegardé, Pacman peut être lancé avec l’agent utilisant l’heuristique apprise.

Commande d’exécution :

```python pacman.py -l smallSearch -p LearnedAStarFoodAgent```

Cette commande :

* charge le modèle food_model.joblib
* extrait les features à chaque état
* prédit le coût restant
* utilise cette prédiction comme heuristique dans A*

---

### Comparaison des heuristiques

Les performances sont comparées selon :

* le nombre de nœuds étendus
* le coût final de la solution
* le temps d’exécution

Exemple de comparaisons :

- Heuristique du TP1
   Commande :
  ```python pacman.py -l trickySearch -p AStarFoodSearchAgent```

- Heuristique apprise
   Commande :
   ```python pacman.py -l trickySearch -p LearnedAStarFoodAgent```

Les résultats observés montrent que :
* l'heuristique apprise n’est pas admissible ni consistante
* Elle peut explorer plus de nœuds
* Les solutions restent généralement proches de l’optimal
