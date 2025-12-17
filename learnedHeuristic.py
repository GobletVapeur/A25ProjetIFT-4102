# Apprentissage d'une heuristique pour FoodSearchProblem avec scikit-learn.
# - X : features extraites de l'état (position + nourriture restante)
# - y : coût optimal obtenu par A* (mais avec une heuristique CHEAP pour éviter l'explosion)

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

import search
import util
from searchAgents import foodHeuristic


def extractFeatures(state):
    """
    Transforme un état (pacmanPosition, foodGrid) en vecteur de caractéristiques.
    state : ( (x,y), foodGrid )
    return : vecteur caractéristiques x
    """
    position, foodGrid = state
    foods = foodGrid.asList()

    num_food = len(foods)

    # Aucun objectif restant
    if num_food == 0:
        return np.array([0, 0, 0, 0, 0], dtype=float)

    # Distances Manhattan aux nourritures restantes (trained with fixed walls)
    # Avec MazeDistance, faudrait faire BFS (nb_dataset * nb_nourritures)
    distToFoodList = [util.manhattanDistance(position, f) for f in foods]

    min_dist = min(distToFoodList)
    max_dist = max(distToFoodList)
    mean_dist = sum(distToFoodList) / len(distToFoodList)

    # Aire du rectangle englobant les nourritures (indicateur de dispersion)
    xs = [f[0] for f in foods]
    ys = [f[1] for f in foods]
    bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)

    return np.array(
        [num_food, min_dist, max_dist, mean_dist, bbox_area],
        dtype=float
    )


def manhattanDistanceFoodHeuristic(state, problem=None):
    """
    Heuristique admissible et très rapide pour FoodSearchProblem.

    Important:
    - On NE veut PAS utiliser foodHeuristic du TP1 ici (qui appelle mazeDistance/BFS),
      sinon A* devient pratiquement inutilisable pour générer un dataset.
    - Cette heuristique est une borne inférieure simple: max distance Manhattan vers une nourriture.
    """
    position, foodGrid = state
    foods = foodGrid.asList()
    if not foods:
        return 0
    return max(util.manhattanDistance(position, f) for f in foods)


def generateDataset(gameStates, heuristic, FoodSearchProblem):
    """
    Génère (X, y) à partir d'une liste de GameState.

    Paramètres:
    - gameStates : liste de GameState (Pacman)
    - heuristic  : (gardé pour compatibilité avec ton appel) MAIS on ne l'utilise pas ici
    - FoodSearchProblem : classe du problème (SearchProblem)

    y:
    - coût optimal via A*.
    - On utilise cheapFoodHeuristic pour que la génération soit faisable.
    """
    X = []
    y = []

    for index, gameState in enumerate(gameStates):
        # Feedback minimal pour voir l'avancement
        print(f"A* (labels) sur état {index + 1}/{len(gameStates)}")

        problem = FoodSearchProblem(gameState)

        actions = search.aStarSearch(problem, foodHeuristic)
        cost = problem.getCostOfActions(actions)

        X.append(extractFeatures(problem.getStartState()))
        y.append(cost)

    return np.array(X), np.array(y)


def trainRegressor(X, y):
    """
    Entraîne un régresseur sklearn qui prédit y à partir de X.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def saveModel(model, filename="food_model.joblib"):
    """
    Sauvegarde model entrainé
    """
    joblib.dump(model, filename)


def loadModel(filename="food_model.joblib"):
    """Returns : Model appris"""
    return joblib.load(filename)
