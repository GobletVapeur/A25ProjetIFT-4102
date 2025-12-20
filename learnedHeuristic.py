# Apprentissage d'une heuristique pour FoodSearchProblem avec scikit-learn.
# - X : features extraites de l'état (position + nourriture restante)
# - y : coût optimal obtenu par A* (mais avec une heuristique CHEAP pour éviter l'explosion)

import time

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

import search
import util
from searchAgents import foodHeuristic


def extractFeatures(state, feature_list):
    position, foodGrid = state
    foods = foodGrid.asList()

    if not foods:
        return np.zeros(len(feature_list), dtype=float)

    x_pac, y_pac = position
    w, h = float(foodGrid.width), float(foodGrid.height)

    # Distances Manhattan vers toutes les nourritures
    dists = [util.manhattanDistance(position, f) for f in foods]

    xs = np.array([f[0] for f in foods], dtype=float)
    ys = np.array([f[1] for f in foods], dtype=float)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    bbox_area = (max_x - min_x + 1.0) * (max_y - min_y + 1.0)
    log_bbox_area = float(np.log(bbox_area + 1.0))

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    dist_center = abs(x_pac - center_x) + abs(y_pac - center_y)

    values = {
        # Position Pacman (normalisée)
        "pacman_x": x_pac / w,
        "pacman_y": y_pac / h,

        # Nourriture / distances
        "num_food": float(len(foods)),
        "min_dist": float(min(dists)),
        "max_dist": float(max(dists)),
        "mean_dist": float(sum(dists) / len(dists)),

        # Versions log (stabilisation d'échelle)
        "log_num_food": float(np.log(len(foods) + 1.0)),
        "log_max_dist": float(np.log(max(dists) + 1.0)),
        "log_mean_dist": float(np.log((sum(dists) / len(dists)) + 1.0)),

        # Géométrie globale
        "bbox_area": float(bbox_area),
        "log_bbox_area": log_bbox_area,
        "var_x": float(xs.var()),
        "var_y": float(ys.var()),
        "dist_center": float(dist_center),
    }
    return np.array([values[name] for name in feature_list], dtype=float)


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


def generateDataset(gameStates, heuristic, FoodSearchProblem, featureList):
    X = []
    y = []

    if featureList is None:
        Exception("No features provided")

    for gs in gameStates:
        problem = FoodSearchProblem(gs)

        actions = search.aStarSearch(problem, heuristic)
        cost = problem.getCostOfActions(actions)

        X.append(extractFeatures(problem.getStartState(), featureList))
        y.append(cost)

    return np.array(X), np.array(y)


def trainRegressor(x, y):
    """
    Entraîne un régresseur sklearn qui prédit y à partir de X.
    """
    model = LinearRegression()
    model.fit(x, y)
    return model


def saveModel(model, filename="food_model.joblib"):
    """
    Sauvegarde model entrainé
    """
    joblib.dump(model, filename)


def loadModel(filename="food_model.joblib"):
    """Returns : Model appris"""
    return joblib.load(filename)

# Avec Lasso a=0.01, log_num_food et log_mean_dist -> 0
FEATURE_CONFIGS = dict(#baseline=["num_food", "min_dist", "max_dist", "mean_dist", "bbox_area"],
                       #no_bbox=["num_food", "min_dist", "max_dist", "mean_dist"],
                       #log_bbox=["num_food", "min_dist", "max_dist", "mean_dist", "log_bbox_area"],
                       #no_max=["num_food", "min_dist", "mean_dist", "bbox_area"],
                       #dist_only=["min_dist", "max_dist", "mean_dist"],
                       #num_and_dist=["num_food", "min_dist", "mean_dist"],
                       stable_only=["log_num_food", "min_dist", "log_bbox_area", "log_mean_dist", "var_y", "dist_center"],
                       giroux = ["pacman_x","pacman_y", "num_food", "min_dist", "max_dist", "mean_dist", "var_x", "var_y", "dist_center"]
                       #,full=["pacman_x", "pacman_y", "num_food", "min_dist", "max_dist","mean_dist","log_num_food","log_max_dist", "log_mean_dist","bbox_area","log_bbox_area","var_x","var_y","dist_center",]
)


def rmse(y_theo, y_pred):
    return np.sqrt(np.mean((y_theo - y_pred) ** 2))

def run_ablation(gameStates, FoodSearchProblem):
    for name, feature_list in FEATURE_CONFIGS.items():
        print(f"Configuration : {name}")
        print(f"Features       : {feature_list}")

        X, y = [], []

        # 1) Génération des labels optimaux (A*)
        for gs in gameStates:
            problem = FoodSearchProblem(gs)
            actions = search.aStarSearch(problem, foodHeuristic)
            cost = problem.getCostOfActions(actions)

            X.append(extractFeatures(problem.getStartState(), feature_list))
            y.append(cost)

        X = np.array(X)
        y = np.array(y)

        # 2) Entraînement
        model = LinearRegression()
        model.fit(X, y)

        # 2.1) Affichage des poids
        print("Poids du modèle :")
        for fname, coef in zip(feature_list, model.coef_):
            print(f"  {fname:15s} : {coef:.5f}")
        print(f"  intercept      : {model.intercept_:.5f}")

        # 3) RMSE (sur train)
        preds = model.predict(X)
        error = rmse(y, preds)
        print(f"RMSE           : {error:.2f}")

        # 4) Test A* avec heuristique apprise
        total_expanded = 0
        total_cost = 0
        start_time = time.time()

        for gs in gameStates:
            problem = FoodSearchProblem(gs)

            def learnedHeuristic(state, problem=None):
                x = extractFeatures(state, feature_list).reshape(1, -1)
                return max(0.0, model.predict(x)[0])

            actions = search.aStarSearch(problem, learnedHeuristic)
            total_cost += problem.getCostOfActions(actions)
            total_expanded += problem._expanded

        elapsed = time.time() - start_time

        print(f"Noeuds étendus  : {total_expanded}")
        print(f"Coût moyen     : {total_cost / len(gameStates):.2f}")
        print(f"Temps total    : {elapsed:.2f} sec")
        print("-" * 20)


def run_lasso(X, y, feature_list, alpha=0.01):
    """
    Applique LASSO pour sélectionner automatiquement les features.
    """
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)

    print("\nRésultat LASSO")
    print(f"alpha = {alpha}\n")

    kept = []

    for name, coef in zip(feature_list, model.coef_):
        print(f"{name:15s} : {coef:.5f}")
        if abs(coef) > 1e-6:
            kept.append(name)

    print("\nFeatures conservées par LASSO:")
    print(kept)

    return kept
