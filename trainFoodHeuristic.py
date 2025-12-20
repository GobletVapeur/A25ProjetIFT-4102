# Entraîne un régresseur sklearn pour prédire le coût optimal (A*) de FoodSearchProblem.

from layout import getLayout
from pacman import ClassicGameRules
from textDisplay import NullGraphics
from searchAgents import FoodSearchProblem
from learnedHeuristic import *
import random

LAYOUTS = [
    "tinySearch",
    "smallSearch",
    "mediumSearch",
    "trickySearch",
    "boxSearch",
]


def generateStatesFromLayouts(layoutNames, n_per_layout=100):
    all_states = []

    for layoutName in layoutNames:
        print(f"Génération des états pour {layoutName}")
        states = generateRandomStates(
            layoutName=layoutName,
            n=n_per_layout
        )
        all_states.extend(states)

    print(f"Total GameStates générés: {len(all_states)}")
    return all_states


def generateRandomStates(layoutName="smallSearch", n=200):
    """
    Génère n GameState initiaux à partir d'un layout, sans jouer de partie.
    On randomise ensuite la grille de nourriture (foodGrid) et la pos de
    de pacman pour créer des instances différentes.
    """
    layout = getLayout(layoutName)
    rules = ClassicGameRules()
    graphics = NullGraphics()

    states = []

    for _ in range(n):
        game = rules.newGame(layout, [], [], graphics, quiet=True)
        state = game.state

        # Randomise la nourriture
        food = state.getFood().copy()
        foodList = food.asList()

        # Varie le temps d'exécution (+noeuds A*, profondeur)
        MAX_FOOD = 10
        foodQuantity = random.randint(1, min(MAX_FOOD, len(foodList)))
        foodPosList = set(random.sample(foodList, foodQuantity))

        for x, y in foodList:
            if (x, y) not in foodPosList:
                food[x][y] = False

        # Injecte la nouvelle grille de nourriture dans l'état.
        state.data.food = food

        # Randomise la position initiale de Pacman (non-joué)
        walls = state.getWalls()
        availableSpace = []

        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y] and not food[x][y]:
                    availableSpace.append((x, y))

        state.data.agentStates[0].configuration.pos = random.choice(availableSpace)
        states.append(state)

    return states


if __name__ == "__main__":
    print("Génération des états...")
    '''Nous privilégions un grand nombre d’instances simples plutôt qu’un petit nombre d’instances complexes afin
     d’assurer une génération efficace de labels optimaux.'''
    states = generateStatesFromLayouts(layoutNames=LAYOUTS, n_per_layout=100)
    print("Nombre d'états générés :", len(states))
    selected_features = FEATURE_CONFIGS.get(0)
    print("Etude d'ablation (y/n)?")
    i = str(input())
    if i == "y":
        run_ablation(states, FoodSearchProblem)


    print("Sortie Lasso sur les 9 features (y/n)?")
    i = str(input())
    if i == "y":
        feature_list = FEATURE_CONFIGS["full"]
        x, y = generateDataset(
            states,
            heuristic=foodHeuristic,
            FoodSearchProblem=FoodSearchProblem,
            featureList=feature_list)

        selected_features = run_lasso(x, y, feature_list, alpha=0.01)
        print(f"Selon Lasso:\n{selected_features}")
        bestFeatureList = selected_features

    else:
        selected_features = FEATURE_CONFIGS["stable_only"]
    bestFeatureList = selected_features
    print(f"Générer dataset pour {bestFeatureList} (y/n)?")
    i = str(input())
    if (i == "y") & (bestFeatureList != []):
        print("Génération du dataset (A*)...")

        x, y = generateDataset(states, heuristic=foodHeuristic, FoodSearchProblem=FoodSearchProblem,
                               featureList=bestFeatureList)
        print("Taille du dataset :", len(x))

        print("Entraînement du modèle...")
        model = trainRegressor(x, y)

        saveModel(model, "food_model.joblib")
        print("Modèle sauvegardé : food_model.joblib")
