# Entraîne un régresseur sklearn pour prédire le coût optimal (A*) de FoodSearchProblem.

from layout import getLayout
from pacman import ClassicGameRules
from textDisplay import NullGraphics

from searchAgents import foodHeuristic, FoodSearchProblem
import search

from learnedHeuristic import generateDataset, trainRegressor, saveModel
import random


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
        MAX_FOOD = 6
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
    states = generateRandomStates(layoutName="smallSearch", n=200)
    print("Nombre d'états générés :", len(states))

    print("Génération du dataset (A*)...")
    X, y = generateDataset(states, heuristic=foodHeuristic, FoodSearchProblem=FoodSearchProblem)
    print("Taille du dataset :", len(X))

    print("Entraînement du modèle...")
    model = trainRegressor(X, y)

    saveModel(model, "food_model.joblib")
    print("Modèle sauvegardé : food_model.joblib")
