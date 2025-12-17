# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import math, random, time


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class Node:
    """
    A node class for your search algorithms
    """

    def __init__(self, state=None, parent=None, action=None, cost=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = float(cost)  # path cost g from root to this node

        # --- MCTS additions ---
        self.children = {}  # action -> Node
        self.visits = 0
        self.value = 0.0  # accumulated rollout reward

    def generateChildren(self, successor):
        """
        successor: (next_state, action, step_cost) triple from problem.getSuccessors
        """
        child = Node(state=successor[0], parent=self,
                     action=successor[1], cost=self.cost + float(successor[2]))
        self.children[successor[1]] = child
        return child

    def getPath(self):
        """Return actions from root to this node."""
        "*** YOUR CODE HERE ***"
        liste_actions = []
        noeud_courant = self
        while noeud_courant is not None and noeud_courant.action is not None: # todo manque quand on est a racine??
            liste_actions.append(noeud_courant.action) # Donne fin ordre noeud ->root
            # remonte
            noeud_courant = noeud_courant.parent
        liste_actions.reverse() # affecte direct
        return liste_actions

    # ---------- MCTS helpers ----------
    def untriedActions(self, problem):
        """
        Actions available at this state that do not yet have a child.
        
        """
        succ = problem.getSuccessors(self.state)
        available = [a for (_, a, _c) in succ]
        return [a for a in available if a not in self.children]

    def fullyExpanded(self, problem):
        succ = problem.getSuccessors(self.state)
        if not succ:  # dead end
            return True
        return len(self.untriedActions(problem)) == 0

    def bestChild(self, c_explore):
        """
        UCT selection among expanded children.
        If any child is unvisited, pick it immediately (ensure exploration).
        c_explore is the constant C in the UCB1 formula.
        """
        if not self.children:
            return None
        "*** YOUR CODE HERE ***"
        # Reste enfants a visiter If any child is unvisited, pick it immediately (ensure exploration).
        for enfant in self.children.values():
            if enfant.visits ==0:
                return enfant

            # Voir notes, tout visité
        best_ucb1 = -math.inf
        enfant_explorer = None
        for enfant_courant in self.children.values():
            nb_playouts_n = enfant_courant.visits # N(n)
            nb_playouts_n_parent = self.visits # N(Parent(n))

            exploitation = enfant_courant.value / nb_playouts_n
            exploration = c_explore * math.sqrt(math.log(max(1, nb_playouts_n_parent)) / nb_playouts_n)
            ucb1_courant = exploitation + exploration
            if best_ucb1 <= ucb1_courant:
                enfant_explorer = enfant_courant
                best_ucb1 = ucb1_courant

        return enfant_explorer


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()  ## Liste Ouverte, LIFO
    pos_depart = problem.getStartState()
    stack.push((pos_depart,[],0))
    pos_visites = []

    if problem.isGoalState(pos_depart): return [] ## On commence au goal

    while not stack.isEmpty():

        pos, chemin, cout = stack.pop()
        if pos in pos_visites:
            continue
        pos_visites.append(pos)

        if problem.isGoalState(pos):
            return chemin
        else:
            ## Classe PositionSearchProblem: this should return a list of triples,(successor, action, stepCost)
            noeudSucc = problem.getSuccessors(pos)

            ## Etendre le noeud et ajouter au stack
            for noeudEnfant in noeudSucc:
                if noeudEnfant[0] not in pos_visites: # Ne revisite pas les memes
                    nouv_chemin = chemin + [noeudEnfant[1]]
                    ##print("CHEMIN ACTION: " + noeudEnfant[1])
                    stack.push((noeudEnfant[0], nouv_chemin, 1))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()  ## Liste Ouverte, FIFO
    pos_depart = problem.getStartState()
    queue.push((pos_depart, [], 0))
    pos_visites = [] ## set() peut aussi

    if problem.isGoalState(pos_depart): return []  ## On commence au goal

    while (True):
        if queue.isEmpty(): return []  ## Queue vide, Failure

        pos, chemin, cout = queue.pop()

        if pos in pos_visites:
            continue
        pos_visites.append(pos)

        if problem.isGoalState(pos): return chemin
        else:
            ## Classe PositionSearchProblem: this should return a list of triples,(successor, action, stepCost)
            noeudSucc = problem.getSuccessors(pos)

            ## Étendre le noeud et ajouter à la queue
            for noeudEnfant in noeudSucc:
                if noeudEnfant[0] not in pos_visites:
                    nouv_chemin = chemin + [noeudEnfant[1]]
                    queue.push((noeudEnfant[0], nouv_chemin, 1))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()  ## Liste Ouverte, avec priorités (coût total)
    pos_depart = problem.getStartState()
    queue.push((pos_depart, [], 0), 0) # priorité initial 0
    pos_visites = []

    if problem.isGoalState(pos_depart): return []  ## On commence au goal

    while (True):
        if queue.isEmpty(): return []  ## Queue vide, Failure, pareil que largeur si couts pareils

        pos, chemin, cout = queue.pop()
        if pos in pos_visites:
            continue
        pos_visites.append(pos)

        if problem.isGoalState(pos): return chemin
        else:
            ## Classe PositionSearchProblem: this should return a list of triples,(successor, action, stepCost)
            noeudSucc = problem.getSuccessors(pos)

            ## Étendre le noeud et ajouter à la queue avec la priorité du coût total
            for noeudEnfant in noeudSucc:
                pos_enfant, direction, cout_step = noeudEnfant ## cout_step = 1

                if pos_enfant not in pos_visites: # Ne reviste pas les memes etats
                    nouv_chemin = chemin + [direction]
                    nouv_cout = cout + cout_step
                    queue.push((pos_enfant, nouv_chemin, nouv_cout), nouv_cout)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()  ## Liste Ouverte, avec priorités (coût total + heuristique)
    pos_depart = problem.getStartState()
    queue.push((pos_depart, [], 0), heuristic(pos_depart, problem))  # priorité initiale = h(start)
    pos_visites = []

    if problem.isGoalState(pos_depart): return []  ## On commence au goal

    while (True):
        if queue.isEmpty(): return []  ## Queue vide, Failure

        pos, chemin, cout = queue.pop()
        if pos in pos_visites:
            continue
        pos_visites.append(pos)

        if problem.isGoalState(pos):
            return chemin
        else:
            ## Classe PositionSearchProblem: this should return a list of triples,(successor, action, stepCost)
            noeudSucc = problem.getSuccessors(pos)

            ## etendre le noeud et ajouter à la queue avec la priorité cout + heuristique
            for noeudEnfant in noeudSucc:
                pos_enfant, direction, delta_cout = noeudEnfant

                if pos_enfant not in pos_visites:  # Ne revisite pas les mêmes états
                    nouv_chemin = chemin + [direction]
                    nouv_cout = cout + delta_cout
                    ## f(n) = g(n) + h(n) = Cout pour se rendre + residuel
                    priorite = nouv_cout + heuristic(pos_enfant, problem)
                    queue.push((pos_enfant, nouv_chemin, nouv_cout), priorite) ## + petit = + priorite






def _rollout_random(problem, start_state, max_steps):
    """
    Simulate uniformly at random up to max_steps.
    Returns (tail_actions, tail_cost, reached_goal_bool).
    tail_actions is the list of actions taken during the rollout,
    tail_cost is the cost of the path in the rollout,
    reached_goal_bool is a boolean declaring if a goal has been reached in the rollout.
    """
    s = start_state
    actions = []
    total_cost = 0.0
    for _ in range(max_steps):
        if problem.isGoalState(s):
            return actions, total_cost, True
        succ = problem.getSuccessors(s)
        if not succ:
            break
        s2, a, c = random.choice(succ)
        s = s2
        actions.append(a)
        total_cost += float(c)
    if problem.isGoalState(s):
        return actions, total_cost, True
    return actions, total_cost, False


def monteCarloTreeSearch(problem,
                         iterations=5000,
                         rollout_depth=100,
                         c_explore=math.sqrt(2.0),
                         gamma=0.993):
    """
    Offline MCTS with selection policy UCT.

    Args:
        iterations     : number of MCTS iterations to run
        rollout_depth  : maximum steps per rollout
        c_explore      : UCT exploration constant c (sqrt(2) is standard)
        gamma          : cost-to-reward decay; reward = gamma ** total_cost (for successful rollouts)

    Returns:
        List of actions (best plan found). If no complete plan discovered, returns a best-effort prefix
        by following the most-visited child chain.
    """

    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    root = Node(state=start, parent=None, action=None, cost=0.0)
    incumbent_plan = None
    incumbent_cost = float("inf")

    #  incumbent_plan is the best complete start→goal action sequence found so far in this MCTS run.
    #  None means no successful rollout has reached a goal yet.
    #  incumbent_cost is the total path cost of incumbent_plan. Initialized to +inf.
    #  Updated only when MCTS discovers a complete path with strictly lower cost.

    for _ in range(iterations):
        node = root

        # ----- SELECTION ----- La sélection descend l’arbre jusqu’à un nœud non entièrement développé ou terminal.
        "*** YOUR CODE HERE ***"
        ## pacman flicker (quick look back)
        trajectoire_sim = [node]
        while True:
            # Atteint le but ou pogner ou enfants pas encore eu dexpension (ex: root choisi pour expension)
            successeurs = problem.getSuccessors(node.state)
            if problem.isGoalState(node.state) or not node.fullyExpanded(problem) or not successeurs:
                break

            # Choisi enfant selon mcts
            enfant_suivant = node.bestChild(c_explore)
            # Pas d'enfants
            if enfant_suivant is None: break
            node = enfant_suivant
            trajectoire_sim.append(node)

        # ----- EXPANSION ----- Ajout dans noeud
        "*** YOUR CODE HERE ***"
        # Pas atteint le but
        if not problem.isGoalState(node.state):
            actions_untried = node.untriedActions(problem)
            # Des actions a essayer
            if actions_untried:
                # Choisir n'importequel option pour implémenter dans l'arbre de noeuds
                action_choisie = random.choice(actions_untried)
                # Ajout enfant dans arbre
                for (etat_suiv, action, cout_etape) in problem.getSuccessors(node.state):
                    if action == action_choisie:
                        node = node.generateChildren((etat_suiv, action, cout_etape))
                        trajectoire_sim.append(node)
                        break

        # ----- ROLLOUT  ----- Simulation
        "*** YOUR CODE HERE ***"
        # Val debut pour simul alea
        # Lorsqu’on atteint une feuille de l’arbre courant, générer un
        # enfant ‘n’ et initialiser U(n)=0, N(n)=0.
        etat = node.state
        goal = problem.isGoalState(etat)
        depth, actions_sim, cout_rollout = 0, [], 0.0

        # Simulation: générer un rollout à partir de ‘n’ jusqu’à la fin
        # de la partie pour obtenir un résultat
        while (not goal) and depth < rollout_depth:
            successeurs = problem.getSuccessors(etat)

            if not successeurs:
                break
            etat2, action2, cout2 = random.choice(successeurs)

            actions_sim.append(action2)
            cout_rollout += float(cout2)
            etat = etat2
            depth += 1
            goal = problem.isGoalState(etat)

        # On utilise alors la formule suivante pour la recompense: (seul. dans l'énoncé...)
        if goal:
            c_prefixe = node.cost  # g(noeud)
            c_total = c_prefixe + cout_rollout
            r = gamma ** c_total
            # Chemin meilleur si c_total petit
            if c_total < incumbent_cost:
                incumbent_cost = c_total
                incumbent_plan = node.getPath() + actions_sim
        else:
            r = 0.0

        # print(r)

        # ----- BACKPROP -----
        "*** YOUR CODE HERE ***"
        # Ajuste U(n) et N(n) à chacun des noeuds de l'arbre
        for nd in trajectoire_sim:
            nd.visits += 1
            nd.value += r

    # PROF PAS TOUCHE EN BAS

    # Si un plan complet a été trouvé
    if incumbent_plan is not None:
        return incumbent_plan

    # Fallback: follow most-visited children to produce a sensible prefix
    actions, n = [], root
    while n.children:
        n = max(n.children.values(), key=lambda ch: ch.visits)
        if n.action is None:
            break
        actions.append(n.action)
        if problem.isGoalState(n.state):
            break
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mcts = monteCarloTreeSearch
