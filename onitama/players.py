from board import Board
from constants import *
import random
import math
import numpy as np
from card import Card
from heuristic import HeuristicEvaluation
import sys
sys.path.append('../api/')
from exceptions import InvalidMoveException

# Classe générale pour un joueur
class Player:
    def __init__(self):
        self.position = None

    def set_position(self, position:int):
        self.position = position

# Joueur purement aléatoire
class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = "RandomPlayer"

    def play(self, board:Board):
        available_moves = board.get_available_moves()
        if len(available_moves) == 0:
            return None
        return random.choice(available_moves)

# Joueur "humain" API (utilisé pour les APIs)
class ApiPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = "ApiPlayer"
        self.from_pos = None
        self.to_pos = None
        self.card_idx = None
    
    def set_next_move(self, from_pos:tuple, to_pos:tuple, card_idx:int):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.card_idx = card_idx

    def play(self, board:Board):
        available_moves = board.get_available_moves()
        for move in available_moves:
            if move.from_pos == self.from_pos and move.card_idx == self.card_idx and move.to_pos == self.to_pos:
                return move
            
        raise InvalidMoveException()



# Joueur "humain" (pour tester en mode console)
class HumanPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = "HumanPlayer"

    def play(self, board:Board):
        available_moves = board.get_available_moves()

        player_student = PLAYER_ONE_STUDENT if self.position == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        player_master = PLAYER_ONE_MASTER if self.position == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        
        while True:
            try:
                from_position = input("Enter the position of the student ou master you would loke to move. (Ex. A1)  ").strip().upper()
                from_col = ord(from_position[0]) - ord('A')
                from_row = int(from_position[1:]) - 1
                if board.board[from_col][from_row] not in [player_student, player_master]:
                    print("You must choose one of your students or your master")
                else:
                    break
            except (IndexError, ValueError):
                print("This input is not correct !")

        print("Which action would you like to do ? ")
        actions = []
        for action in available_moves:
            if from_col == action.from_pos[0] and from_row == action.from_pos[1]:
                actions.append(action)

        for i in range(len(actions)):
            print(f"{i}. Card {Card.getCard(card_idx=actions[i].card_idx).name} to {chr(65 + actions[i].to_pos[0])}:{actions[i].to_pos[1]+1}")

        
        while True:
            try:
                selected_action = int(input("Select the action number. (Ex. 0)  ").strip().upper())
                return actions[selected_action]
            except Exception:
                print("Incorrect action selection !")        

# Joueur utilisant des règles d'heuristiques pour déterminer la meilleure action à utiliser
# heuristic_function:str : permet de spécifier la fonction de la classe HeuristicEvaluation à utiliser
class HeuristicPlayer(Player):
    def __init__(self, heuristic_function:str="heuristic_regular"):
        super().__init__()
        self.name = "HeuristicPlayer"
        self.heuristic_function = heuristic_function

    def play(self, board:Board):
        best_move = None
        best_score = float('-inf')

        #On récupère les mouvements possibles
        available_moves = board.get_available_moves()
        if len(available_moves) == 0:
            return None
        random.shuffle(available_moves)

        #On joue chaque action et on regarde le score qu'on obtient, on garde la meilleure action
        for action in available_moves:
            last_move = board.play_move(action=action)
            score = getattr(HeuristicEvaluation, self.heuristic_function)(board=board, from_current_player_point_of_view=False)
            board.cancel_last_move(last_move=last_move)

            if score > best_score:
                best_score = score
                best_move = action

        return best_move
    

# Joueur utilisant des règles d'heuristique + un algorithme minimax sur N niveaux
# max_depth:int : niveau max de profondeur de l'algo minimax
# heuristic_function:str : permet de spécifier la fonction de la classe HeuristicEvaluation à utiliser
class LookAheadHeuristicPlayer(Player):
    def __init__(self, max_depth:int=2, heuristic_function:str="heuristic_evaluation"):
        super().__init__()
        self.max_depth = max_depth
        self.original_player = None  # Pour savoir qui est l'IA
        self.name = "LookAheadHeuristicPlayer"
        self.heuristic_function = heuristic_function

    def play(self, board:Board):
        #Minimax algo
        best_move = None
        best_score = float('-inf')

        # On mémorise notre position pour l'évaluation terminale
        self.original_player = self.position

        #On récupère les actions possibles
        available_actions = board.get_available_moves()

        if len(available_actions) == 0:
            return None

        for action in available_actions:
            #On joue le coup
            last_move = board.play_move(action=action)
            #On descend d'un niveaux
            score = self._minimax(board=board, depth=0, is_maximizing=False)

            #On annule le coup
            board.cancel_last_move(last_move=last_move)

            if score > best_score:
                best_score = score
                best_move = action

        return best_move

    def _minimax(self, board:Board, depth:int, is_maximizing:bool, consecutive_default_moves:int=0):
        #On vérifie dabord si dans cet état, le jeu est terminé
        game_ended, winner = board.game_has_ended()
        if game_ended:
            # On retourne un score positif si l'IA gagne, négatif sinon
            if winner == self.original_player:
                return 1000
            else:
                return -1000

        #On a atteint le niveau maximum, on retourne le score de l'état
        if depth >= self.max_depth:
            # Le score doit être du point de vue de l'IA (original_player)
            # is_maximizing=True signifie que c'est le tour de l'IA
            return getattr(HeuristicEvaluation, self.heuristic_function)(board, from_current_player_point_of_view=is_maximizing)

        #Coups disponibles à partir de cette position
        available_moves = board.get_available_moves()

        # Si aucun coup n'est disponible, on joue le coup par défaut (échange de carte)
        if len(available_moves) == 0:
            # Protection contre les boucles infinies de default moves
            if consecutive_default_moves >= 4:
                # Les deux joueurs sont bloqués, retourner une évaluation neutre
                return getattr(HeuristicEvaluation, self.heuristic_function)(board, from_current_player_point_of_view=is_maximizing)
            last_default_move = board.play_default_move()
            score = self._minimax(board=board, depth=depth+1, is_maximizing=not is_maximizing, consecutive_default_moves=consecutive_default_moves+1)
            board.cancel_default_move(last_default_move)
            return score

        best_score = float('-inf') if is_maximizing else float('inf')

        for action in available_moves:
            last_move = board.play_move(action=action)
            score = self._minimax(board=board, depth=depth+1, is_maximizing=not is_maximizing, consecutive_default_moves=0)
            board.cancel_last_move(last_move=last_move)

            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score


class MCTSNode:
    def __init__(self, parent=None, action=None, depth=0):
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = None

    def ucb1(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_constant=1.41):
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def best_action(self):
        return max(self.children, key=lambda c: c.visits).action

# Joueur implémentant la technique de Monte Carlo Tree Search
class MCTSPlayer(Player):
    def __init__(self, num_simulations:int=1000, exploration_constant:float=1.41):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.name = "MCTSPlayer"

    def play(self, board:Board):
        root = MCTSNode()
        root.untried_actions = board.get_available_moves()

        if len(root.untried_actions) == 0:
            return None
        
        original_player = board.current_player

        # On effectue N simulations
        for _ in range(self.num_simulations):
            node = root
            moves_stack = []

            # Selection : descendre dans l'arbre en choisissant le meilleur enfant
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = node.best_child(self.exploration_constant)
                last_move = board.play_move(node.action)
                moves_stack.append((False, last_move))

            # Expansion : si le noeud a des actions non essayées, en choisir une
            if node.untried_actions is not None and len(node.untried_actions) > 0:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                last_move = board.play_move(action)
                moves_stack.append((False, last_move))

                child = MCTSNode(parent=node, action=action, depth=node.depth + 1)
                child.untried_actions = board.get_available_moves()
                node.children.append(child)
                node = child

            # Simulation : jouer aléatoirement jusqu'à la fin
            game_ended, winner = board.game_has_ended()
            while not game_ended:
                available_moves = board.get_available_moves()
                if available_moves:
                    action = random.choice(available_moves)
                    last_move = board.play_move(action)
                    moves_stack.append((False, last_move))
                else:
                    last_move = board.play_default_move()
                    moves_stack.append((True, last_move))
                game_ended, winner = board.game_has_ended()

            # Calculer le résultat
            if winner == original_player:
                result = 1
            elif winner is None:
                result = 0.5
            else:
                result = 0

            # Backpropagation : remonter les résultats en alternant la perspective
            # La perspective de départ dépend de la profondeur du nœud feuille :
            # - profondeur impaire = original_player a joué pour atteindre ce nœud → current_result = result
            # - profondeur paire = l'adversaire a joué pour atteindre ce nœud → current_result = 1 - result
            current_result = result if node.depth % 2 == 1 else 1 - result
            while node is not None:
                node.visits += 1
                node.wins += current_result
                current_result = 1 - current_result
                node = node.parent

            # Annuler tous les coups joués
            while moves_stack:
                is_default, last_move = moves_stack.pop()
                if is_default:
                    board.cancel_default_move(last_move)
                else:
                    board.cancel_last_move(last_move)

        return root.best_action()


class MCTSHeuristicPlayer(Player):
    """
    MCTS avec rollouts guidés par l'heuristique au lieu de coups aléatoires.
    Utilise une stratégie epsilon-greedy : avec probabilité epsilon, joue aléatoirement,
    sinon joue le meilleur coup selon l'heuristique.
    """
    def __init__(self, num_simulations:int=1000, exploration_constant:float=1.41, epsilon:float=0.1):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.epsilon = epsilon  # Probabilité de jouer aléatoirement (pour exploration)
        self.name = "MCTSHeuristicPlayer"

    def _select_rollout_action(self, board:Board):
        """Sélectionne une action pour le rollout en utilisant l'heuristique."""
        available_moves = board.get_available_moves()

        if not available_moves:
            return None

        # Epsilon-greedy : parfois on joue aléatoirement pour diversifier
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        # Sinon, on choisit le meilleur coup selon l'heuristique
        best_move = None
        best_score = float('-inf')

        for action in available_moves:
            last_move = board.play_move(action)
            # Évaluer du point de vue du joueur qui vient de jouer (False car on a changé de joueur)
            score = board.heuristic_evaluation(from_current_player_point_of_view=False)
            board.cancel_last_move(last_move)

            if score > best_score:
                best_score = score
                best_move = action

        return best_move

    def play(self, board:Board):
        root = MCTSNode()
        root.untried_actions = board.get_available_moves()

        if len(root.untried_actions) == 0:
            return None
        
        original_player = board.current_player

        for _ in range(self.num_simulations):
            node = root
            moves_stack = []

            # Selection : descendre dans l'arbre en choisissant le meilleur enfant
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = node.best_child(self.exploration_constant)
                last_move = board.play_move(node.action)
                moves_stack.append((False, last_move))

            # Expansion : si le noeud a des actions non essayées, en choisir une
            if node.untried_actions is not None and len(node.untried_actions) > 0:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                last_move = board.play_move(action)
                moves_stack.append((False, last_move))

                child = MCTSNode(parent=node, action=action, depth=node.depth + 1)
                child.untried_actions = board.get_available_moves()
                node.children.append(child)
                node = child

            # Simulation : jouer avec l'heuristique jusqu'à la fin
            game_ended, winner = board.game_has_ended()
            while not game_ended:
                action = self._select_rollout_action(board)
                if action is not None:
                    last_move = board.play_move(action)
                    moves_stack.append((False, last_move))
                else:
                    last_move = board.play_default_move()
                    moves_stack.append((True, last_move))
                game_ended, winner = board.game_has_ended()

            # Calculer le résultat
            if winner == original_player:
                result = 1
            elif winner is None:
                result = 0.5
            else:
                result = 0

            # Backpropagation : perspective de départ selon la profondeur du nœud feuille
            current_result = result if node.depth % 2 == 1 else 1 - result
            while node is not None:
                node.visits += 1
                node.wins += current_result
                current_result = 1 - current_result
                node = node.parent

            # Annuler tous les coups joués
            while moves_stack:
                is_default, last_move = moves_stack.pop()
                if is_default:
                    board.cancel_default_move(last_move)
                else:
                    board.cancel_last_move(last_move)

        return root.best_action()


class MCTSEvalPlayer(Player):
    """
    MCTS avec évaluation directe au lieu de rollouts.
    Au lieu de simuler une partie jusqu'à la fin, évalue directement la position
    avec l'heuristique et convertit le score en probabilité de victoire.
    """
    def __init__(self, num_simulations:int=1000, exploration_constant:float=1.41):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.name = "MCTSEvalPlayer"

    def _score_to_result(self, score:float, original_player:int, current_player:int):
        """
        Convertit un score heuristique en probabilité de victoire [0, 1].
        Le score est du point de vue du current_player.
        On veut un résultat du point de vue de original_player.
        """
        # Normaliser le score avec une sigmoïde
        # score ~= 0 → 0.5, score >> 0 → 1, score << 0 → 0
        result = 1 / (1 + math.exp(-score / 100))

        # Si le score était du point de vue de l'adversaire, inverser
        if current_player != original_player:
            result = 1 - result

        return result

    def play(self, board:Board):
        root = MCTSNode()
        root.untried_actions = board.get_available_moves()

        if len(root.untried_actions) == 0:
            return None
        
        original_player = board.current_player

        for _ in range(self.num_simulations):
            node = root
            moves_stack = []

            # Selection : descendre dans l'arbre en choisissant le meilleur enfant
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = node.best_child(self.exploration_constant)
                last_move = board.play_move(node.action)
                moves_stack.append((False, last_move))

            # Expansion : si le noeud a des actions non essayées, en choisir une
            if node.untried_actions is not None and len(node.untried_actions) > 0:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                last_move = board.play_move(action)
                moves_stack.append((False, last_move))

                child = MCTSNode(parent=node, action=action, depth=node.depth + 1)
                child.untried_actions = board.get_available_moves()
                node.children.append(child)
                node = child

            # Évaluation directe au lieu de rollout
            game_ended, winner = board.game_has_ended()
            if game_ended:
                # Partie terminée : résultat binaire
                if winner == original_player:
                    result = 1
                elif winner is None:
                    result = 0.5
                else:
                    result = 0
            else:
                # Évaluer la position avec l'heuristique
                score = board.heuristic_evaluation(from_current_player_point_of_view=True)
                result = self._score_to_result(score, original_player, board.current_player)

            # Backpropagation : perspective de départ selon la profondeur du nœud feuille
            current_result = result if node.depth % 2 == 1 else 1 - result
            while node is not None:
                node.visits += 1
                node.wins += current_result
                current_result = 1 - current_result
                node = node.parent

            # Annuler tous les coups joués
            while moves_stack:
                is_default, last_move = moves_stack.pop()
                if is_default:
                    board.cancel_default_move(last_move)
                else:
                    board.cancel_last_move(last_move)

        return root.best_action()

