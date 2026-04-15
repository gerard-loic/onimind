from board import Board
from constants import *
import random
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

