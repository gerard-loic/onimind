from players import Player
from board import Board
import numpy as np

# Joueur utilisant un réseau + un algorithme minimax sur N niveaux
# max_depth:int : niveau max de profondeur de l'algo minimax
# dl_player:Player : CNN player à utiliser
# n_best_moves:int : la recherche minimax est effectuée sur les N meilleurs coups prédits par le réseau
class LookAheadDlPlayer(Player):
    def __init__(self, max_depth:int, dl_player:Player, n_best_moves:int=5):
        super().__init__()
        self.max_depth = max_depth
        self.original_player = None  # Pour savoir qui est l'IA
        self.name = "LookAheadDlPlayer"
        self.dl_player = dl_player
        self.n_best_moves = n_best_moves

    def play(self, board:Board):
        #Minimax algo
        best_move = None
        best_score = float('-inf')

        # On mémorise notre position pour l'évaluation terminale
        self.original_player = self.position

        #On utilise le réseau pour obtenir potentiellement la meilleure action possible
        available_actions, value = self._inference(board=board)

        #Cas particulier : quand aucune actrin n'est dispo
        if len(available_actions) == 0:
            return None

        for action in available_actions:
            #On joue le coup
            last_move = board.play_move(action=action)

            #On descend d'un niveaux
            score = self._minimax(board=board, depth=0, is_maximizing=False)

            #On annule le coup
            board.cancel_last_move(last_move=last_move)

            #Si le score obtenu est meilleur, on le conserve ainsi que le meilleur coup
            if score > best_score:
                best_score = score
                best_move = action

        return best_move
    
    def _inference(self, board:Board):
        #On récupère le state
        state = np.array(board.get_state())

        #On le transpose (10, 5, 5) => (5, 5, 10)
        state = np.transpose(state, (1, 2, 0))

        #On effectue la prédiction
        policy_logits, value = self.dl_player.predict(state)

        # value = float entre -1 (position perdue) et +1 (position gagnée)
        policy_logits = np.array(policy_logits).flatten()  # (1300,)

        #On récupère les mouvements possibles
        available_moves = board.get_available_moves()

        #Créer un masque des actions valides
        #On met -inf partout sauf pour les actions valides
        masked_logits = np.full(1300, -np.inf)

        #Pour chaque action valide, on conserve le logit correspondant
        action_to_move = {}  # flat_idx -> Action (pour retrouver l'action après)
        for action in available_moves:
            col, row = action.from_pos
            move_idx = action.move_idx
            #Calcul de l'index flat : col * (5 * 52) + row * 52 + move_idx
            flat_idx = col * (5 * 52) + row * 52 + move_idx
            masked_logits[flat_idx] = policy_logits[flat_idx]
            action_to_move[flat_idx] = action
        
        value = float(value.numpy()[0][0])

        probs = self.dl_player._softmax(masked_logits)

        #On récupères les N meilleures actuibs
        top_k = min(self.n_best_moves, len(action_to_move))
        top_flat_idxs = np.argsort(probs)[::-1][:top_k]
        top_flat_idxs = [idx for idx in top_flat_idxs if idx in action_to_move]

        best_actions = []
        for i in top_flat_idxs:
            best_actions.append(action_to_move[i])

        return best_actions, value

    def _get_value(self, board:Board) -> float:
        state = np.array(board.get_state())
        state = np.transpose(state, (1, 2, 0))
        _, value = self.dl_player.predict(state)
        return float(value.numpy()[0][0])

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
            value = self._get_value(board)
            return value if is_maximizing else value * -1

        #On utilise le réseau pour obtenir potentiellement la meilleure action possible
        available_moves, value = self._inference(board=board)

    
        # Si aucun coup n'est disponible, on joue le coup par défaut (échange de carte)
        if len(available_moves) == 0:
            # Protection contre les boucles infinies de default moves
            if consecutive_default_moves >= 4:
                # Les deux joueurs sont bloqués, retourner une évaluation neutre
                if is_maximizing:
                    return value
                else:
                    return value * -1
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
