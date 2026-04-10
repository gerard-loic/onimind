from players import Player, LookAheadHeuristicPlayer
from board import Board
import numpy as np
from dl_players_v7 import DensePlayer_v7
from dl_players_v6 import CNNPlayer_v6
from dl_minimax import LookAheadDlPlayer


class MCTSNode:
    def __init__(self, P:float, action=None):
        self.P = P
        self.N = 0  #Visites
        self.Q = 0.0    #Valeur moyenne
        self.W = 0.0    #Total cumulé des Q
        self.action = action    #L'action qui mène à ce noeud
        self.children = None

class AlphaZeroPlayer(Player):
    def __init__(self, dl_player:Player, num_simulations:int=100, c_puct:float=3, diagnose_winning_branches:bool=False, winning_q_threshold:float=-0.95):
        self.name = "AlphaZeroPlayer"
        self.player = dl_player
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.diagnose_winning_branches = diagnose_winning_branches
        self.winning_q_threshold = winning_q_threshold  # child.Q <= threshold => branche gagnante prouvée
        super().__init__()

    def play(self, board:Board):
        tree = MCTSNode(P=0)

        #Premiere sim pour créée les enfants de la racine
        self._selection(board=board, node=tree)

        #Appliquer le bruit de dirichlet sur les priors de la racine
        children = list(tree.children.values())
        noise = np.random.dirichlet([0.3] * len(children))
        for child, eta in zip(children, noise):
            child.P = 0.75 * child.P + 0.25 * eta


        for i in range(self.num_simulations - 1):
            self._selection(board=board, node=tree)
            #print(f"\n--- Simulation {i+1} ---")
            if i == 998:
                pass
                #print(f"\n--- Simulation {i+1} ---")
                #self._print_tree(tree)

        #On choisit l'enfant le plus visité
        best_child = max(tree.children.values(), key=lambda c: c.N)

        if self.diagnose_winning_branches:
            pass
            #self._diagnose_winning_branches(tree, best_child)

        #On retourne l'action
        return best_child.action


    def _selection(self, board:Board, node:MCTSNode):
        #Fin de partie ?
        ended, winner = board.game_has_ended()

        if ended:
            value = 1.0 if winner == board.current_player else -1.0
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            return value
        
        if node.children == None:
            #Pas d'enfant -> expand -> predict -> créer les enfants
            value = self._expand_node(parent_node=node, board=board)    #Value : P du nouveau noeud
            
            #backpropagation
            node.N += 1                 #On a visité le noeud
            node.W += value             #On ajoute la valeur au total
            node.Q = node.W / node.N    #On met à jour Q

            return value
        else:
            #Des enfants à visiter
            #Selection UCB -> jouer le coup -> descendre

            best_child = None
            best_ucb = -np.inf
            for hash, children in node.children.items():
                #Calcul de l'UCB
                ucb = -children.Q + self.c_puct * children.P * np.sqrt(node.N) / (1 + children.N)
                if ucb > best_ucb:
                    best_child = children
                    best_ucb = ucb

            #On joue le coup
            m = board.play_move(action=best_child.action)

            #On descend dans l'arbre
            value = -self._selection(board=board, node=best_child) #On utilise - car on va considérer la situation du point de vue du joueur suivant

            #On annule le coup
            board.cancel_last_move(last_move=m)

            #backpropagation
            node.N += 1                 #On a visité le noeud
            node.W += value             #On ajoute la valeur au total
            node.Q = node.W / node.N    #On met à jour Q

            return value
            
    

    def _expand_node(self, parent_node:MCTSNode, board:Board):
        
        #Etat courant
        state = np.transpose(np.array(board.get_state()), (1, 2, 0))

        #Prédiction du réseau
        policy_logits, value = self.player.predict(state=state)
        policy_logits = policy_logits.numpy()[0]  # shape (1300,)

        #Coups possibles
        available_moves = board.get_available_moves()

        #Masque sur les actions valides uniquement
        mask = np.full(1300, -np.inf)

        for move in available_moves:
            col, row = move.from_pos
            flat_idx = col * 260 + row * 52 + move.move_idx
            mask[flat_idx] = 0.0

        # Appliquer le masque et softmax pour obtenir les probabilités
        masked_logits = policy_logits + mask

        # softmax stable
        masked_logits -= np.max(masked_logits[np.isfinite(masked_logits)])
        exp = np.exp(masked_logits)
        exp[~np.isfinite(masked_logits)] = 0.0
        priors = exp / exp.sum()  # (1300,) — 0 sur l

        parent_node.children = {}

        #On ajoute les enfants
        for move in available_moves:
            m = board.play_move(action=move)
            node_hash = repr(board)
            board.cancel_last_move(last_move=m)
            col, row = move.from_pos
            flat_idx = col * 260 + row * 52 + move.move_idx
            parent_node.children[node_hash] = MCTSNode(P=priors[flat_idx], action=move)

        #on renvoie la valeur du noeud estimée par le réseau
        return float(value.numpy()[0][0])
    
    def _diagnose_winning_branches(self, tree: MCTSNode, best_child: MCTSNode):
        """Détecte si des branches gagnantes prouvées existent mais ne sont pas choisies."""
        children = [c for c in tree.children.values() if c.N > 0]

        # Log de diagnostic : Q min et distribution pour calibrer le seuil
        if children:
            q_values = sorted(c.Q for c in children)
            best_by_n = max(children, key=lambda c: c.N)
            min_q_child = min(children, key=lambda c: c.Q)
            print(f"[DIAG] Q min={q_values[0]:.3f}(N={min_q_child.N}), max={q_values[-1]:.3f}, "
                  f"n<-0.8={sum(1 for q in q_values if q < -0.8)}, "
                  f"n<-0.95={sum(1 for q in q_values if q < -0.95)}, "
                  f"total enfants visités={len(children)} | "
                  f"choix: N={best_by_n.N} Q={best_by_n.Q:.3f} → {best_by_n.action}")

        # child.Q <= threshold signifie que la position est très mauvaise pour l'adversaire = victoire prouvée
        winning_branches = [c for c in children if c.Q <= self.winning_q_threshold]

        if not winning_branches:
            return

        best_is_winning = best_child.Q <= self.winning_q_threshold
        if best_is_winning:
            return

        print("\n[DIAG] Branches gagnantes prouvées ignorées !")
        print(f"  Choix (N={best_child.N} Q={best_child.Q:.3f} P={best_child.P:.3f}) → {best_child.action}")
        print(f"  Branches gagnantes (Q <= {self.winning_q_threshold}) :")
        for c in sorted(winning_branches, key=lambda c: c.Q):
            print(f"    N={c.N} Q={c.Q:.3f} P={c.P:.3f} → {c.action}")
        print(f"  Tous les enfants (triés par N) :")
        for c in sorted(children, key=lambda c: -c.N)[:10]:
            marker = " <-- GAGNANT" if c.Q <= self.winning_q_threshold else ""
            print(f"    N={c.N} Q={c.Q:.3f} P={c.P:.3f} → {c.action}{marker}")

    def _print_tree(self, node: MCTSNode, depth: int = 0, max_depth: int = 40):
        indent = "  " * depth
        action_str = str(node.action) if node.action else "root"
        print(f"{indent}[{action_str}] N={node.N} Q={node.Q:.3f} P={node.P:.3f}")
        if node.children and depth < max_depth:
            for child in sorted(node.children.values(), key=lambda c: -c.N):
                self._print_tree(child, depth + 1, max_depth)
   


