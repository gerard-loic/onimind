from players import Player
from board import Board
import numpy as np


#Noeud (état) de l'arbre des possibilités lors de la recherche MCTS
class MCTSNode:
    def __init__(self, P:float, action=None):
        self.P = P  #Prior donné par le réseau
        self.N = 0  #Visites
        self.Q = 0.0    #Valeur moyenne
        self.W = 0.0    #Total cumulé des Q
        self.action = action    #L'action qui mène à ce noeud
        self.children = None

class AlphaZeroPlayer(Player):
    def __init__(self, dl_player:Player, num_simulations:int=100, c_puct:float=3):
        self.name = "AlphaZeroPlayer"
        self.player = dl_player
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        super().__init__()

    def play(self, board:Board):
        #Racine
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

        #On retourne l'action
        return best_child.action


    def _selection(self, board:Board, node:MCTSNode):
        #Fin de partie ?
        ended, winner = board.game_has_ended()

        if ended:
            #Fin du jeu, l'exploration de l'arbre s'arrête là et on retourne 1 ou -1 en fonction du joueur courant
            value = 1.0 if winner == board.current_player else -1.0
            node.N += 1 #Visite +1
            node.W += value #Total des Q
            node.Q = node.W / node.N    #MAJ valeur moyenne
            return value
        
        if node.children == None:
            #Pas d'enfant, donc pas encore visité -> expand -> predict -> créer les enfants
            value = self._expand_node(parent_node=node, board=board)    #Value : P du nouveau noeud
            
            #backpropagation
            node.N += 1                 #On a visité le noeud
            node.W += value             #On ajoute la valeur au total
            node.Q = node.W / node.N    #On met à jour Q

            #C'est un nouveau noeud donc on s'arrête là
            return value
        else:
            #Des enfants à visiter
            #Selection UCB -> jouer le coup -> descendre

            best_child = None
            best_ucb = -np.inf
            for hash, children in node.children.items():
                #Calcul de l'UCB pour chaque enfant
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
            
    
    #Pour un noeud pas encore visité
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
    

    def _print_tree(self, node: MCTSNode, depth: int = 0, max_depth: int = 40):
        indent = "  " * depth
        action_str = str(node.action) if node.action else "root"
        print(f"{indent}[{action_str}] N={node.N} Q={node.Q:.3f} P={node.P:.3f}")
        if node.children and depth < max_depth:
            for child in sorted(node.children.values(), key=lambda c: -c.N):
                self._print_tree(child, depth + 1, max_depth)
   


