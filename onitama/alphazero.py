from players import Player
import numpy as np
from collections import deque
import tensorflow as tf
from tqdm import tqdm
from game import Game
from board import Board
import copy

class AlphaZeroTrainer:
    def __init__(
        self,
        player_p1: Player,
        player_p2: Player,
        n_games: int = 100,          # parties de self-play par itération
        n_simulations: int = 100,    # simulations MCTS par coup
        c_puct: float = 1.5,         # exploration MCTS
        dirichlet_alpha: float = 0.3,   #???
        dirichlet_eps: float = 0.25, #???
        temperature_moves: int = 15, # coups avec τ=1 en début de partie
        n_epochs: int = 5,      #nombre de passes d'entraînement dans le réseau
        minibatch_size: int = 128,  #nombre de transitions avec lesquelles on fait l'entrainement (sample du Buffer)
        learning_rate: float = 1e-3,
        replay_buffer_size: int = 50_000,  # garder les N dernières transitions
        policy_coef: float = 1.0,   #???
        value_coef: float = 1.0,    #???
    ):
    
        self.player_p1 = player_p1
        self.player_p2 = player_p2
        self.player_p2.model = self.player_p1.model #Poids partagés entre les deux joueurs

        self.n_games = n_games
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature_moves = temperature_moves
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.policy_coeff = policy_coef
        self.value_coeff = value_coef

    #Méthode principale
    def train(self, n_iterations:int):
        #Initialisation
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.player_p1.compile_for_rl()
        self.buffer = AlphaZeroBuffer(capacity=self.replay_buffer_size)

        #Initialisation de l'historique des métriques
        history = {
            'policy_loss' : [],
            'value_loss' : [],
            'wins' : [],
            'losses' : [],
            'draws' : [],
            'buffer_size' : []
        }

        #Boucle principale
        for i in range(n_iterations):
            # Self-play avec MCTS (pour remplir le buffer)
            win, loss, draw = self._collect()

            # Si buffer avec suffismt de data on peut entrainer
            if len(self.buffer) >= self.minibatch_size:
                metrics = self._update()
            else:
                metrics = {'policy_loss' : None, 'value_loss' : None}

            #Historique
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])
            history['wins'].append(win)
            history['losses'].append(loss)
            history['draws'].append(draw)
            history['buffer_size'].append(len(self.buffer))

            #Logs
            print(f"[{i+1:4d}/{n_iterations}] "
              f"buffer={len(self.buffer)} | "
              f"W/L/D={win}/{loss}/{draw} | "
              f"p_loss={metrics['policy_loss']} | "
              f"v_loss={metrics['value_loss']}")
            
            #Todo : sauvegarde peruidique

        return history
    
   

    #Self play avec MCTS, retourne une liste de (state, distribution de la politique MCTS, résolutat final de la partie)
    def _collect(self) -> list:
        #métrique de suivi
        wins = losses = draws = 0

        for _ in tqdm(range(self.n_games), desc="Self-play"):
            #Initialisation du jeu via Game (on l'utilise quand mm pour profiter des fonctions d'initialisation)
            game = Game(
                player_one=self.player_p1,
                player_two=self.player_p2,
                verbose=False
            )

            step = 0

            #Boucle pour jouer la partie
            while True:
                step += 1

                # En cas de boucle infinie (joueurs tournant en boucle sans arriver à une fin de jeu)
                if step > 200:
                    self.buffer.close(winner_player_i=None)
                    draws += 1
                    break

                #On récupère l'état
                state = np.transpose(np.array(game.board.get_state()), (1, 2, 0))

                # qui est le joueur actif
                current_player_i = 1 if game.current_player == game.player_one else 2

                #MCTS -> π_mcts (1300,)
                pi_mcts = self._mcts_search(game.board)

                #Sauvegarde dans le buffer temporaire
                self.buffer.save_experience(state=state, pi_mcts=pi_mcts, player_i=current_player_i)

                #Choisir l'action en prent en cpt la T°
                action = self._select_action(pi_mcts, game.board, step)

                #Jouer le coup
                if action is not None:
                    game.board.play_move(action=action)
                else:
                    game.board.play_default_move()

                #verifier fin de partie
                game_ended, winner = game.board.game_has_ended()
                if game_ended:
                    winner_player_i = 1 if winner == game.player_one.position else 2
                    self.buffer.close(winner_player_i=winner_player_i)
                    if winner_player_i == 1:
                        wins += 1
                    else:
                        losses += 1
                    break
                    
                #Intervertir les joueurs pour le coup suivant
                game.current_player, game.next_player = game.next_player, game.current_player
        return wins, losses, draws
    
    #Sélectionne l'action finale
    def _select_action(self, pi_mcts:list, board:Board, step:int):
        moves = board.get_available_moves()

        #Aucune aciton possible
        if not moves:
            return None
        
        if step <= self.temperature_moves:
            # t=1 : on échantillonne depuis π_mcts directmt
            valid_indices = [m.from_pos[0]*260 + m.from_pos[1]*52 + m.move_idx for m in moves]  #???
            probs = np.array([pi_mcts[i] for i in valid_indices])   #???
            probs /= probs.sum() #???
            chosen_idx = np.random.choice(len(moves), p=probs)
        else:
            # t=0 : greedy sur les coups valides
            valid_indices = [m.from_pos[0]*260 + m.from_pos[1]*52 + m.move_idx for m in moves]
            chosen_idx = np.argmax([pi_mcts[i] for i in valid_indices])
        
        return moves[chosen_idx]


    #Gradient sur cross-entropy + MSE(z, value)
    def _update(self) -> dict:
        policy_losses, value_losses = [], []

        #N passes dans le réseau
        for _ in range(self.n_epochs):
            #on échantillonne un mini batch depui le buffer
            states, pi, z = self.buffer.sample(self.minibatch_size)

            states = tf.constant(states)    #Etats (n, 5, 5, 10)
            pi = tf.constant(pi)    # Distribution MCTS (n, 1300)
            z = tf.constant(z) #Résultat +/-1 (n,)

            with tf.GradientTape() as tape: #???
                policy_logits, values = self.player_p1.model(states, training=True) #Ici on veut que la normalisation et le drpout soient actifs
                #policy_logits : (n, 1300)
                #values : (n, 1)

                loss_policy = -tf.reduce_mean(  #???
                    tf.reduce_sum(pi * tf.nn.log_softmax(policy_logits, axis=-1), axis=-1)
                )

                loss_value = tf.reduce_mean(    #???
                    tf.square(tf.squeeze(values, axis=-1) - z)
                )

                total_loss = self.policy_coeff * loss_policy + self.value_coeff * loss_value

            grads = tape.gradient(total_loss, self.player_p1.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)   #???
            self.optimizer.apply_gradients( #???
                zip(grads, self.player_p1.model.trainable_variables)
            )

            policy_losses.append(float(loss_policy))
            value_losses.append(float(loss_value))
        
        return {
            'policy_loss' : np.mean(policy_losses),
            'value_loss' : np.mean(value_losses)
        }

    #On lance n simulations, retourne les probabilités des 1300 actions
    def _mcts_search(self, board:Board) -> np.ndarray:
        #Stocle les noeuds
        tree = {}

        """
        {
        'N': {flat_idx: 0, ...},      # visites par action
        'W': {flat_idx: 0.0, ...},    # valeur cumulée par action
        'Q': {flat_idx: 0.0, ...},    # valeur moyenne = W/N
        'P': {flat_idx: prior, ...},  # priors du réseau (avec Dirichlet à la racine)
        'moves': {flat_idx: action},  # pour rejouer le coup pendant la simulation
        }
        """

        #Evalue le noeud avec le réseau, retoure (node, value)
        def expand(b:Board):
            valid_moves = b.get_available_moves()

            #evaluation
            state = np.transpose(np.array(b.get_state()), (1, 2, 0))
            policy_logits, value = self.player_p1.predict(state)
            policy_logits = np.array(policy_logits).flatten()

            #masque + softmax sur coups valides ???
            mask = np.full(1300, -1e9, dtype=np.float32)    #???
            moves_dict = {}
            for m in valid_moves:
                flat_idx = m.from_pos[0] * 260 + m.from_pos[1] * 52 + m.move_idx
                mask[flat_idx] = 0.0
                moves_dict[flat_idx] = m

            masked_logits = policy_logits + mask    #???
            max_l = np.max(masked_logits[mask == 0.0])
            exp_l = np.where(mask == 0.0, np.exp(masked_logits - max_l), 0.0) #???
            priors = exp_l / exp_l.sum()

            node = {
                'N': {idx: 0 for idx in moves_dict},
                'W': {idx: 0.0 for idx in moves_dict},
                'Q': {idx: 0.0 for idx in moves_dict},
                'P': {idx: float(priors[idx]) for idx in moves_dict},
                'moves': moves_dict,
            }
            return node, float(tf.squeeze(value))
        
        #Créer et initialiser le noeud racine
        root_hash = repr(board)
        root_node, _ = expand(board)

        #Bruit de dirichlet à la racine ???
        indices = list(root_node['P'].keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(indices))
        for i, idx in enumerate(indices):
            root_node['P'][idx] = (
                (1 - self.dirichlet_eps) * root_node['P'][idx]
                + self.dirichlet_eps * noise[i]
            )

        tree[root_hash] = root_node

        #N simulation ???
        for _ in range(self.n_simulations):
            b = copy.deepcopy(board)
            path = []   #liste de (hash, flat_idx) pour le backup

            #Selection
            h = root_hash
            while h in tree:
                node = tree[h]
                if not node['N']:   #Etat terminal (aucun coup possible)
                    break

                N_total = sum(node['N'].values())
                best_score, best_idx = -float('inf'), None

                for flat_idx in node['N']:
                    q = node['Q'][flat_idx]
                    p = node['P'][flat_idx]
                    n = node['N'][flat_idx]
                    score = q + self.c_puct * p * (N_total ** 0.5) / (1 + n)
                    if score > best_score:
                        best_score, best_idx = score, flat_idx
                
                path.append((h, best_idx))
                b.play_move(action=node['moves'][best_idx])

                #Vérifier fin de partie
                game_ended, _ = b.game_has_ended()
                if game_ended:
                    break

                h = repr(b)
            
            #Expansion + evaluation
            game_ended, winner = b.game_has_ended()
            if game_ended:
                #Noeud terminale : valeur certaine (pas besoin du réseau)
                #victoire de l'adversaire, donc -1 pour le joueur qui doit jouer
                value = -1.0
            else:
                h = repr(b)
                if h not in tree:
                    node, value = expand(b)
                    tree[h] = node
                else:
                    #Noeud déjà connu, on réutilise Q moyen comme valeur
                    node = tree[h]
                    vals = [node['Q'][i] for i in node['Q'] if node['N'][i] > 0]
                    value = np.mean(vals) if vals else 0.0
            
            #Backup
            #On remonte en alternant, la valeur est du point de vue du joueur qui a joué
            for h, flat_idx in reversed(path):
                value = -value
                tree[h]['N'][flat_idx] += 1
                tree[h]['W'][flat_idx] += value
                tree[h]['Q'][flat_idx] = (
                    tree[h]['W'][flat_idx] / tree[h]['N'][flat_idx]
                )

            
        #Construire π_mcts depuis les visites de la racine
        pi = np.zeros(1300, dtype=np.float32)
        root = tree[root_hash]
        N_total = sum(root['N'].values())
        if N_total > 0:
            for flat_idx, n in root['N'].items():
                pi[flat_idx] = n / N_total

        return pi
        
    


class AlphaZeroBuffer():
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._pending = [] #Transitions de la partie en cours

    # appelé à chaque coup pendant la partie
    def save_experience(self, state:list, pi_mcts, player_i:int):
        self._pending.append((state, pi_mcts, player_i))

    #Appelé en fin de partie - assigne z et transfert vers le buffer
    def close(self, winner_player_i:int | None):
        for i, (state, pi, player_i) in enumerate(self._pending):
            #z vu du joueur qui a joué (p1 = pair, p2 = impair)
            if winner_player_i is None:
                z = 0.0
            else:
                z = 1.0 if player_i == winner_player_i else -1.0
            self.buffer.append((state, pi, z))
        self._pending.clear()

    #Echantillonne un mini batch aléatoire
    def sample(self, batch_size:int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([x[0] for x in batch], dtype=np.float32)
        pi = np.array([x[1] for x in batch], dtype=np.float32)
        z = np.array([x[2] for x in batch], dtype=np.float32)
        return states, pi, z
    
    def __len__(self):
        return len(self.buffer)