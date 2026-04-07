from players import Player
import numpy as np
from collections import deque
import tensorflow as tf
from tqdm import tqdm
from game import Game, GameSession
from board import Board

# Évite que TF alloue toute la mémoire GPU/CPU d'un coup
for _dev in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_dev, True)

class AlphaZeroTrainer:
    def __init__(
        self,
        player_p1: Player,
        player_p2: Player,
        n_games: int = 100,          # parties de self-play par itération
        n_simulations: int = 100,    # simulations MCTS par coup
        c_puct: float = 1.5,         # exploration MCTS
        dirichlet_alpha: float = 0.3,   # concentration du bruit Dirichlet (0.3 pour jeux à ~20 coups valides)
        dirichlet_eps: float = 0.25,    # poids du bruit Dirichlet vs prior réseau à la racine
        temperature_moves: int = 15,    # nombre de coups avec τ=1 (exploration) en début de partie
        n_epochs: int = 5,      #nombre de passes d'entraînement dans le réseau
        minibatch_size: int = 128,  #nombre de transitions avec lesquelles on fait l'entrainement (sample du Buffer)
        learning_rate: float = 1e-3,
        replay_buffer_size: int = 50_000,  # garder les N dernières transitions
        policy_coef: float = 1.0,
        value_coef: float = 1.0,
        checkpoint_path: str = None,  # ex: '../saved-models/Tairanauchu_rl' → sauvegarde tous les 10 iters
        eval_player: Player = None,   # joueur de référence pour l'évaluation (ex: LookAheadHeuristicPlayer)
        eval_games: int = 100,        # nombre de parties pour l'évaluation
        eval_every: int = 10,         # évaluer toutes les N itérations
        update_opponent_every: int = 10,          # tenter une mise à jour de p2 toutes les N itérations
        update_opponent_threshold: float = 0.55,  # p1 doit gagner >55% du self-play pour remplacer p2
    ):

        self.player_p1 = player_p1
        self.player_p2 = player_p2
        # p2 garde son propre modèle (pas de partage de référence)
        # Les poids initiaux sont copiés depuis p1, puis p2 est mis à jour périodiquement

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
        self.checkpoint_path = checkpoint_path
        self.eval_player = eval_player
        self.eval_games = eval_games
        self.eval_every = eval_every
        self.update_opponent_every = update_opponent_every
        self.update_opponent_threshold = update_opponent_threshold

    #Méthode principale
    def train(self, n_iterations:int, freeze_trunk_iters:int=10):
        #Initialisation
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.player_p1.compile_for_rl()
        # Geler le tronc pour les premières itérations : protège le savoir supervisé
        # pendant que les têtes s'adaptent au signal RL
        if freeze_trunk_iters > 0:
            self.player_p1.freeze_trunk()
        # Synchroniser p2 avec les poids initiaux de p1 (copie, pas partage de référence)
        self.player_p2.model.set_weights(self.player_p1.model.get_weights())
        self.buffer = AlphaZeroBuffer(capacity=self.replay_buffer_size)
        # Compteurs pour la décision de mise à jour de p2
        wins_since_last_update = 0
        games_since_last_update = 0

        #Initialisation de l'historique des métriques
        history = {
            'policy_loss' : [],
            'value_loss' : [],
            'wins' : [],
            'losses' : [],
            'draws' : [],
            'buffer_size' : [],
            'eval_win_rate' : [],   # taux de victoire vs eval_player (None si pas d'évaluation)
        }

        #Boucle principale
        for i in range(n_iterations):
            # Dégeler le tronc après freeze_trunk_iters itérations
            if freeze_trunk_iters > 0 and i == freeze_trunk_iters:
                print(f"[iter {i+1}] Dégel du tronc")
                self.player_p1.unfreeze_trunk()

            # Self-play avec MCTS (pour remplir le buffer)
            win, loss, draw = self._collect()
            wins_since_last_update += win
            games_since_last_update += win + loss  # draw = timeouts discardés, exclus du win_rate

            # Entraîner dès que le buffer est suffisant.
            # Pendant la phase de gel du tronc, seules les têtes sont entraînées
            # (le tronc gelé est exclu de trainable_variables automatiquement).
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
              f"v_loss={metrics['value_loss']}", flush=True)

            # Tentative de mise à jour de p2 toutes les update_opponent_every itérations
            if (i + 1) % self.update_opponent_every == 0 and games_since_last_update > 0:
                win_rate = wins_since_last_update / games_since_last_update
                if win_rate >= self.update_opponent_threshold:
                    self.player_p2.model.set_weights(self.player_p1.model.get_weights())
                    print(f"  → p2 mis à jour (win_rate={win_rate:.0%} >= {self.update_opponent_threshold:.0%})", flush=True)
                else:
                    print(f"  → p2 conservé  (win_rate={win_rate:.0%} < {self.update_opponent_threshold:.0%})", flush=True)
                wins_since_last_update = 0
                games_since_last_update = 0

            # Évaluation et sauvegarde toutes les eval_every itérations
            if (i + 1) % self.eval_every == 0:
                eval_win_rate = self._evaluate()
                history['eval_win_rate'].append(eval_win_rate)

                if self.checkpoint_path:
                    path = f"{self.checkpoint_path}_iter{i+1}.weights.h5"
                    self.player_p1.save_weights(path)
                    print(f"  → checkpoint sauvegardé : {path}", flush=True)
            else:
                history['eval_win_rate'].append(None)

        return history

    # Évalue le réseau courant contre eval_player sur eval_games parties (moitié en tant que P1, moitié en P2)
    def _evaluate(self) -> float | None:
        if self.eval_player is None:
            return None

        half = self.eval_games // 2

        # Moitié des parties : réseau en P1
        session1 = GameSession(player_one=self.player_p1, player_two=self.eval_player,
                               number_of_games=half, verbose=False)
        session1.start()
        s1 = session1.getStats()

        # Moitié des parties : réseau en P2
        session2 = GameSession(player_one=self.eval_player, player_two=self.player_p1,
                               number_of_games=half, verbose=False)
        session2.start()
        s2 = session2.getStats()

        wins  = s1['p1_win'] + s2['p2_win']
        draws = s1['draw']   + s2['draw']
        win_rate = (wins + 0.5 * draws) / self.eval_games

        print(f"  → eval vs {self.eval_player.name} ({self.eval_games} parties) : "
              f"W={wins} D={draws} L={self.eval_games - wins - draws} | "
              f"win_rate={win_rate:.2%}", flush=True)
        return win_rate


    #Self play avec MCTS, retourne une liste de (state, distribution de la politique MCTS, résolutat final de la partie)
    def _collect(self) -> list:
        #métrique de suivi
        wins = losses = draws = 0
        game_lengths = []

        for _ in tqdm(range(self.n_games), desc="Self-play"):
            #Initialisation du jeu via Game (on l'utilise quand mm pour profiter des fonctions d'initialisation)
            game = Game(
                player_one=self.player_p1,
                player_two=self.player_p2,
                verbose=False
            )

            step = 0
            # Position absolue (PLAYER_ONE_POSITION ou PLAYER_TWO_POSITION) jouée par p1 dans cette partie
            # Peut changer d'une partie à l'autre selon la couleur de la carte neutre
            p1_board_position = self.player_p1.position

            #Boucle pour jouer la partie
            while True:
                step += 1

                # En cas de boucle infinie (joueurs tournant en boucle sans arriver à une fin de jeu)
                if step > 200:
                    self.buffer.discard_pending()  # signal trop bruité, on ignore la partie
                    draws += 1
                    game_lengths.append(step)
                    break

                #On récupère l'état
                state = np.transpose(np.array(game.board.get_state()), (1, 2, 0))

                # qui est le joueur actif
                current_player_i = 1 if game.current_player == game.player_one else 2

                #MCTS -> π_mcts (1300,)
                pi_mcts = self._mcts_search(game.board, p1_board_position)

                # Masque des coups valides : 0 pour valide, -1e9 pour invalide
                valid_moves = game.board.get_available_moves()
                mask = np.full(1300, -1e9, dtype=np.float32)
                for m in valid_moves:
                    mask[m.from_pos[0] * 260 + m.from_pos[1] * 52 + m.move_idx] = 0.0

                #Sauvegarde dans le buffer temporaire
                self.buffer.save_experience(state=state, pi_mcts=pi_mcts, mask=mask, player_i=current_player_i)

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
                    game_lengths.append(step)
                    if winner_player_i == 1:
                        wins += 1
                    else:
                        losses += 1
                    break

                #Intervertir les joueurs pour le coup suivant
                game.current_player, game.next_player = game.next_player, game.current_player
        avg_len = np.mean(game_lengths) if game_lengths else 0
        print(f"  → longueur parties : avg={avg_len:.1f} min={min(game_lengths)} max={max(game_lengths)}", flush=True)
        return wins, losses, draws
    
    #Sélectionne l'action finale
    def _select_action(self, pi_mcts:list, board:Board, step:int):
        moves = board.get_available_moves()

        #Aucune aciton possible
        if not moves:
            return None
        
        if step <= self.temperature_moves:
            # t=1 : on échantillonne depuis π_mcts directmt
            valid_indices = [m.from_pos[0]*260 + m.from_pos[1]*52 + m.move_idx for m in moves]
            probs = np.array([pi_mcts[i] for i in valid_indices])
            if probs.sum() == 0:
                probs = np.ones(len(moves))  # fallback uniforme si MCTS n'a pas visité la racine
            probs /= probs.sum()
            chosen_idx = np.random.choice(len(moves), p=probs)
        else:
            # t=0 : greedy sur les coups valides
            valid_indices = [m.from_pos[0]*260 + m.from_pos[1]*52 + m.move_idx for m in moves]
            chosen_idx = np.argmax([pi_mcts[i] for i in valid_indices])
        
        return moves[chosen_idx]


    @tf.function
    def _train_step(self, states, pi, masks, z):
        """Step d'entraînement compilé (graph mode) — évite la retraçage TF à chaque epoch."""
        with tf.GradientTape() as tape:
            policy_logits, values = self.player_p1.model(states, training=True)
            # Masquer les coups invalides avant le log_softmax
            masked_logits = policy_logits + masks
            loss_policy = -tf.reduce_mean(
                tf.reduce_sum(pi * tf.nn.log_softmax(masked_logits, axis=-1), axis=-1)
            )
            loss_value = tf.reduce_mean(
                tf.square(tf.squeeze(values, axis=-1) - z)
            )
            total_loss = self.policy_coeff * loss_policy + self.value_coeff * loss_value
        grads = tape.gradient(total_loss, self.player_p1.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(
            zip(grads, self.player_p1.model.trainable_variables)
        )
        return loss_policy, loss_value

    #Gradient sur cross-entropy + MSE(z, value)
    def _update(self) -> dict:
        policy_losses, value_losses = [], []

        #N passes dans le réseau
        for _ in range(self.n_epochs):
            states, pi, masks, z = self.buffer.sample(self.minibatch_size)
            lp, lv = self._train_step(
                tf.constant(states, dtype=tf.float32),
                tf.constant(pi,     dtype=tf.float32),
                tf.constant(masks,  dtype=tf.float32),
                tf.constant(z,      dtype=tf.float32),
            )
            policy_losses.append(float(lp))
            value_losses.append(float(lv))

        return {
            'policy_loss' : np.mean(policy_losses),
            'value_loss' : np.mean(value_losses)
        }

    #On lance n simulations, retourne les probabilités des 1300 actions
    # p1_board_position : position absolue (PLAYER_ONE_POSITION ou PLAYER_TWO_POSITION) de p1 dans cette partie
    def _mcts_search(self, board:Board, p1_board_position:int) -> np.ndarray:
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

        #Evalue le noeud avec le réseau, retourne (node, value)
        def expand(b:Board):
            valid_moves = b.get_available_moves()

            #evaluation — utilise le modèle du joueur courant
            state = np.transpose(np.array(b.get_state()), (1, 2, 0))
            if b.current_player == p1_board_position:
                policy_logits, value = self.player_p1.predict(state)
            else:
                policy_logits, value = self.player_p2.predict(state)
            value = float(tf.squeeze(value))

            # Cas "default move" : aucun coup valide mais partie non terminée
            # Nœud vide — la boucle de sélection s'arrêtera dessus (if not node['N'])
            if not valid_moves:
                return {'N': {}, 'W': {}, 'Q': {}, 'P': {}, 'moves': {}}, value

            policy_logits = np.array(policy_logits).flatten()

            # masque + softmax sur coups valides uniquement
            mask = np.full(1300, -1e9, dtype=np.float32)
            moves_dict = {}
            for m in valid_moves:
                flat_idx = m.from_pos[0] * 260 + m.from_pos[1] * 52 + m.move_idx
                mask[flat_idx] = 0.0
                moves_dict[flat_idx] = m

            valid_mask = mask == 0.0
            masked_logits = policy_logits + mask
            max_l = np.max(masked_logits[valid_mask])
            exp_l = np.where(valid_mask, np.exp(masked_logits - max_l), 0.0)
            priors = exp_l / exp_l.sum()

            node = {
                'N': {idx: 0 for idx in moves_dict},
                'W': {idx: 0.0 for idx in moves_dict},
                'Q': {idx: 0.0 for idx in moves_dict},
                'P': {idx: float(priors[idx]) for idx in moves_dict},
                'moves': moves_dict,
            }
            return node, value
        
        #Créer et initialiser le noeud racine
        root_hash = repr(board)
        root_node, _ = expand(board)

        #Bruit de dirichlet à la racine
        indices = list(root_node['P'].keys())
        if indices:  # garde : la racine peut n'avoir aucun coup valide
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(indices))
            for i, idx in enumerate(indices):
                root_node['P'][idx] = (
                    (1 - self.dirichlet_eps) * root_node['P'][idx]
                    + self.dirichlet_eps * noise[i]
                )

        tree[root_hash] = root_node

        #N simulation ???
        for _ in range(self.n_simulations):
            # Pas de deepcopy : on joue directement sur board et on annule après
            path = []       #liste de (hash, flat_idx) pour le backup
            undo_stack = [] #liste de last_move pour annuler les coups joués

            #Selection (profondeur max = 30 pour éviter les cycles d'état — une partie dure ~20-30 coups)
            h = root_hash
            while h in tree and len(path) < 30:
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
                last_move = board.play_move(action=node['moves'][best_idx])
                undo_stack.append(last_move)

                #Vérifier fin de partie
                game_ended, _ = board.game_has_ended()
                if game_ended:
                    break

                h = repr(board)

            #Expansion + evaluation
            game_ended, _ = board.game_has_ended()
            if game_ended:
                #Noeud terminale : valeur certaine (pas besoin du réseau)
                #victoire de l'adversaire, donc -1 pour le joueur qui doit jouer
                value = -1.0
            else:
                h = repr(board)
                if h not in tree:
                    node, value = expand(board)
                    tree[h] = node
                else:
                    #Noeud déjà connu : valeur du point de vue du joueur qui vient d'arriver
                    # Q est du POV du joueur qui joue depuis ce noeud → négation
                    node = tree[h]
                    vals = [node['Q'][i] for i in node['Q'] if node['N'][i] > 0]
                    value = -np.mean(vals) if vals else 0.0

            #Backup
            #On remonte en alternant, la valeur est du point de vue du joueur qui a joué
            for h, flat_idx in reversed(path):
                value = -value
                tree[h]['N'][flat_idx] += 1
                tree[h]['W'][flat_idx] += value
                tree[h]['Q'][flat_idx] = (
                    tree[h]['W'][flat_idx] / tree[h]['N'][flat_idx]
                )

            #Annuler tous les coups joués (restaure le board à son état initial)
            for last_move in reversed(undo_stack):
                board.cancel_last_move(last_move)

            
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
    def save_experience(self, state:list, pi_mcts, mask, player_i:int):
        # float16 pour pi_mcts : précision suffisante, mémoire divisée par 2
        self._pending.append((state, pi_mcts.astype(np.float16), mask, player_i))

    #Appelé en cas de timeout : on ignore la partie (signal trop bruité)
    def discard_pending(self):
        self._pending.clear()

    #Appelé en fin de partie - assigne z et transfert vers le buffer
    def close(self, winner_player_i:int | None, gamma:float = 0.99):
        n = len(self._pending)
        for t, (state, pi, mask, player_i) in enumerate(self._pending):
            #z vu du joueur qui a joué (p1 = pair, p2 = impair)
            if winner_player_i is None:
                z = 0.0
            else:
                # discount selon la distance à la fin : positions récentes ont un signal plus fort
                steps_to_end = n - 1 - t
                z = (1.0 if player_i == winner_player_i else -1.0) * (gamma ** steps_to_end)
            self.buffer.append((state, pi, mask, z))
        self._pending.clear()

    #Echantillonne un mini batch aléatoire
    def sample(self, batch_size:int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([x[0] for x in batch], dtype=np.float32)
        pi    = np.array([x[1] for x in batch], dtype=np.float32)  # reconverti en float32 pour TF
        masks = np.array([x[2] for x in batch], dtype=np.float32)
        z     = np.array([x[3] for x in batch], dtype=np.float32)
        return states, pi, masks, z
    
    def __len__(self):
        return len(self.buffer)