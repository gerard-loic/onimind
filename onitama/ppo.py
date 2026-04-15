from players import Player
import tensorflow as tf
from game import Game
import numpy as np
from tqdm import tqdm
from trainer import DataTrainer


#Classe de gestion du buffer pour l'entraînement du réseau en self-play avec PPO
class PPOBuffer(DataTrainer):
    # gamma : facteur d'actualisation (proche de 1 = récompenses futures comptent beaucoup)
    # lam   : facteur lambda pour GAE (0 = TD pur, haute variance ; 1 = Monte Carlo, haut biais)
    def __init__(self, p1:Player, p2:Player, gamma:float=0.99, lam:float=0.95):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.gamma = gamma
        self.lam = lam
        self._clear()

    def _clear(self):
        #(5,5,10) état vu du joueur courant (ou input du réseau)
        #int : index flat de l'action (savoir quelle action évaluer)
        # P(action|state) au moment de la collecte (utilisé pour calculer le ration PPO)
         #V(s) estimé par le réseau (utilisé pour calculer GAE)
        self.p1_states, self.p1_actions, self.p1_log_probs, self.p1_values, self.p1_masks = [], [], [], [], []
        self.p2_states, self.p2_actions, self.p2_log_probs, self.p2_values, self.p2_masks = [], [], [], [], []

        self._traj_start_p1 = 0 #Pointeur du début de la trajectoire en cours (partie)
        self._traj_start_p2 = 0 #Pointeur du début de la trajectoire en cours (partie)

        #Listes fusionnées (p1 + p2, remplies dans close)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.masks = []
        self.advantages = [] # float - calculé dans close() (utilisé pour loss policy PPO)
        self.returns = [] #float - calculé dans close() (cible pour la value head)

    # Enregistre une expérience, c'est à dire un couple état / action (met en cache)
    # player:Player : joueur
    # state:list[5:5:10] : Etat (Matrice de 5x5x10)
    # action:list(rel_x:int, rel_y:int, move_idx:int) : Action
    # probability:float : Probabilité de l'action sous la politique courante
    # value:float : Estimation de V(s)
    def save_experience(self, player:Player, state:list, action:list, log_prob:float, value:float, valid_mask=None):
        #Joueur non-réseau (heuristique) : pas de log_prob ni value, on n'enregistre pas
        if log_prob is None:
            return

        #Transposition
        state_t = np.transpose(np.array(state), (1, 2, 0))

        col, row = action.from_pos
        flat_idx = col * 260 + row * 52 + action.move_idx

        if player == self.p1:
            self.p1_states.append(state_t)
            self.p1_actions.append(flat_idx)
            self.p1_log_probs.append(log_prob)
            self.p1_values.append(value)
            self.p1_masks.append(valid_mask)
        else:
            self.p2_states.append(state_t)
            self.p2_actions.append(flat_idx)
            self.p2_log_probs.append(log_prob)
            self.p2_values.append(value)
            self.p2_masks.append(valid_mask)

    # Termine la trajectoire et calcule GAE
    def close(self, winner):
        #Score en fonction du gagnant
        if winner == self.p1:
            p1_reward, p2_reward = 1.0, -1.0
        elif winner == self.p2:
            p1_reward, p2_reward = -1.0, 1.0
        else:
            # Joueur alternatif (heuristique) a gagné : p1 a perdu, p2 (CNN) n'a pas d'expériences
            p1_reward, p2_reward = -1.0, 0.0

        #Calcul GAE pour p1 et p2
        adv_p1, ret_p1 = self._compute_gae(self.p1_values, self._traj_start_p1, p1_reward)
        adv_p2, ret_p2 = self._compute_gae(self.p2_values, self._traj_start_p2, p2_reward)

        #Fusion dans les listes finales
        self.states.extend(self.p1_states[self._traj_start_p1:])
        self.states.extend(self.p2_states[self._traj_start_p2:])
        self.actions.extend(self.p1_actions[self._traj_start_p1:])
        self.actions.extend(self.p2_actions[self._traj_start_p2:])
        self.log_probs.extend(self.p1_log_probs[self._traj_start_p1:])
        self.log_probs.extend(self.p2_log_probs[self._traj_start_p2:])
        self.masks.extend(self.p1_masks[self._traj_start_p1:])
        self.masks.extend(self.p2_masks[self._traj_start_p2:])
        self.advantages.extend(adv_p1.tolist())
        self.advantages.extend(adv_p2.tolist())
        self.returns.extend(ret_p1.tolist())
        self.returns.extend(ret_p2.tolist())

        #Avancer les pointeurs
        self._traj_start_p1 = len(self.p1_states)
        self._traj_start_p2 = len(self.p2_states)

    def _compute_gae(self, values_list, traj_start, winner_reward):
        #On récupère les valeurs de la trajectoire, càd la partie en cours
        values = np.array(values_list[traj_start:], dtype=np.float32)

        #Joueur heuristique : aucune expérience enregistrée, on retourne des tableaux vides
        if len(values) == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty

        #0 à chaque pas, winner_reward uniquement au dernier
        rewards = np.zeros(len(values), dtype=np.float32)
        rewards[-1] = winner_reward

        #Calcul de delta_T : V(s_{T+1}) = last_value (on ajoute une valeur pour que le calcul à T+1 soit toujours juste)
        values_ext = np.append(values, 0.0)

        #GAE
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   ← erreur TD
        # A_t     = delta_t + (gamma * lam) * A_{t+1}   ← avantage lissé
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        #Etape 1 : on remonte le temps depuis le dernier coup de la partie vers le premier
        for t in reversed(range(len(rewards))):
            #On calcule la différence entre ce que le réseau estimait et ce qu'on obtenu.
            #On a donc un delta positif quand le réseau était trop optimiste, négatif quand il était pessimiste
            #Gamma pondère combien on fait confiance à la valeur du prochain état pour estimer si le coup était bon
            # C'est à dire ici le delta c'est la récompense immédiate + la veleur future pondérée par gamma - ce que le réseau pensait comme valeur pour cet état
            # Concrètement, plus une victoire est lointaine (donc le gain lointain, plus la pondération est faible)
            delta = rewards[t] + self.gamma * values_ext[t+1] - values_ext[t]

            #On calcule l'avantage cumulé, recalculé pour chaque t
            #Donc à t=0 : somme pondérée de toutes les erreurs TD (temporal difference) futures avec un poids qui décroit exponentiellement
            #Temporal diffrence : idée fondamentale : estimer la valeur d'un état en se basant sur l'état suivant sans attendre la fin de la partir
            #C'est le rôle de lam (TD pur lam=0, MonteCarlo lam=1, GAE  lam=0.96 c'est une interpolation entre TD et MOnteCarlo)
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae

        return advantages, advantages + values  # (advantages, returns)

    # Retourne les données sous forme de tensors et vide le buffer
    def get(self) -> dict:
        data = {
            'states' : np.array(self.states, dtype=np.float32), #(N, 5, 5, 10)
            'actions' : np.array(self.actions, dtype=np.int32),
            'log_probs' : np.array(self.log_probs, dtype=np.float32),
            'masks' : np.array(self.masks, dtype=np.float32), #(N, 1300) : 0 valide, -1e9 invalide
            'advantages' : np.array(self.advantages, dtype=np.float32),
            'returns' : np.array(self.returns, np.float32)
        }

        #Normalisation des avantages, permet de stabiliser les gradients PPO
        adv = data['advantages']
        data['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        self._clear()
        return data

    def __len__(self):
        return len(self.states)

class PPOTrainer:
    def __init__(
            self,
            player1:Player,  #Le réseau à entraîner
            player2:Player,  #Le réseau opposé
            n_games:int = 64,   #Nombre de parties en self-play, par itération
            n_epochs:int = 4,   #Nombre de passes sur les données collectées
            minibatch_size:int = 64, #Taille des mini batch pour la mise à jour
            learning_rate:float = 3e-4, #Taux d'apprentissage Adam
            clip_epsilon:float = 0.2, #Borne du cli^pping PPO
            value_coef:float = 0.5, #Poids de la value loss dans la loss totale
            entropy_coef:float = 0.03, #Poids du bonus d'entropie (exploration)
            gamma: float = 0.99, #Facteur d'actualisation (pondère l'importance des récompenses futures par rapport aux récompenses immédiates)
            lam:float = 0.85, #Controle du compromis biais/variance dan l'estimation de l'avantage GAE'
            alternative_players:list = [],   #Joueurs alternatifs utilisés dans le self play
            alternative_players_ratio:list = [], #Ratio d'utilisation des joueurs alternatifs
            past_self:Player = None,          #Ancienne version de player1 (version gelée, mise à jour toutes les N itérations)
            past_self_ratio:int = 0,          #% de parties jouées contre past_self (0 = désactivé)
            past_self_update_every:int = 10   #Mise à jour de past_self toutes les N itérations
    ):
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.alternative_players = alternative_players
        self.alternative_players_ratio = alternative_players_ratio
        self.past_self = past_self
        self.past_self_ratio = past_self_ratio
        self.past_self_update_every = past_self_update_every

        #Initialisation de l'optimizeur
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        #Stats par itération
        self.wins = 0
        self.losses = 0
        self.draws = 0

    #Boucle principale
    def train(self,
              n_iterations:int,
              save_every:int=50,
              save_path:str=None,
              on_iteration_end=None):

        self.player1.setPPOTraining(True)
        self.player2.setPPOTraining(True)
        self.player1.compile_for_rl()
        self.player2.compile_for_rl()

        # Initialisation de past_self avec les poids courants de player1 (version antérieure du réseau)
        if self.past_self is not None and self.past_self_ratio > 0:
            self.past_self.setPPOTraining(True)
            self.past_self.model.set_weights(self.player1.model.get_weights())
            print(f"past_self initialisé (mis à jour toutes les {self.past_self_update_every} itérations)")

        history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'wins': [],
            'losses': [],
            'draws': [],
            'transitions': []
        }

        for i in range(n_iterations):
            #On joue les parties
            data = self._collect()

            #On fait l'apprentissage
            metrics = self._update(data)

            #On ajoute aux métriques pour suivre
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])
            history['entropy'].append(metrics['entropy'])
            history['wins'].append(self.wins)
            history['losses'].append(self.losses)
            history['draws'].append(self.draws)
            history['transitions'].append(len(data['states']))

            print(f"[{i+1:4d}/{n_iterations}] "
                  f"transitions={len(data['states'])} | "
                  f"W/L/D={self.wins}/{self.losses}/{self.draws} | "
                  f"p_loss={metrics['policy_loss']:.4f} | "
                  f"v_loss={metrics['value_loss']:.4f} | "
                  f"entropy={metrics['entropy']:.4f}")

            # Mise à jour périodique de past_self
            if self.past_self is not None and self.past_self_ratio > 0 and (i + 1) % self.past_self_update_every == 0:
                self.past_self.model.set_weights(self.player1.model.get_weights())
                print(f"past_self mis à jour (iter {i+1})")

            #Sauvegarde régulière
            if save_path and (i + 1) % save_every == 0:
                self.player1.save_weights(f"{save_path}_iter{i+1}.weights.h5")
                print(f"Sauvegardé : {save_path}_iter{i+1}.weights.h5")

            if on_iteration_end is not None:
                on_iteration_end(i + 1, metrics, history)

        return history

    #Collecte: joue n_games parties et remplit le buffer
    def _collect(self)->dict:
        p1 = self.player1
        p2 = self.player2
        p2.model = p1.model  #Self-play live : p2 partage les poids courants de p1

        #Buffer utilisé pour stocker les trajectoires
        buffer = PPOBuffer(p1=p1, p2=p2, gamma=self.gamma, lam=self.lam)
        self.wins = self.losses = self.draws = 0

        # Construction de la liste d'adversaires et des seuils
        # Ordre : past_self en premier, puis alternative_players, puis self-play live (p2)
        seuils = []
        players = []
        cumul = 0

        if self.past_self is not None and self.past_self_ratio > 0:
            cumul += self.past_self_ratio * self.n_games // 100
            seuils.append(cumul)
            players.append(self.past_self)

        for r in self.alternative_players_ratio:
            cumul += r * self.n_games // 100
            seuils.append(cumul)
        for p in self.alternative_players:
            players.append(p)

        seuils.append(self.n_games)
        players.append(p2)

        # Jour n_games 
        for i in tqdm(range(self.n_games), desc="self-play"):
            for j, seuil in enumerate(seuils):
                if i < seuil:
                    player = players[j]
                    game = Game(player_one=p1, player_two=player, verbose=False, trainer=buffer)
                    result = game.playGame(return_winner=True)
                    if result == 1: self.wins += 1
                    elif result == 2: self.losses += 1
                    else: self.draws += 1
                    break

        return buffer.get()
    
    #Mise à jour PPO sur les données collectées pendant le rollout
    def _update(self, data:dict)->dict:
        n = len(data['states']) #Nombre de transitions collectées (taille du buffer)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.n_epochs):  #On réutilise les même données plusieurs fois
            indices = np.random.permutation(n)  #Génère un ordre aléatoire des indicies, pour mélanger les transitions et éviter les corrélations entre les mini batchs

            #Itère syur des tranches de taille minibatch_size
            for start in range(0, n, self.minibatch_size):
                #Sélectionne les index du mini batch courant
                batch_idx = indices[start:start + self.minibatch_size]

                states = tf.constant(data['states'][batch_idx]) #Etats du mini batch
                actions = tf.cast(data['actions'][batch_idx], tf.int32) #Actions du mini batch
                old_lp = tf.constant(data['log_probs'][batch_idx]) #log-proba de l'ancienne politique (ref PPO pour calculer le ratio r_t = exp(new_lp - old_lp))
                masks = tf.constant(data['masks'][batch_idx]) #Masque des actions valides (0=valide, -1e9=invalide)
                advantages = tf.constant(data['advantages'][batch_idx]) #Avantages estimés via GAE. Indique si l'action prise était meilleure ou pire qu'attendu
                returns = tf.constant(data['returns'][batch_idx]) #Retours cibles pour la value fonction (somme des récompenses actualisées)

                #Mécanisme d'auto-différentiation de tensorflow
                #Tensorflow enregistre sur la "bande magnétique" toutes les opérations effectuées sur les tenseurs à l'intérieur du bloc
                with tf.GradientTape() as tape:
                    #Forward pass du réseau de neurones sur le minibatch (retourne les deux têtes du réseau)
                    #policy_logits : [batch, n_actions] scores bruts pour chaque action avant softmax
                    #values : [batch, 1] estimation de la valeur de l'état
                    policy_logits, values = self.player1.model(states, training=False)

                    #Appliquer le masque pour ne considérer que les actions valides (cohérent avec la collecte)
                    masked_logits = policy_logits + masks

                    #log-probs sur la distribution masquée uniquement
                    log_probs_all = tf.nn.log_softmax(masked_logits, axis=-1) #(batch, 1300)

                    #Extraire le log-prob de l'action effectivement jouée
                    batch_size = tf.shape(states)[0]
                    #Correspondance entre la ligne et l'action correspondante
                    gather_idx = tf.stack([tf.range(batch_size), actions], axis=-1)
                    new_log_probs = tf.gather_nd(log_probs_all, gather_idx)

                    #Coeur mathématique de PPO

                    #Ratio  π_new / π_old
                    #CAD calcule le ratio de probabilité entre la nouvelle et l'ancienne politique
                    #On clippe le log-ratio avant exp pour éviter les explosions numériques
                    #(exp(20) ≈ 5×10^8, exp(4) ≈ 55 : borne suffisante pour PPO)
                    log_ratio = tf.clip_by_value(new_log_probs - old_lp, -4.0, 4.0)
                    ratio = tf.exp(log_ratio)

                    #Policy loss PPO clippée
                    #Contrainte de ratio dans l'intervale [1 - ε, 1 + ε]
                    #C'est la limite de confiance PPO : on ne laisse pas la politique trop s'éloigner de l'ancienne politique en 1 seule update
                    clipped = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    #Loss PPO clipée (le minimum choisit la mise à jour la plus conservatrice)
                    #Si avantage > 0 (bonne action) ; on ne laisse pas le ratio monter au dela de 1+ ε
                    #Si avantage < 0 (mauvaise action) : on ne laisse pas le ratio descendre en dessous de 1-ε
                    #- devant reduce_mean car on maximise l'expérience, donc on minimise son négatif
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
        
                    values = tf.squeeze(values, axis=-1)
                    value_loss = tf.reduce_mean(tf.square(values - returns))

                    #Entropie sur la distribution masquée (exploration sur actions valides uniquement)
                    probs = tf.nn.softmax(masked_logits, axis=-1)
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs_all, axis=-1))

                    #Losse combinée avec les deux hyperparamètres d'équilibre 
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    #On soustrait l'entropie car on minimise la loss — soustraire l'entropie revient à la maximiser.

                grads = tape.gradient(total_loss, self.player1.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                self.optimizer.apply_gradients(zip(grads, self.player1.model.trainable_variables))

                policy_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                entropies.append(float(entropy))

        return {
            'policy_loss' : np.mean(policy_losses),
            'value_loss' : np.mean(value_losses),
            'entropy' : np.mean(entropies)
        }
    
    