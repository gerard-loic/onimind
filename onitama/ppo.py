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
    
    