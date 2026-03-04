from players import Player
from dl_players_v2 import CNNPlayer_v2
import tensorflow as tf
from trainer import PPOBuffer
from game import Game
import numpy as np
from tqdm import tqdm

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
            entropy_coef:float = 0.01, #Poids du bonus d'entropie (exploration)
            gamma: float = 0.99,
            lam:float = 0.85
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

        #Initialisation de l'optimizeur
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        #Stats par itération
        self.wins = 0
        self.losses = 0
        self.draws = 0

    #Collecte: joue n_games parties et remplit le buffer
    def _collect(self)->dict:
        #Deux instances distinctes qui partagent le même modèle
        p1 = self.player1
        p2 = self.player2
        p2.model = p1.model #Même poids

        buffer = PPOBuffer(p1=p1, p2=p2, gamma=self.gamma, lam=self.lam)
        self.wins = self.losses = self.draws = 0

        for _ in tqdm(range(self.n_games), desc="self-play"):
            game = Game(player_one=p1, player_two=p2, verbose=False, trainer=buffer)
            result = game.playGame(return_winner=True)
            if result == 1: self.wins += 1
            elif result == 2: self.losses += 1
            else: self.draws += 1

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
                advantages = tf.constant(data['advantages'][batch_idx]) #Avantages estimés via GAE. Indique si l'action prise était meilleure ou pire qu'attendu
                returns = tf.constant(data['returns'][batch_idx]) #Retours cibles pour la value fonction (somme des récompenses actualisées)

                #Mécanisme d'auto-différentiation de tensorflow
                #Tensorflow enregistre sur la "bande magnétique" toutes les opérations effectuées sur les tenseurs à l'intérieur du bloc
                with tf.GradientTape() as tape:
                    #Forward pass du réseau de neurones sur le minibatch (retourne les deux têtes du réseau)
                    #policy_logits : [batch, n_actions] scores bruts pour chaque action avant softmax
                    #values : [batch, 1] estimation de la valeur de l'état
                    policy_logits, values = self.player1.model(states, training=True)

                    #log-probs de toutes les actions
                    log_probs_all = tf.nn.log_softmax(policy_logits, axis=-1) #(batch, 1300)

                    #Extraire le log-prob de l'action effectivement jouée
                    batch_size = tf.shape(states)[0]
                    #Correspondance entre la ligne et l'action correspondante
                    gather_idx = tf.stack([tf.range(batch_size), actions], axis=-1)
                    new_log_probs = tf.gather_nd(log_probs_all, gather_idx)

                    #Coeur mathématique de PPO

                    #Ratio  π_new / π_old
                    #CAD calcule le ratio de probabilité entre la nouvelle et l'ancienne politique
                    #On utilise exp pour la stabilité numérique. (si nouvelle politique est la mm que l'ancienne, ratio=1)
                    ratio = tf.exp(new_log_probs - old_lp)

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

                    #Entropie (encourage l'exploration)
                    probs = tf.nn.softmax(policy_logits, axis=-1)
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs_all, axis=-1))

                    #Losse combinée avec les deux hyperparamètres d'équilibre 
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    #On soustrait l'entropie car on minimise la loss — soustraire l'entropie revient à la maximiser.

                grads = tape.gradient(total_loss, self.player1.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.player1.model.trainable_variables))

                policy_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                entropies.append(float(entropy))

        return {
            'policy_loss' : np.mean(policy_losses),
            'value_loss' : np.mean(value_losses),
            'entropy' : np.mean(entropies)
        }
    
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
            data = self._collect()
            metrics = self._update(data)

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

            if save_path and (i + 1) % save_every == 0:
                self.player1.save_weights(f"{save_path}_iter{i+1}.weights.h5")
                print(f"  → Sauvegardé : {save_path}_iter{i+1}.weights.h5")

            if on_iteration_end is not None:
                on_iteration_end(i + 1, metrics, history)

        return history


if __name__ == "__main__":
    p1 = CNNPlayer_v2()
    p1.load_weights('../saved-models/CNNPlayer-v1-withdropout-datalarge-dropout-weights.weights.h5')
    p2 = CNNPlayer_v2()

    trainer = PPOTrainer(
        player1=p1,
        player2=p2,
        n_games=64,          # parties par itération
        n_epochs=4,          # passes sur les données collectées
        minibatch_size=64,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95
    )

    #trainer.train(n_iterations=3, save_every=10, save_path=None)

    trainer.train(
        n_iterations=10,
        save_every=10,
        save_path='../saved-models/ppo-v2'
    )


