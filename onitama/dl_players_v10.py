from players import Player
from board import Board
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
import numpy as np


#Métrique top K accuracy (pour évaluation tête de politique) - si le bon coup est dans les K meilleurs coups prédits par le réseau
def top_k_accuracy(k):
    def metric(y_true, y_pred):
        return metrics.sparse_top_k_categorical_accuracy(
            tf.argmax(y_true, axis=-1),
            y_pred,
            k=k
        )
    metric.__name__ = f'top_{k}_accuracy'
    return metric

#Loss personnalisée pour la tête de politique. (Le masque force softmax à ne distribuer la probabilité qu'entre les coups légaux. )
def masked_categorical_crossentropy(label_smoothing=0.1):
    def loss(y_true, y_pred):
        one_hot = y_true[:, :1300]
        mask = y_true[:, 1300:]   # 0 pour coups valides, -1e9 pour invalides
        masked_logits = y_pred + mask
        return tf.keras.losses.categorical_crossentropy(one_hot, masked_logits, from_logits=True, label_smoothing=label_smoothing)
    loss.__name__ = 'masked_categorical_crossentropy'
    return loss

#Accuracy sur les coups valides uniquement
def masked_accuracy():
    def metric(y_true, y_pred):
        one_hot = y_true[:, :1300]
        mask = y_true[:, 1300:]
        masked_logits = y_pred + mask
        return metrics.categorical_accuracy(one_hot, tf.nn.softmax(masked_logits))
    metric.__name__ = 'policy_logits_accuracy'
    return metric

#Idem top_k_accuracy mai sur les courps valides uniquement
def masked_top_k_accuracy(k):
    """Top-k accuracy sur coups valides uniquement : y_true = concat([one_hot, valid_mask])"""
    def metric(y_true, y_pred):
        one_hot = y_true[:, :1300]
        mask = y_true[:, 1300:]
        masked_logits = y_pred + mask
        return metrics.sparse_top_k_categorical_accuracy(tf.argmax(one_hot, axis=-1), masked_logits, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric


# Architecture sur réseau dense, sans LayerNormalization, sans Dropout, allégée avec skip connection 
class DensePlayer_v10(Player):
    #Méthodes statiques
    #------------------------------------------------------------------------------------------------------------------------------------

    # Décode un vecteur aplati (1300,) en [col, ligne, move_idx]
    # Pour usage avec un array (1300,) en one-hot ou probabilités
    # retourne col, ligne, move_idx
    @staticmethod
    def decode_flat_policy(flat_policy):
        best_index = np.argmax(flat_policy)
        col = best_index // (5 * 52)
        ligne = (best_index // 52) % 5
        move_id = best_index % 52
        return int(col), int(ligne), int(move_id)

    #------------------------------------------------------------------------------------------------------------------------------------

    # Constructeur
    def __init__(self):
        super().__init__()
        self.name = "DensePlayer-v10"

        #Paramètres du réseau
        self.hidden_units = [128, 128, 64]
        self.n_moves = 52
        self.with_ppo = False

        #Construction du réseau
        self.model = self._build_model()

        # Garder des références aux différentes parties du réseau
        self._identify_heads()

    #Active le mode entrainement PPO (en PPO on échantillonne depuis la distribution, en inférence on retourne la meilleure action)
    def setPPOTraining(self, with_ppo:bool):
        self.with_ppo = with_ppo

    #Effectue la prédiction du coup à jouer
    def play(self, board:Board):
        state = np.array(board.get_state())
        state = np.transpose(state, (1, 2, 0))  #(10, 5, 5) => (5, 5, 10) TODO ; à optimiser

        available_moves = board.get_available_moves()

        if len(available_moves) == 0:
            return None

        #On effectue la prédiction
        policy_logits, value = self.predict(state) #Value entre -1 et +1
        policy_logits = np.array(policy_logits).flatten()  # (1300,)

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

        #Appliquer softmax pour obtenir les probabilités
        probs = self._softmax(masked_logits)

        if self.with_ppo:
            #En PPO on échantillonne depuis la distribution
            p = probs / probs.sum() #Normalisation pour éviter un crash si ne fait pas exactement 1
            best_flat_idx = np.random.choice(len(probs), p=p)
            #Pour éviter qu'on choisisse malgré tout une action avec une probabilité quasi nulle
            if best_flat_idx not in action_to_move:
                best_flat_idx = np.argmax(probs)  # fallback greedy
            #log de la probabilité de l'action choisie sur la distribution MASQUÉE
            x_safe = np.where(masked_logits == -np.inf, -1e9, masked_logits)
            max_x = np.max(x_safe)
            log_prob = x_safe[best_flat_idx] - max_x - np.log(np.sum(np.exp(x_safe - max_x)))
            #Masque à stocker dans le buffer : 0 pour actions valides, -1e9 pour invalides
            valid_mask = np.where(masked_logits == -np.inf, -1e9, 0.0).astype(np.float32)
        else:
            #Sélectionner l'action avec la plus haute probabilité
            best_flat_idx = np.argmax(probs)

        best_action = action_to_move[best_flat_idx]

        if self.with_ppo:
            return best_action, log_prob, float(value.numpy()[0][0]), valid_mask
        else:
            return best_action

    def _softmax(self, x):
        """Softmax stable numériquement (gère les -inf)"""
        x_safe = np.where(x == -np.inf, -1e9, x)
        exp_x = np.exp(x_safe - np.max(x_safe))
        return exp_x / exp_x.sum()

    # Réalise une prédiction
    # state:(5,5,10) ou (batch,5,5,10) — sera aplati en interne
    # Retourne :
    # policy_logits : (batch, 1300)
    # value : (batch, 1)
    def predict(self, state:dict):
        # Ajouter dimension batch si nécessaire
        if len(state.shape) == 3:
            state = tf.expand_dims(state, 0)

        return self._predict_compiled(tf.cast(state, tf.float32))

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 5, 5, 10), dtype=tf.float32)])
    def _predict_compiled(self, state):
        return self.model(state, training=False)

    #Configure l'optimizeur et la loss
    def compile(self, learning_rate:float=0.001):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss=[
                keras.losses.CategoricalCrossentropy(from_logits=True),  # Policy
                keras.losses.MeanSquaredError()  # Value
            ],
            loss_weights=[1.0, 0.5],
            metrics=[
                ['accuracy'],
                ['mae']
            ]
        )

    #Compiler pour entraînement supervisé (on entraîne uniquement la policy)
    # use_mask=True : y doit être concat([one_hot (1300), valid_mask (1300)]) → shape (N, 2600)
    # use_mask=False : y est le one_hot classique → shape (N, 1300)
    def compile_for_supervised_policy(self, learning_rate=0.001, label_smoothing=0.0, weight_decay=1e-4, use_mask=False):
        # Geler la tête de valeur
        self.freeze_value_head()

        if use_mask:
            policy_loss = masked_categorical_crossentropy(label_smoothing=label_smoothing)
            policy_metrics = [masked_accuracy(), masked_top_k_accuracy(3), masked_top_k_accuracy(5), masked_top_k_accuracy(10)]
        else:
            policy_loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
            policy_metrics = ['accuracy', top_k_accuracy(3), top_k_accuracy(5), top_k_accuracy(10)]

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss=[policy_loss, None],
            metrics=[policy_metrics, []]
        )

        print(f"Modèle compilé pour entraînement supervisé (policy seulement, label_smoothing={label_smoothing}, weight_decay={weight_decay}, use_mask={use_mask})")

    #Compiler pour entraînement RL (tout entraînable)
    def compile_for_rl(self, learning_rate=3e-4):
        self.unfreeze_value_head()
        self.unfreeze_trunk()
        print("Toutes les couches dégelées pour RL")

    #Entraîne le modèle
    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)

    #Sauvegarde le modèle
    def save(self, filepath):
        self.model.save(filepath)

    #Charge les poids
    def load_weights(self, filepath, **kwargs):
        self.model.load_weights(filepath, **kwargs)

    #Sauvegarde les poids
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    #Affiche un résumé du modèle
    def summary(self):
        return self.model.summary()

    #Gèle la tête de valeur
    def freeze_value_head(self):
        for layer in self.value_layers:
            layer.trainable = False
        print(f"Gelé {len(self.value_layers)} layers de la tête de valeur")

    #Dégèle la tête de valeur
    def unfreeze_value_head(self):
        for layer in self.value_layers:
            layer.trainable = True
        print(f"Dégelé {len(self.value_layers)} layers de la tête de valeur")

    #Gèle le tronc commun
    def freeze_trunk(self):
        for layer in self.trunk_layers:
            layer.trainable = False
        print(f"Gelé {len(self.trunk_layers)} layers du tronc")

    #Dégèle le tronc commun
    def unfreeze_trunk(self):
        for layer in self.trunk_layers:
            layer.trainable = True
        print(f"Dégelé {len(self.trunk_layers)} layers du tronc")

    #Retourne les variables entraînables (utile pour PPO)
    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    #Construction du réseau dense
    def _build_model(self):
        #Couche d'entrée (5, 5, 10)
        inputs = keras.Input(shape=(5, 5, 10), name='state_input')

        #Aplatir le board en vecteur (250,)
        x = layers.Flatten(name='trunk_flatten')(inputs)

        #Tronc commun : couches denses empilées avec skip connection sur les blocs 0→1 (128→128)
        for i, units in enumerate(self.hidden_units):
            shortcut = x if i == 1 else None   # sauvegarde avant le bloc 1
            x = layers.Dense(units, name=f'trunk_dense_{i}')(x)
            x = layers.Activation('relu', name=f'trunk_relu_{i}')(x)
            if shortcut is not None:            # skip connection après le bloc 1
                x = layers.Add(name='trunk_skip_add')([x, shortcut])

        #Tête de politique (Policy)
        # Couche intermédiaire avant la projection finale (évite le saut direct 256→1300)
        policy = layers.Dense(64, activation='relu', name='policy_hidden')(x)
        policy_logits = layers.Dense(5 * 5 * self.n_moves, activation=None, name='policy_logits')(policy)
        #sortie: policy_logits → shape (batch, 1300)

        #Tête de valeur
        value = layers.Dense(64, activation='relu', name='value_dense1')(x)
        value = layers.Dense(32, activation='relu', name='value_dense2')(value)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value)
        #sortie: value_output → shape (batch, 1)

        model = keras.Model(
            inputs=inputs,
            outputs=[policy_logits, value_output],
            name='OnitamaNetwork-v10-Dense'
        )

        return model

    #Identifie les couches de chaque tête
    def _identify_heads(self):
        self.policy_layers = []
        self.value_layers = []
        self.trunk_layers = []

        for layer in self.model.layers:
            if 'policy' in layer.name:
                self.policy_layers.append(layer)
            elif 'value' in layer.name:
                self.value_layers.append(layer)
            else:
                self.trunk_layers.append(layer)
