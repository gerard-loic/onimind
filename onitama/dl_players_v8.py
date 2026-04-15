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
            tf.argmax(y_true, axis=-1),  # Convertir one-hot en index
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

        # Identifier les actions valides pour y appliquer le label smoothing
        valid = tf.cast(mask > -1e8, tf.float32)                          # (batch, 1300) : 1=valide, 0=invalide
        n_valid = tf.reduce_sum(valid, axis=-1, keepdims=True)            # (batch, 1) : nb de coups valides
        smooth_per_class = label_smoothing / tf.maximum(n_valid, 1.0)    # évite la division par 0
        smoothed = (1.0 - label_smoothing) * one_hot + smooth_per_class * valid

        log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
        return tf.reduce_mean(-tf.reduce_sum(smoothed * log_probs, axis=-1))

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
    def metric(y_true, y_pred):
        one_hot = y_true[:, :1300]
        mask = y_true[:, 1300:]
        masked_logits = y_pred + mask
        return metrics.sparse_top_k_categorical_accuracy(tf.argmax(one_hot, axis=-1), masked_logits, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric

# Issu de l'architecture de v6, mais avec 10 couches de blocs résiduels (plus profond)
class CNNPlayer_v8(Player):

    # Constructeur
    # dropout_rate:float : % de dropout pour les tetes
    # residual_dropout_rate:float : % de dropout sur les blocs résiduels
    def __init__(self, model_file:str=None, dropout_rate:float=0.4, residual_dropout_rate:float=0.1):
        super().__init__()
        self.name = "CNNPlayerV8"

        #Paramètres du réseau
        self.n_filters = 128        
        self.kernel_size = 3        #Taille du filtre
        self.n_residual_blocs = 10   #Nombre de blocs résiduels
        self.n_moves = 52
        self.dropout_rate = dropout_rate                    
        self.residual_dropout_rate = residual_dropout_rate  
        self.with_ppo = False   

        #Construction du réseau
        if model_file:
            self.model = tf.keras.models.load_model(model_file)
        else:
            self.model = self._build_model()

        # Garder des références aux différentes parties du réseau
        self._identify_heads()

    #Active le mode entrainement PPO (en PPO on échantillonne depuis la distribution, en inférence on retourne la meilleure action)
    def setPPOTraining(self, with_ppo:bool):
        self.with_ppo = with_ppo

    #Effectue la prédiction du coup à jouer
    def play(self, board:Board):
        #On récupère le state
        state = np.array(board.get_state())
        state = np.transpose(state, (1, 2, 0))  #(10, 5, 5) => (5, 5, 10) TODO ; à optimiser

        #On récupère les mouvements possibles
        available_moves = board.get_available_moves()

        if len(available_moves) == 0:
            return None

        #On effectue la prédiction
        policy_logits, value = self.predict(state)
        # value = float entre -1 (position perdue) et +1 (position gagnée)
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

            #Sécurité
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

    #Softmax stable numériquement (gère les -inf)
    def _softmax(self, x):
        x_safe = np.where(x == -np.inf, -1e9, x)
        exp_x = np.exp(x_safe - np.max(x_safe))
        return exp_x / exp_x.sum()

    # Réalise une prédiction
    # state:(5,5,10) ou (batch,5,5,10)
    # Retourne :
    # policy_logits : (batch, 1300)
    # value : (batch, 1)
    def predict(self, state:dict):
        # Ajouter dimension batch si nécessaire
        if len(state.shape) == 3:
            state = tf.expand_dims(state, 0)

        return self._predict_compiled(tf.cast(state, tf.float32))

    #Compile la fonction la première fois qu'elle est appelée, les appels suivant executent dctmt la version comppulée. Plus rapide surtout en inférence répétée
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 5, 5, 10), dtype=tf.float32)])
    def _predict_compiled(self, state):
        return self.model(state, training=False)
    

    #Compiler pour entraînement supervisé (on entraîne uniquement la policy)
    # use_mask=True : y doit être concat([one_hot (1300), valid_mask (1300)]) → shape (N, 2600)
    # use_mask=False : y est le one_hot classique → shape (N, 1300)
    def compile_for_supervised_policy(self, learning_rate=0.001, label_smoothing=0.1, weight_decay=1e-4, use_mask=False):
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
    
    #Compiler pour entraînement RL (nom de méthode à reprendre, pas vraiment compilation)
    def compile_for_rl(self, learning_rate=3e-4):

        # Dégeler tout
        self.unfreeze_value_head()
        self.unfreeze_trunk()
        
        # Pour PPO on n'utilise pas compile, mais on s'assure que tout est dégelé
        print("Toutes les couches dégelées pour RL")

    #Entraîne le modèle (kwargs permet de transmettre à fit tous les autres arguments nommés supplémentaires éventuellement transmis via la fonction)
    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)
    
    #Sauvegarde le modèle (y compris l'archirecrure, les poids, l'optimiseur et la configuration de la compulation)
    def save(self, filepath):
        self.model.save(filepath)
    
    #Charge les poids
    def load_weights(self, filepath, **kwargs):
        self.model.load_weights(filepath, **kwargs)
    
    #Sauvegarde les poids (uniquement les poids)
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    #Affiche un résumé du modèle
    def summary(self):
        return self.model.summary()
    
    #Gèle la tête de valeur (value) -> qui du coup ne sera pas entraînée
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
        """Dégèle le tronc commun"""
        for layer in self.trunk_layers:
            layer.trainable = True
        print(f"Dégelé {len(self.trunk_layers)} layers du tronc")
    
    #Retourne les variables entraînables (utile pour PPO)
    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    #Construction du réseau
    def _build_model(self):
        #Couche d'entrée (5, 5, 10)
        inputs = keras.Input(shape=(5, 5, 10), name='state_input')

        #Tronc commun
        #Couche de convolution 2D
        x = layers.Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same', 
            name='conv_input'
        )(inputs)  

        #Normalise les valeurs
        x = layers.BatchNormalization(name='bn_input')(x)

        #Applique ReLU
        x = layers.Activation('relu', name='relu_input')(x)

        #Blocs résiduels
        for i in range(self.n_residual_blocs):
            x = self._residual_block(x, name=f'res_block_{i}', dropout_rate=self.residual_dropout_rate)

        #Tête de politique (Policy) => prévoit l'action à réaliser
        # Couche intermédiaire : combine les canaux du tronc (128 → 64)
        policy = layers.Conv2D(
            filters=64,
            kernel_size=1,
            padding='same',
            name='policy_conv'
        )(x)
        policy = layers.BatchNormalization(name='policy_bn')(policy)
        policy = layers.Activation('relu', name='policy_relu')(policy)
        policy = layers.Dropout(self.dropout_rate, name='policy_dropout')(policy)

        #Dernière couche de la tête politique : produit les scores pour chaque action
        policy_logits = layers.Conv2D(
            filters=self.n_moves,   
            kernel_size=1,
            padding='same',
            activation=None,    #Pas d'activation, on garde les valeurs brutes
            #Le softmax sera appliqué plus tard pour convertir en probabilités
            name='policy_conv_out'
        )(policy)

        #Aplatir : (batch, 5, 5, 52) → (batch, 1300)
        policy_logits = layers.Reshape((5 * 5 * self.n_moves,), name='policy_logits')(policy_logits)
        #sortie: policy_logits → shape (batch, 1300)

        #Tête de valeur (estime si l'état est favorable ou non)
        value = layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding='same',
            name='value_adapter_conv'
        )(x)
        value = layers.BatchNormalization(name='value_adapter_bn')(value)
        value = layers.Activation('relu', name='value_adapter_relu')(value)

        value = layers.Conv2D(
            filters=16,      
            kernel_size=1,
            padding='same',
            name='value_conv'
        )(value)
        value = layers.BatchNormalization(name='value_bn')(value)
        value = layers.Activation('relu', name='value_relu')(value)
        value = layers.Flatten(name='value_flatten')(value)
        #évite l'overfitting
        value = layers.Dropout(self.dropout_rate, name='value_dropout')(value)
        #Couche Fully connectée, sert à combiner toutes les informations spatiales pour évaluer la position globale
        value = layers.Dense(128, activation='relu', name='value_dense1')(value)
        #Sortie : 1 neurone, le score de la position, tanh me permet de borner la sortie entre -1 et 1
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value)
        
        #Compiler le modèle
        model = keras.Model(
            inputs=inputs,
            outputs=[policy_logits, value_output],
            name='OnitamaNetwork-v8'
        )
        
        return model
        

    #Construction d'un bloc résiduel
    def _residual_block(self, x, name:str, dropout_rate:float=0.0):
        # Branche principale
        conv1 = layers.Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            name=f'{name}_conv1'
        )(x)
        bn1 = layers.BatchNormalization(name=f'{name}_bn1')(conv1)
        relu1 = layers.Activation('relu', name=f'{name}_relu1')(bn1)

        conv2 = layers.Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            name=f'{name}_conv2'
        )(relu1)
        bn2 = layers.BatchNormalization(name=f'{name}_bn2')(conv2)

        # Skip connection : addition puis activation
        add = layers.Add(name=f'{name}_add')([bn2, x])
        output = layers.Activation('relu', name=f'{name}_relu2')(add)

        if dropout_rate > 0.0:
            output = layers.Dropout(dropout_rate, name=f'{name}_dropout')(output)

        return output
    
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