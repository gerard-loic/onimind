from players import Player
from board import Board
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class CNNPlayer_v1(Player):
    
    #Méthodes statiques
    #------------------------------------------------------------------------------------------------------------------------------------

    # Décode un vecteur aplati (1300,) en [col, ligne, move_idx] 
    # Pour usage avec un array (1300,) en one-hot ou probabilités
    # retourne col, ligne, move_idx
    @staticmethod
    def decode_flat_policy(flat_policy):
        # Trouver l'index du maximum (ou du 1.0 si one-hot)
        best_index = np.argmax(flat_policy)
        
        # Décoder l'index
        col = best_index // (5 * 52)
        ligne = (best_index // 52) % 5
        move_id = best_index % 52
        
        return int(col), int(ligne), int(move_id)


    #------------------------------------------------------------------------------------------------------------------------------------
    """
    n_filters: nombre de filtres (ou canaux) dans les couches convolutionnelles
    """

    # Constructeur
    # dropout_rate:float : % de dropout
    def __init__(self, dropout_rate:float=0.3):
        super().__init__()
        self.name = "CNNPlayer_v1"

        #Paramètres du réseau
        self.n_filters = 128        #Canaux de sortie de la couche de convolution (chaque filtre détecte un motif différent)
        self.kernel_size = 3        #Taille du filtre : 3x3 pixels
        self.n_residual_blocs = 5   #Nombre de blocs résiduels
        self.n_moves = 52
        self.dropout_rate = dropout_rate  #Taux de dropout pour la régularisation

        #Construction du réseau
        self.model = self._build_model()

        # Garder des références aux différentes parties du réseau
        self._identify_heads()


    def play(self, board:Board):
        #On récupère le state
        state = np.array(board.get_state())
        #On le transpose (10, 5, 5) => (5, 5, 10)
        state = np.transpose(state, (1, 2, 0))

        #On récupère les mouvements possibles
        available_moves = board.get_available_moves()

        if len(available_moves) == 0:
            return None

        #On effectue la prédiction
        policy_logits, value = self.predict(state, training=False)
        policy_logits = np.array(policy_logits).flatten()  # (1300,)

        #Créer un masque des actions valides
        #On met -inf partout sauf pour les actions valides
        masked_logits = np.full(1300, -np.inf)

        #Pour chaque action valide, on conserve le logit correspondant
        action_to_move = {}  # flat_idx -> Action (pour retrouver l'action après)
        #TODO : implémeter le get flat index dans l'action ?
        for action in available_moves:
            col, row = action.from_pos
            move_idx = action.move_idx
            #Calcul de l'index flat : col * (5 * 52) + row * 52 + move_idx
            flat_idx = col * (5 * 52) + row * 52 + move_idx
            masked_logits[flat_idx] = policy_logits[flat_idx]
            action_to_move[flat_idx] = action

        #Appliquer softmax pour obtenir les probabilités
        probs = self._softmax(masked_logits)

        #Sélectionner l'action avec la plus haute probabilité
        best_flat_idx = np.argmax(probs)
        best_action = action_to_move[best_flat_idx]

        return best_action

    def _softmax(self, x):
        """Softmax stable numériquement (gère les -inf)"""
        #Remplacer -inf par une très petite valeur pour éviter les NaN
        x_safe = np.where(x == -np.inf, -1e9, x)
        exp_x = np.exp(x_safe - np.max(x_safe))
        return exp_x / exp_x.sum()

    # Réalise une prédiction
    # state:dict(5,5,10) ou (batch,5,5,10)
    # training:bool : ????
    # Retourne : 
    # policy_logits : (batch, 5, 5, 52) 
    # value : (batch, 1)
    def predict(self, state:dict, training:bool=False):
        # Ajouter dimension batch si nécessaire
        if len(state.shape) == 3:
            state = tf.expand_dims(state, 0)
        
        return self.model(state, training=training)
    
    #Configure l'optimizeur et la loss
    def compile(self, learning_rate:float=0.001):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss=[
                #Le modèle a deux sorties (politique et valeur), donc deux loss différentes
                keras.losses.CategoricalCrossentropy(from_logits=True),  # Policy
                keras.losses.MeanSquaredError()  # Value
            ],
            #La politique a plus de poids (1.0 vs 0.5), donc le modèle se concentre davantage sur bien jouer que sur bien évaluer.
            loss_weights=[1.0, 0.5],
            metrics=[
                ['accuracy'],  # Policy metrics -> % de coups correctement prédits
                ['mae']  # Value metrics -> Erreur moyenne absolue sur le score
            ]
        )

    #Compiler pour entraînement supervisé (on entraîne uniquement la policy)
    def compile_for_supervised_policy(self, learning_rate=0.001, label_smoothing=0.1):
        # Geler la tête de valeur
        self.freeze_value_head()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[
                #Label smoothing réduit la confiance excessive du modèle (régularisation)
                keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),  # Policy
                None  # Pas de loss pour la valeur
            ],
            metrics=[
                ['accuracy'],  # Policy metrics
                []  # Pas de metrics pour value
            ]
        )

        print(f"Modèle compilé pour entraînement supervisé (policy seulement, label_smoothing={label_smoothing})")
    
    #Compiler pour entraînement RL (tout entraînable)
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
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
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
        #L'entrée inputs est une grille (le plateau d'Onitama, 5×5)
        #Chaque filtre 3×3 parcourt cette grille
        #À chaque position, il calcule une somme pondérée des 9 valeurs voisines
        #Cela produit une "feature map" qui capture des motifs locaux (positions de pièces adjacentes, menaces, etc.)
        x = layers.Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same', #Ajoute du padding pour que la sortie ait la même taille que l'entrée
            name='conv_input'
        )(inputs)   #Applique la couche aux données

        #Normalise les valeurs pour qu'elles aient une moyenne proche de 0 et un écart-type proche de 1
        #Pourquoi ?
        #Stabilise l'entraînement en évitant que les valeurs explosent ou s'effondrent au fil des couches
        #Accélère la convergence : le réseau apprend plus vite car les gradients sont mieux calibrés
        #Réduit légèrement l'overfitting
        x = layers.BatchNormalization(name='bn_input')(x)
        #Applique ReLU
        x = layers.Activation('relu', name='relu_input')(x)

        #Blocs résiduels
        for i in range(self.n_residual_blocs):
            x = self._residual_block(x, name=f'res_block_{i}')

        #Tête de politique (Policy) => prévoit l'action à réaliser
        #Un filtre 1×1 ne regarde qu'un seul pixel à la fois. Son rôle n'est pas de détecter des patterns spatiaux, mais de combiner les canaux (réduire/mélanger les features). C'est une sorte de "projection" dans un espace de dimension 32.
        policy = layers.Conv2D(
            filters=32,     #On réduit la dimentionnalité
            kernel_size=1,
            padding='same',
            name='policy_conv'
        )(x)
        policy = layers.BatchNormalization(name='policy_bn')(policy)
        policy = layers.Activation('relu', name='policy_relu')(policy)
        #Dropout pour régularisation (évite l'overfitting)
        policy = layers.Dropout(self.dropout_rate, name='policy_dropout')(policy)
        #Dernière couche de la tête politique : produit les scores pour chaque action
        policy_logits = layers.Conv2D(
            filters=self.n_moves,   #Un filtre par mouvement possible (les déplacements des cartes)
            kernel_size=1,
            padding='same',
            activation=None,    #Pas d'activation, on garde les valeurs brutes
            #Le softmax sera appliqué plus tard pour convertir en probabilités
            name='policy_conv_out'
        )(policy)
        #Aplatir pour la cross-entropy : (batch, 5, 5, 52) → (batch, 1300)
        policy_logits = layers.Reshape((5 * 5 * self.n_moves,), name='policy_logits')(policy_logits)
        #sortie: policy_logits → shape (batch, 1300)

        #Tête de valeur (estime si l'état est favorable ou non)
        value = layers.Conv2D(
            filters=4,      #Valeur plus simple, donc moins de filtres
            kernel_size=1,
            padding='same',
            name='value_conv'
        )(x)
        value = layers.BatchNormalization(name='value_bn')(value)
        value = layers.Activation('relu', name='value_relu')(value)
        #Applatit le tenseur 3D en vecteur 1D
        value = layers.Flatten(name='value_flatten')(value)
        #Dropout pour régularisation (évite l'overfitting)
        value = layers.Dropout(self.dropout_rate, name='value_dropout')(value)
        #Couche Fully connectée, sert à combiner toutes les informations spatiales pour évaluer la position globale
        value = layers.Dense(64, activation='relu', name='value_dense1')(value)
        #Sortie : 1 neurone, le score de la position, tanh me permet de borner la sortie entre -1 et 1
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value)
        
        #Compiler le modèle
        model = keras.Model(
            inputs=inputs,
            outputs=[policy_logits, value_output],
            name='OnitamaNetwork'
        )
        
        return model
        

    #Construction d'un bloc résiduel
    #Le bloc résiduel vise à résoudre deux problèmes :
    #Problème 1 : Vanishing Gradient
    #Lors du backpropagation, le gradient se multiplie à chaque couche
    #gradient × 0.9 × 0.9 × 0.9 × ... (20 fois) ≈ 0.12 (très petit !)
    #Les premières couches n'apprennent presque plus
    #Problème 2 : Dégradation
    #Paradoxalement, ajouter plus de couches peut diminuer les performances
    #Le réseau a du mal à apprendre même la fonction identité
    def _residual_block(self, x, name:str):
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

        # Skip connection
        add = layers.Add(name=f'{name}_add')([bn2, x])
        output = layers.Activation('relu', name=f'{name}_relu2')(add)

        return output
    
    #Identifie les couches de chaque tête
    def _identify_heads(self):
        """Identifie les layers de chaque tête"""
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