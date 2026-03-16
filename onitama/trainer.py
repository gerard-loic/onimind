
from players import Player, HumanPlayer, RandomPlayer, HeuristicPlayer, LookAheadHeuristicPlayer
from constants import *
from board import Board
from game import Game, GameSession
from pathlib import Path
import pickle
import numpy as np

# Classe "parente" des data trainers
class DataTrainer:
    def __init__(self):
        pass

    def save_experience(self, player:Player, state:list, action:list, log_prob:float, value:float, valid_mask=None):
        pass

    def close(self, winner:Player):
        pass

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
    # probability:float : Probabilité de 'laction sous la politique courante
    # value:float : Estimation de V(s)
    def save_experience(self, player:Player, state:list, action:list, log_prob:float, value:float, valid_mask=None):
        super().save_experience(player, state, action, log_prob, value, valid_mask)

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

    
    # termine la trajectoire et calcule GAE
    def close(self, winner):
        super().close(winner)

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


    
    # retourne les données sous forme de tensors et vide le buffer
    def get(self)->dict:
        data = {
            'states' : np.array(self.states, dtype=np.float32), #(N, 5, 5, 10)
            'actions' : np.array(self.actions, dtype=np.int32),
            'log_probs' : np.array(self.log_probs, dtype=np.float32),
            'masks' : np.array(self.masks, dtype=np.float32), #(N, 1300) : 0 valide, -1e9 invalide
            'advantages' : np.array(self.advantages, dtype=np.float32),
            'returns' : np.array(self.returns, np.float32)
        }

        #Normalisation des avantages, permet de stabiliser les dradients PPO
        adv = data['advantages']
        data['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        self._clear()
        return data
    
    def __len__(self):
        return len(self.states)
    


    


# Classe permettant la gestion des parties visant à générer des données d'entraînement
class RegularDataTrainer(DataTrainer):
    #Méthodes statiques
    #------------------------------------------------------------------------------------------------------------------------------------

    # Permet de récupérer des données d'entraînement déjà générées
    @staticmethod
    def getTrainedData(filepath:str):
        all_data = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    all_data.extend(batch)
                except EOFError:
                    break
        return all_data
    
    #------------------------------------------------------------------------------------------------------------------------------------

    # Constructeur
    # p1:Player : Joueur P1
    # p2:Player : Joueur P2
    # p1_record:bool : Est ce qu'on enregistre les données pour P1
    # p2_record:bool : Est ce qu'on enregistre les données pour P2
    # save_only_wins:bool : Est ce qu'on enregistre seulement les données des parties où le joueur a gagné
    # x_file_destination:str : Emplacement du fichier de destination des features
    # y_file_destination:str : Emplacement du fichier de destination des labels
    # v_file_destination:str : Emplacement du fichier de destination des outcomes (±1) pour la value head (optionnel)
    # override:bool : Si le fichier existe déjà on l'écrase
    def __init__(self, p1:Player, p2:Player, p1_record:bool, p2_record:bool, save_only_wins:bool, x_file_destination:str, y_file_destination:str, v_file_destination:str=None, m_file_destination:str=None, override:bool=False):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p1_record = p1_record
        self.p2_record = p2_record
        self.save_only_wins = save_only_wins
        self.x_file_destination = x_file_destination
        self.y_file_destination = y_file_destination
        self.v_file_destination = v_file_destination
        self.m_file_destination = m_file_destination

        #Initialisation du cache
        self._init_cache()

        #Si suppression
        if override:
            for filepath in [self.x_file_destination, self.y_file_destination, self.v_file_destination, self.m_file_destination]:
                if filepath is None:
                    continue
                f = Path(filepath)
                if f.exists():
                    f.unlink()
                    print(f"File {filepath} deleted !")

    # Initialise le cache
    def _init_cache(self):
        self.cache_states_p1 = []
        self.cache_actions_p1 = []
        self.cache_masks_p1 = []
        self.cache_states_p2 = []
        self.cache_actions_p2 = []
        self.cache_masks_p2 = []
        self.cache_n_steps_p1 = 0  # nombre de pas de la partie en cours pour p1
        self.cache_n_steps_p2 = 0  # nombre de pas de la partie en cours pour p2

    # Enregistre une expérience, c'est à dire un couple état / action (met en cache)
    # player:Player : joueur
    # state:list[5:5:10] : Etat (Matrice de 5x5x10)
    # action:list(rel_x:int, rel_y:int, move_idx:int) : Action 
    def save_experience(self, player:Player, state:list, action:list, log_prob:float, value:float, valid_mask=None):
        super().save_experience(player, state, action, log_prob, value, valid_mask)

        t = np.array(state)

        action = [action.from_pos[0], action.from_pos[1], action.move_idx]

        # Mise en cache
        if (player == self.p1 and self.p1_record):
            self.cache_states_p1.append(state)
            self.cache_actions_p1.append(action)
            self.cache_masks_p1.append(valid_mask)
            self.cache_n_steps_p1 += 1
        elif (player == self.p2 and self.p2_record):
            self.cache_states_p2.append(state)
            self.cache_actions_p2.append(action)
            self.cache_masks_p2.append(valid_mask)
            self.cache_n_steps_p2 += 1

    # Enregistre les expériences en cache dans le fichier (quand une partie est terminée)
    def close(self, winner:Player):
        super().close(winner)

        x_to_write = []
        y_to_write = []
        v_to_write = []
        m_to_write = []

        # Outcomes du point de vue de chaque joueur
        # get_state() est perspective-aware : +1 si le joueur courant à cet état a finalement gagné
        outcome_p1 = +1.0 if winner == self.p1 else -1.0
        outcome_p2 = +1.0 if winner == self.p2 else -1.0

        # Si on enregistre uniquement les parties qui se solvent par une victoire, ne prendre en compte que si le joueur a hgagné
        if self.save_only_wins:
            if winner == self.p1:
                x_to_write += self.cache_states_p1
                y_to_write += self.cache_actions_p1
                v_to_write += [outcome_p1] * self.cache_n_steps_p1
                m_to_write += self.cache_masks_p1
            elif winner == self.p2:
                x_to_write += self.cache_states_p2
                y_to_write += self.cache_actions_p2
                v_to_write += [outcome_p2] * self.cache_n_steps_p2
                m_to_write += self.cache_masks_p2
        else:
            x_to_write += self.cache_states_p1
            x_to_write += self.cache_states_p2
            y_to_write += self.cache_actions_p1
            y_to_write += self.cache_actions_p2
            v_to_write += [outcome_p1] * self.cache_n_steps_p1
            v_to_write += [outcome_p2] * self.cache_n_steps_p2
            m_to_write += self.cache_masks_p1
            m_to_write += self.cache_masks_p2

        # Enregistrement à la suite du fichier
        with open(self.x_file_destination, 'ab') as f:
            pickle.dump(x_to_write, f)
        with open(self.y_file_destination, 'ab') as f:
            pickle.dump(y_to_write, f)
        if self.v_file_destination is not None:
            with open(self.v_file_destination, 'ab') as f:
                pickle.dump(v_to_write, f)
        if self.m_file_destination is not None:
            with open(self.m_file_destination, 'ab') as f:
                pickle.dump(m_to_write, f)

        # On réinitialise le cache
        self._init_cache()




if __name__ == "__main__":
    
    training_plan = [
        (
            LookAheadHeuristicPlayer(max_depth=1, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=1, heuristic_function="heuristic_defensive"),
            "test"
        ),
        """
        (
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_mobility"),
            "agressive3-vs-mobility3"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_positional"),
            "agressive3-vs-positional3"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_regular"),
            "agressive3-vs-regular3"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_defensive"),
            "agressive3-vs-defensive2"
        )
        """
    ]

    i = 1
    for p1, p2, filename in training_plan:
        print(f"Training session {i}")
        trainer = RegularDataTrainer(
            p1=p1,
            p2=p2,
            p1_record=True,
            p2_record=True,
            save_only_wins=True,
            x_file_destination=f"../data/{filename}-states.pkl",
            y_file_destination=f"../data/{filename}-actions.pkl",
            override=True
        )
        gameSession = GameSession(player_one=p1, player_two=p2, number_of_games=5000, trainer=trainer)
        gameSession.start()
        print(gameSession.getStats())

    
    


    """
    

    all = RegularDataTrainer.getTrainedData(filepath="../data/training-data-heuristic-vs-laheuristic3-actions.pkl")
    print(len(all))


    all2 = RegularDataTrainer.getTrainedData(filepath="../data/training-data-heuristic-vs-laheuristic2-actions.pkl")
    print(len(all2))
    print(all2[1])
    """