
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


