from __future__ import annotations
from typing import TYPE_CHECKING

from board import Board
from card import Card
from players import Player, RandomPlayer, HumanPlayer, HeuristicPlayer, LookAheadHeuristicPlayer, MCTSPlayer
from dl_players_v2 import CNNPlayer_v2
from dl_players_v1 import CNNPlayer_v1
from dl_players_v3 import CNNPlayer_v3
from dl_players_v4 import CNNPlayer_v4
from dl_minimax import LookAheadDlPlayer
import random
from constants import *
from tqdm import tqdm
import numpy as np
import time
from card import Card

if TYPE_CHECKING:
    from trainer import DataTrainer

# Permet de jouer une partie
class Game:
    # Constructeur
    # player_one:Player : joueur P1
    # player_two:Player : joueur P2
    # verbose:bool : indique si la partie est silencieuse ou si on fait un affichage dans la console
    # trainer:DataTrainer : si transmis, utilise le controlleur fourni pour sauvegarder les données d'entraînement
    def __init__(self, player_one:Player, player_two:Player, verbose:bool=True, trainer:DataTrainer=None):

        self.verbose = verbose
        self.timesP1 = []
        self.timesP2 = []
        self.trainer = trainer


        self.player_one = player_one
        self.player_two = player_two

        # On initialise un nouveau jeu

        # on choisit aléatoirement les cartes avec lesquelles on va jouer
        samples = Card.getCards(nb=5)
        current_player_cards = samples[0:2]
        next_player_cards = samples[2:4]
        neutral_card = samples[4]
        
        # On détermine le premier joueur
        self.first_player = COLOR_MAPPING[neutral_card.color]

        #On initialise les joueurs
        self.current_player = player_one if self.first_player == 1 else player_two
        self.current_player.set_position(position=PLAYER_ONE_POSITION)
        self.next_player = player_two if self.first_player == 1 else player_one
        self.next_player.set_position(position=PLAYER_TWO_POSITION)

        # On initialise le plateau
        self.board = Board(
            player_to_move=PLAYER_ONE_POSITION,
            current_player_cards=current_player_cards,
            next_player_cards=next_player_cards,
            neutral_card=neutral_card
        )

        # On place les pièces (disposition de  ddépart de base)
        self.board.board[0][0] = PLAYER_TWO_STUDENT
        self.board.board[1][0] = PLAYER_TWO_STUDENT
        self.board.board[2][0] = PLAYER_TWO_MASTER
        self.board.board[3][0] = PLAYER_TWO_STUDENT
        self.board.board[4][0] = PLAYER_TWO_STUDENT
        self.board.board[0][4] = PLAYER_ONE_STUDENT
        self.board.board[1][4] = PLAYER_ONE_STUDENT
        self.board.board[2][4] = PLAYER_ONE_MASTER
        self.board.board[3][4] = PLAYER_ONE_STUDENT
        self.board.board[4][4] = PLAYER_ONE_STUDENT

        if self.verbose:
            print("#####################################################################")
            print("# NEW GAME !")
            print(f"# PLAYER X : {self.current_player.name}")
            print(f"# PLAYER O : {self.next_player.name}")
            print("#####################################################################")

    # Démarre la partie
    # return_winner:bool : Si TRUE : retourne une information sur le gagnant
    # max_turns:int : limite de tours, pour éviter les situations où les joueurs jouent en boucle sans jamais s'arrêter
    def playGame(self, return_winner:bool=False, max_turns:int=200, play_once_only:bool=False, return_move:bool=False):
        turn_count = 0
        while True:
            if self.verbose:
                print(self.board)

            # Protection contre les boucles infinies
            turn_count += 1
            if turn_count > max_turns:
                if self.verbose:
                    print(f"Game ended by turn limit ({max_turns} turns) - Draw")
                if return_winner:
                    if self.trainer:
                        self.trainer.close(winner=None)
                    return 0  # Match nul
                break

            if self.trainer is not None:
                state = self.board.get_state()
            
            #On fait jouer le joueur
            ts_before = int(time.time() * 1000)
            result = self.current_player.play(board=self.board)
            #Dans le cas d'un entraînement PPO, retourne 3 ou 4 valeurs
            if type(result) is tuple:
                if len(result) == 4:
                    action, log_prob, value, valid_mask = result
                else:
                    action, log_prob, value = result
                    valid_mask = None
            else:
                action, log_prob, value, valid_mask = result, None, None, None
            ts_after = int(time.time() * 1000)

            #On retourne l'information au trainer (si défini)
            if self.trainer is not None and action is not None:
                self.trainer.save_experience(player=self.current_player, state=state, action=action, log_prob=log_prob, value=value, valid_mask=valid_mask)

            #On réalise l'action (uniquement si action possible)
            if action is not None:
                self.board.play_move(action=action)
            else:
                self.board.play_default_move()
            
            #On enregistre els stats
            if self.current_player == self.player_one:
                self.timesP1.append(ts_after-ts_before)
            else:
                self.timesP2.append(ts_after-ts_before)

            #On récupère l'état du jeu
            game_ended, winner = self.board.game_has_ended()

            #Si le jeu est terminé
            if game_ended:
                if self.trainer is not None:
                    if winner == self.player_one.position:
                        self.trainer.close(winner=self.player_one)
                    else:
                        self.trainer.close(winner=self.player_two)

                if self.verbose:
                    print(self.board)
                    typePlayer = 'X' if winner == PLAYER_ONE_POSITION else 'O'
                    if winner == self.player_one.position:
                        print(f"Player {typePlayer} - {self.player_one.name} wins !")
                    else:
                        print(f"Player {typePlayer} - {self.player_two.name} wins !")
                if play_once_only:
                    if return_move:
                        return action
                if return_winner:
                    if winner == self.player_one.position:
                        return 1
                    else:
                        return 2
                break
            
            #On intervertit les références des joueurs en cours et suivant
            self.current_player, self.next_player = self.next_player, self.current_player

            if play_once_only:
                if return_move:
                    return action
                    
                return None



# Joue un nombre défini de parties
class GameSession:
    # Constructeur
    # player_one:Player : joueur P1
    # player_two:Player : joueur P2
    # number_of_games:int : nombre de parties à jouer
    # verbose:bool : indique si la partie est silencieuse ou si on fait un affichage dans la console
    # trainer:DataTrainer : si transmis, utilise le controlleur fourni pour sauvegarder les données d'entraînement
    def __init__(self, player_one:Player, player_two:Player, number_of_games:int=1, verbose:bool=False, trainer:DataTrainer=None):
        self.player_one = player_one
        self.player_two = player_two
        self.number_of_games = number_of_games
        self.verbose = verbose

        self.winP1 = 0
        self.winP2 = 0
        self.draw = 0
        self.timeP1 = []
        self.timeP2 = []

        self.trainer = trainer

    # Démarre les parties (avec une barre de progression)
    def start(self):
        for _ in tqdm(range(self.number_of_games), f"Games  "):
            game = Game(
                player_one=self.player_one, 
                player_two=self.player_two, 
                verbose=self.verbose, 
                trainer=self.trainer
            )
            r = game.playGame(return_winner=True)
            if r == 1:
                self.winP1 += 1
            elif r == 2:
                self.winP2 += 1
            else:
                self.draw += 1  # r == 0 signifie match nul
            self.timeP1.append(np.mean(game.timesP1))
            self.timeP2.append(np.mean(game.timesP2))

    # Retourne des stats sur les parties jouées
    def getStats(self):
        return {
            'p1_win' : self.winP1,
            'p2_win' : self.winP2,
            'draw' : self.draw,
            'p1_avg_time' : np.mean(self.timeP1),
            'p2_avg_time' : np.mean(self.timeP2)
        }

if __name__ == "__main__":

    #humain = HumanPlayer()
    #p2 = LookAheadHeuristicPlayer()
    #p1 = CNNPlayer_v1()
    #p1.load_weights("../saved-models/CNNPlayer-withdropout-weights.weights.h5")
    #p2 = CNNPlayer_v1()
    #p2.load_weights("../saved-models/CNNPlayer-withdropout-augmented-weights.weights.h5")
    #p3 = CNNPlayer_v2(with_heuristic=True)
    #p3.load_weights("../saved-models/CNNPlayer-withdropout-datalarge-weights.weights.h5")
    #p4 = CNNPlayer()
    #p4.load_weights("../saved-models/CNNPlayer-withdropout-datalarge-dropout-weights.weights.h5")

    #p4 = CNNPlayer_v3()
    #p4.load_weights("../saved-models/CNNPlayer-v3-weights.weights.h5")
    
    #pr = LookAheadHeuristicPlayer()
    #p2 = CNNPlayer()
    #p2.load_weights("../saved-models/CNNPlayer-withdropout-weights.weights.h5")
    #p2.load_weights("../saved-models/CNNPlayer-withdropout-augmented-weights.weights.h5")
    #p1 = CNNPlayer_v2()
    #p1.load_weights('../saved-models/CNNPlayer-v1-withdropout-datalarge-dropout-weights.weights.h5')

    #p2 = HeuristicPlayer(heuristic_function="heuristic_defensive")
    #p2 = LookAheadHeuristicPlayer(heuristic_function="heuristic_defensive", max_depth=2)

    #game = Game(verbose=True, player_one=p1, player_two=p2)
    #game.playGame()

    #gameSession = GameSession(player_one=p1, player_two=p2, number_of_games=100)
    #gameSession.start()
    #print(gameSession.getStats())


    #p1 = CNNPlayer_v2()
    #p1.load_weights('../saved-models/ppo-v2_iter10.weights.h5')

    #p2 = HeuristicPlayer(heuristic_function="heuristic_defensive")
    #p2 = LookAheadHeuristicPlayer(heuristic_function="heuristic_defensive", max_depth=2)

    #game = Game(verbose=True, player_one=p1, player_two=p2)
    #game.playGame()

    p1r = LookAheadHeuristicPlayer(heuristic_function="heuristic_defensive", max_depth=1)

    #p1 = CNNPlayer_v2()
    #p1.load_weights("../saved-models/CNNPlayer-v1-withdropout-datalarge-dropout-weights.weights.h5")

    #cp = CNNPlayer_v4()
    #cp.load_weights("../saved-models/PPO-Model1.weights.h5")

    #p2 = LookAheadDlPlayer(max_depth=2, dl_player=cp, n_best_moves=5)

    p2 = MCTSPlayer(num_simulations=2000)

    gameSession = GameSession(player_one=p1r, player_two=p2, number_of_games=10)
    gameSession.start()
    print(gameSession.getStats())
