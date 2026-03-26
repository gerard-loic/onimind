from sessionmemory import SessionMemory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../onitama/'))
from game import Game
from card import Card
from players import HeuristicPlayer, ApiPlayer, Player, RandomPlayer, LookAheadHeuristicPlayer
from constants import *
from exceptions import InvalidPlayerException, GameEndedException, InvalidSessionException, PlayerNotFoundException

players = {
    'heuristic_regular' : { 'class' : HeuristicPlayer(heuristic_function='heuristic_regular'), 'name' : 'Joueur heuristique (regular)'},
    'random' : { 'class' : RandomPlayer(), 'name' : 'Joueur random'},
    'heuristic_3lookahead_regular' : { 'class' : LookAheadHeuristicPlayer(max_depth=3, heuristic_function='heuristic_regular'), 'name' : 'Joueur 3 LookAhead heuristique (regular)'},
}

class GameManager:
    @staticmethod
    def create(player:str):
        #Initialisation de P1 (Joueur)
        pj = ApiPlayer()

        #Initialisation de P2 (Machine)
        if player in players:
            pm = players[player]['class']
        else:
            raise PlayerNotFoundException(player=player)

        game = Game(player_one=pj, player_two=pm, verbose=False)
            
        uid = SessionMemory.createSession(data={'game' : game, 'turn_num' : 1, 'pj' : pj, 'pm' : pm})
        gm = GameManager(uid=uid)

        return gm


    def __init__(self, uid:str):
        session = SessionMemory.getSession(sessionId=uid)
        if session is None:
            raise InvalidSessionException(session=uid)
        self.uid = uid
        self.game = session['game']
        self.turn_num = session['turn_num']
        self.pj = session['pj']
        self.pm = session['pm']
        self.last_action = None

    def player_play(self, from_pos:tuple, to_pos:tuple, card_idx:int):
        if self.game.current_player == self.pj:
            ended, winner = self.game.board.game_has_ended()
            if ended:
                raise GameEndedException()
            self.pj.set_next_move(from_pos=from_pos, to_pos=to_pos, card_idx=card_idx)
            self.last_action = self.game.playGame(return_winner=False, max_turns=200, play_once_only=True, return_move=True)
            self.turn_num += 1
            self.save()
        else:
            raise InvalidPlayerException("HUMAN")

    def opponent_play(self):
        if self.game.current_player == self.pm:
            ended, winner = self.game.board.game_has_ended()
            if ended:
                raise GameEndedException()
            self.last_action = self.game.playGame(return_winner=False, max_turns=200, play_once_only=True, return_move=True)
            self.turn_num += 1
            self.save()
        else:
            raise InvalidPlayerException("IA")
    
    def save(self):
        SessionMemory.updateSession(sessionId=self.uid, data={'game' : self.game, 'turn_num' : self.turn_num, 'pj' : self.pj, 'pm' : self.pm})

    def get_game_representation(self):
        ended, winner = self.game.board.game_has_ended()
        if ended:
            if winner == self.pj.position:
                winner = "HUMAN"
            else:
                winner = "IA"

        action = self.last_action
        if action is None:
            action_data = None
        else:
            action_card = Card.getCardFromMove(move_idx=action.move_idx)

            action_data = {
                'from_pos' : {
                    'col' : action.from_pos[0],
                    'row' : action.from_pos[1]
                },
                'to_pos' : {
                    'col' : action.to_pos[0],
                    'row' : action.to_pos[1]
                },
                'card_idx' : action_card.idx
            }

        return {
            'game_uid' : self.uid,
            'turn_num' : self.turn_num,
            'current_player' : "HUMAN" if self.game.board.current_player == self.pj.position else "IA",
            'player_cards' : self.game.board.current_player_cards,
            'opponent_cards' : self.game.board.next_player_cards,
            'neutral_card' :self.game.board.neutral_card,
            'board' : self.game.board.board,
            'ended' : ended,
            'winner' : winner,
            'last_move' : action_data
        }
