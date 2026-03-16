import numpy as np
from card import Card
from collections import namedtuple
from constants import *
import random

# Décrit une action, c'est à dire : 
# - from_pos () : coordonnées de la case à partir de laquelle la pièce est déplacée
# - card_idx [int] : identifiant unique de la carte jouée (se réfère à la variable CARDS dans card.py)
# - to_pos () : coordonnées de la case de destination
# - move_idx [int] : identifiant unique du mouvement effectué (se réfère à la variable CARDS dans card.py)
Action = namedtuple('Action', ['from_pos', 'card_idx', 'to_pos', 'move_idx'])

# Classe permettant de décrire un état
class Board:
    #Méthodes statiques
    #------------------------------------------------------------------------------------------------------------------------------------

    # Traduit une valeur de cellule en une représentation "humainement compréhensible" (ex. 1 en x) Sert pour afficher l'état dans la console
    @staticmethod
    def cell_to_string(cell:int):
        return STR_MAPPING[cell]
    
    # Permet de retourner le plateau. (Sert pour inverser la représentation du plateau entre le joueur courant et le joueur suivant. )
    # On considère que le joueur courant est toujours FACE au plateau, c'est à dire en "bas"
    @staticmethod
    def rotate_180(board:list):
        # Inverse l'ordre des colonnes ET l'ordre dans chaque colonne
        return [col[::-1] for col in board[::-1]]


    #------------------------------------------------------------------------------------------------------------------------------------

    # Constructeur
    # player_to_move:int : le joueur dont c'est le tour (c'est à dire PLAYER_ONE_POSITION ou PLAYER_TWO_POSITION)
    # current_player_cards:list(Card) : cartes du joueur dont c'est le tour
    # next_player_cards:list(Card) : cartes du joueur adverse
    # neutral_card:Card : carte neutre
    # board:list : pour forcer un état de plateau(optionnel)
    def __init__(self, player_to_move:int, current_player_cards:list, next_player_cards:list, neutral_card:Card, board:list=None):
        self.first_player = player_to_move
        self.current_player = player_to_move
        self.next_player = PLAYER_TWO_POSITION if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_POSITION
        self.current_player_cards = current_player_cards
        self.next_player_cards = next_player_cards
        self.neutral_card = neutral_card

        # Codes des pièces
        self.current_player_student = PLAYER_ONE_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        self.current_player_master = PLAYER_ONE_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        self.next_player_student = PLAYER_TWO_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        self.next_player_master = PLAYER_TWO_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER

        # Position des temples
        self.current_player_temple = (2, 4)
        self.next_player_temple = (2, 0)

        # 5 colonnes sur 5 lignes
        self.board = [[EMPTY_CELL] * 5 for _ in range(5)] if board is None else board


    # Retourne les mouvements possibles, du point de vue de l'utilisateur courant (retourne un tableau d'actions)
    def get_available_moves(self):
        moves = []
        #Quelles sont les pieces qui nous interessent ?
        pieces = [self.current_player_student, self.current_player_master]

        #On parcourt le plateau à la recherche des pièces
        for col in range(5):
            for row in range(5):
                #On vérifie que c'est une pièce du joueur
                if self.board[col][row] in pieces:
                    #On créé les actions pour chaque carte
                    for card in self.current_player_cards:
                        cardMoves = card.get_moves_from_position(position=(col, row))
                        for moveCol, moveRow, move_idx in cardMoves:
                            #On vérifie qu'on a pas déjà une pièce sur la case
                            if not self.board[moveCol][moveRow] in pieces:
                                #On ajoute l'actions
                                moves.append(Action((col, row), card.idx, (moveCol, moveRow), move_idx))
        return moves

    
    # Joue un coup et retourne sa représentation
    # action:Action : l'action jouée
    def play_move(self, action:Action):
        last_move = {
            'board_to_resume' : [],
            'neutral_card' : None,
            'current_player_cards' : []
        }

        # Position de départ
        from_x, from_y = action.from_pos

        # Position d'arrivée
        to_x, to_y = action.to_pos

        current_player_pieces = [self.current_player_student, self.current_player_master]
        next_player_pieces = [self.next_player_student, self.next_player_master]
        piece_to_move = self.board[from_x][from_y]

        # Vérifier que le coup est valide (la case de destination ne doi pas être occupée par le joueur courant)
        assert self.board[to_x][to_y] not in current_player_pieces, f"The cell {to_x, to_y} is not valid : current player already there. "
        
        # On garde en mémoire pour pouvoir annuler le coup
        last_move['board_to_resume'].append((from_x, from_y, self.board[from_x][from_y]))
        last_move['board_to_resume'].append((to_x, to_y, self.board[to_x][to_y]))
        last_move['neutral_card'] = self.neutral_card
        for card in self.current_player_cards:
            last_move['current_player_cards'].append(card)

        # On place la pièce sur la nouvelle case
        self.board[to_x][to_y] = piece_to_move

        # On vide la case d'origine
        self.board[from_x][from_y] = EMPTY_CELL

        # On fait tourner la carte utilisée (elle va aller en neutre)
        for i in range(len(self.current_player_cards)):
            if self.current_player_cards[i].idx == action.card_idx:
                break
        self.current_player_cards.append(self.neutral_card)
        self.neutral_card = self.current_player_cards.pop(i)

        # On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        # On retourne le plateau pour s'adapter au prochain joueur
        self.board = Board.rotate_180(board=self.board)

        # On retourne la description du coup (permettra de l'annuler)
        return last_move

    # Cas particulier : si aucun coup n'est possible, il faut quand même choisir une carte qui deviendra la carte neutre
    def play_default_move(self):
        # On mélange pour aléatoirement choisir une carte qui sera donnée à l'adversaire
        random.shuffle(self.current_player_cards)

        # On garde la référence à l'ancienne carte neutre et aux cartes courantes de l'utilisateur
        old_neutral = self.neutral_card
        old_current_cards = self.current_player_cards.copy()

        # On ajoute aux cartes du joueur courant la carte neutre
        self.current_player_cards.append(self.neutral_card)

        # On prend la première carte du joueur courant et on la sort, elle devient la carte neutre
        self.neutral_card = self.current_player_cards.pop(0)

        # On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

         #On retourne le plateau
        self.board = Board.rotate_180(board=self.board)

        # On retourne la représentation du coup joué
        return {'neutral_card': old_neutral, 'current_player_cards': old_current_cards}


    # Permet d'annuler un coup "défaut", pour ce faire il faut lui donner la représentation du coup joué
    def cancel_default_move(self, last_move:dict=None):
        # On retourne le plateau
        self.board = Board.rotate_180(board=self.board)

        # On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        if last_move:
            self.neutral_card = last_move['neutral_card']
            self.current_player_cards = last_move['current_player_cards']

    
    # Annule un coup "classique", pour ce faire il faut fournir la représentation du coup joué
    def cancel_last_move(self, last_move:dict):
        if last_move is None:
            print("Nothing to cancel !")
            return
        
        # On retourne le plateau
        self.board = Board.rotate_180(board=self.board)
        
        # On réinitialise le board
        for col, row, pawn in last_move['board_to_resume']:
            self.board[col][row] = pawn

        # On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        # On rétablit les cartes
        self.neutral_card = last_move['neutral_card']
        self.current_player_cards = last_move['current_player_cards']


    # Retourne si le jeu est terminé, et si il y a un gagnant
    def game_has_ended(self):

        # Si le maitre du joueur courant est sur le temple de l'opposant
        if self.board[self.next_player_temple[0]][self.next_player_temple[1]] == self.current_player_master:
            return True, self.current_player
        
        # Si le maitre de l'opposant est sur le temple du joueur courant
        if self.board[self.current_player_temple[0]][self.current_player_temple[1]] == self.next_player_master:
            return True, self.next_player
        
        # Si le joueur courant n'a plus son maitre
        if not any(self.current_player_master in ligne for ligne in self.board):
            return True, self.next_player
        
        # Si l'opposant n'a plus son maitre
        if not any(self.next_player_master in ligne for ligne in self.board):
            return True, self.current_player

        return False, None


    # Retourne la position (x,y) du maitre
    def get_master_position(self, player_position:int):
        code = PLAYER_ONE_MASTER if player_position == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        for col in range(5):
            for row in range(5):
                if self.board[col][row] == code:
                    return (col, row)
        return (None, None)


    # Retourne la matrice représentant l'état du plateau pour le réseau de neurones
    def get_state(self):
        """
        Permet de convertir l'état en une matrice 5x5x10, plateau tourné du point de vue du joueur courant
        Plan 0 : positions des pions du joueur courant (0/1)
        Plan 1 : position du maître du joueur courant (0/1)
        Plan 2 : positions des pions de l'adversaire (0/1)
        Plan 3 : position du maître de l'adversaire (0/1)
        Plan 4 : Mouvements de la carte 1 du joueur courant (0/1) - pièce au centre en (2,2)
        Plan 5 : Mouvements de la carte 2 du joueur courant (0/1) - pièce au centre en (2,2)
        Plan 6 : Mouvements de la carte 1 de l'adversaire (0/1) - pièce au centre en (2,2)
        Plan 7 : Mouvements de la carte 2 de l'adversaire (0/1) - pièce au centre en (2,2)
        Plan 8 : Mouvements de la carte neutre (0/1) _ pièce au centre en (2,2)
        Plan 9 : matrix de 1 si le joueur courant est le premier joueur, -1 sinon
        """

        #Variables adaptées au point de vue
        student_current = PLAYER_ONE_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        student_next = PLAYER_TWO_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        master_current = PLAYER_ONE_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        master_next = PLAYER_TWO_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER

        channels = []
        board_array = np.array(self.board)

        #Plan 0 : positions des pions du joueur courant (0/1)
        channels.append((board_array == student_current).astype(float)) 

        #Plan 1 : position du maître du joueur courant (0/1)
        channels.append((board_array == master_current).astype(float))

        #Plan 2 : positions des pions de l'adversaire (0/1)
        channels.append((board_array == student_next).astype(float))

        #Plan 3 : position du maître de l'adversaire (0/1)
        channels.append((board_array == master_next).astype(float)) 

        #Plan 4
        channels.append(self.current_player_cards[0].getMatrix())

        #Plan 5
        channels.append(self.current_player_cards[1].getMatrix())

        #Plan 6
        channels.append(self.next_player_cards[0].getMatrix())

        #Plan 7
        channels.append(self.next_player_cards[1].getMatrix())

        #Plan 8
        channels.append(self.neutral_card.getMatrix())

        #Plan 9
        if self.first_player == self.current_player:
            channels.append(np.ones((5,5)))
        else:
            channels.append(-np.ones((5,5)))

        return channels

    #------------------------------------------------------------------------------------------------------------------------------------

    #Surcharge print(instance)
    def __str__(self):
        res_str = '**********************************************************************\n'
        if self.current_player == PLAYER_ONE_POSITION:
            res_str += f"Current player : X\n\n"
        else:
            res_str += "Current player : O\n\n"
        res_str = res_str+'\n'.join(
            f"{ row + 1} | " + ' '.join(self.cell_to_string(self.board[col][row]) for col in range(5))
            for row in range(5)
        )
        letters = [chr(i) for i in range(ord('A'), ord('A') + 5)]
        res_str += '\n'
        res_str += '  | ' + ' '.join(letters) + '\n'

        res_str += "\nNext player cards: "
        for card in self.next_player_cards:
            res_str += card.name+" "

        res_str += f"\nNeutral card: {self.neutral_card.name}"

        res_str += "\nCurrent player cards: \n"
        for card in self.current_player_cards:
            res_str += f"{card.print_value}\n"


        return f"{res_str}"

        return res_str

    #Représentation (pour machine learning)
    def __repr__(self):
        board_repr = f"{self.current_player}"
        for col in range(5):
            board_repr += '|' + ''.join([str(plr_type) for plr_type in self.board[col]])
        return board_repr



    