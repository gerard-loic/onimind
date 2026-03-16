from constants import PLAYER_ONE_POSITION, PLAYER_TWO_POSITION, COLOR_BLUE, COLOR_RED
import numpy as np
import random
from collections import namedtuple

# Représente un mouvement possible réalisable à partir d'une carte
# relative_move:(x:int,y:int) : mouvement relatif en x/y à partir du point de départ
# card_idx:int : identifiant unique de la carte
# move_idx:int : identifiant unique du mouvement
Move = namedtuple('Move', ['relative_move', 'card_idx', 'move_idx'])

# Représente une carte de mouvement
class Card:
    #Méthodes statiques
    #------------------------------------------------------------------------------------------------------------------------------------

    # Retourne un nombre de carte aléatoirement choisi
    @staticmethod
    def getCards(nb:int=None):
        global CARDS

        if nb is None:
            return CARDS
        else:
            return random.sample(CARDS, nb)
    
    # Retourne une carte en fonction de son identifiant unique
    @staticmethod
    def getCard(card_idx:int):
        global CARDS
        return CARDS[card_idx]
    
    # Retourne les mouvements d'une carte en fonction de son identifiant unique
    @staticmethod
    def getMoves(card_idx:int=None):
        global CARDS
        if not hasattr(Card, 'moves'):
            Card.moves = []
            for card in  CARDS:
                for m in card.relative_moves:
                    x, y, move_idx = m
                    Card.moves.append(Move((x, y), card.idx, move_idx))

        if card_idx is None:
            return Card.moves
        else:
            card_moves = []
            for move in Card.moves:
                if move.card_idx == card_idx:
                    card_moves.append(move)
            return card_moves
    
    # Retourne un mouvement en fonction de son idenfiant unique
    @staticmethod
    def getMove(move_idx:int):
        return Card.moves[move_idx]
    
    # Retourne une carte en fonction de l'IDX du mouvement
    def getCardFromMove(move_idx:int):
        for card in CARDS:
            for _, _, idx in card.relative_moves:
                if idx == move_idx:
                    return Card.getCard(card_idx=card.idx)
        return None

    

    #------------------------------------------------------------------------------------------------------------------------------------

    #Constructeur
    # idx:int : identifiant numétique unique
    # name:str : nom de la carte
    # relative_moves (x:int, y:int, move_idx:int) : mouvements
    # color:int : COLOR_RED ou COLOR_BLUE (couleur de la carte)
    def __init__(self, idx:int, name:str, relative_moves:tuple, color:int):
        self.name = name
        self.idx = idx
        self.relative_moves = relative_moves
        self.color = color
        self.print_value = self._calcPrint()
        self.opponent_print_value = self._calcPrint(position=PLAYER_TWO_POSITION)

    # Retourne les mouvements possibles [(nex_x:int, new_y:int, move_idx:int)] depuis une position transmise
    def get_moves_from_position(self, position:tuple, from_player_point_of_view:int=1):
        """
        Retourne les positions absolues accessibles depuis from_pos
        
        from_pos: (x, y) position de départ
        player: 0 ou 1 (car les mouvements sont inversés pour joueur 2)
        """
        x, y = position
        valid_moves = []
        
        for dx, dy, move_idx in self.relative_moves:
            # Inverser les mouvements pour le joueur 2
            if from_player_point_of_view == PLAYER_TWO_POSITION:
                dx, dy = -dx, -dy
            
            new_x, new_y = x + dx, y + dy
            
            # Vérifier que c'est dans le plateau
            if 0 <= new_x < 5 and 0 <= new_y < 5:
                valid_moves.append((new_x, new_y, move_idx))
        
        return valid_moves
    
    # crée une représentation matricielle 5x5 des mouvements d'une carte (utilisée pour la représentation d'un état pour les r&seaux de neurones)
    def getMatrix(self):
        matrix = np.zeros((5, 5))
        for dx, dy, _ in self.relative_moves:
            matrix[2+dx][2+dy] = 1
        return matrix

    # Calcule la représentation visuelle de la carte
    def _calcPrint(self, position:int=PLAYER_ONE_POSITION):
        """
        Affiche visuellement les mouvements possibles
        O = case de départ
        X = cases d'arrivée possibles
        - = cases vides
        """
        moves = []
        for mx, my, _ in self.relative_moves:
            if position == PLAYER_ONE_POSITION:
                moves.append((mx, my))
            else:
                moves.append((mx*-1, my*-1))

        
        # Déterminer les bornes de la grille
        min_x = min(dx for dx, dy in moves + [(0, 0)])
        max_x = max(dx for dx, dy in moves + [(0, 0)])
        min_y = min(dy for dx, dy in moves + [(0, 0)])
        max_y = max(dy for dx, dy in moves + [(0, 0)])
        
        # Créer la grille
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = [['-' for _ in range(width)] for _ in range(height)]
        
        # Position de référence (origine dans la grille)
        origin_x = -min_x
        origin_y = -min_y
        
        # Placer l'origine
        grid[origin_y][origin_x] = 'O'
        
        # Placer les mouvements possibles
        for dx, dy in moves:
            grid_x = origin_x + dx
            grid_y = origin_y + dy
            grid[grid_y][grid_x] = 'X'
        
        # Formater l'affichage
        result = f"{self.name}\n"
        for row in grid:
            result += "  ".join(row) + "\n"
        
        return result.strip()

    # Pour affichage
    def __str__(self):
        return self.print_value



CARDS = [
    Card(idx=0, name="Tiger", relative_moves=[(0, -2, 0), (0, 1, 1)], color=COLOR_BLUE),
    Card(idx=1, name="Crab", relative_moves=[(-2, 0, 2), (0, -1, 3), (2, 0, 4)], color=COLOR_BLUE),
    Card(idx=2, name="Monkey", relative_moves=[(-1, -1, 5), (1, -1, 6), (-1, 1, 7), (1, 1, 8)], color=COLOR_BLUE),
    Card(idx=3, name="Crane", relative_moves=[(0, -1, 9), (-1, 1, 10), (1, 1, 11)], color=COLOR_BLUE),
    Card(idx=4, name="Dragon", relative_moves=[(-2, -1, 12), (2, -1, 13), (-1, 1, 14), (1, 1, 15)], color=COLOR_RED),
    Card(idx=5, name="Elephant", relative_moves=[(-1, 0, 16), (1, 0, 17), (-1, -1, 18), (1, -1, 19)], color=COLOR_RED),
    Card(idx=6, name="Mantis", relative_moves=[(-1, -1, 20), (1, -1, 21), (0, 1, 22)], color=COLOR_RED),
    Card(idx=7, name="Boar", relative_moves=[(-1, 0, 23), (0, -1, 24), (1, 0, 25)], color=COLOR_RED),
    Card(idx=8, name="Frog", relative_moves=[(-2, 0, 26), (-1, -1, 27), (1, 1, 28)], color=COLOR_RED),
    Card(idx=9, name="Goose", relative_moves=[(-1, 0, 29), (-1, -1, 30), (1, 0, 31), (1, 1, 32)], color=COLOR_BLUE),
    Card(idx=10, name="Horse", relative_moves=[(-1, 0, 33), (0, -1, 34), (0, 1, 35)], color=COLOR_RED),
    Card(idx=11, name="Eel", relative_moves=[(-1, -1, 36), (1, 0, 37), (-1, 1, 38)], color=COLOR_BLUE),
    Card(idx=12, name="Rabbit", relative_moves=[(-1, 1, 39), (2, 0, 40), (1, -1, 41)], color=COLOR_BLUE),
    Card(idx=13, name="Rooster", relative_moves=[(1, -1, 42), (-1, 0, 43), (1, 0, 44), (-1, 1, 45)], color=COLOR_RED),
    Card(idx=14, name="Ox", relative_moves=[(0, -1, 46), (1, 0, 47), (0, 1, 48)], color=COLOR_BLUE),
    Card(idx=15, name="Cobra", relative_moves=[(-1, 0, 49), (1, -1, 50), (1, 1, 51)], color=COLOR_RED),
]