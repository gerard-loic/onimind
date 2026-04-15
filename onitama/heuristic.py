from board import Board
from constants import *

class HeuristicEvaluation:

    
    # Retourne une évaluation heuristique de l'état sur une base "équilibrée"
    def heuristic_regular(board:Board, from_current_player_point_of_view:bool=True, verbose:bool=False):
        ctx = HeuristicEvaluation._get_evaluation_context(board, from_current_player_point_of_view)

        #Si la partie est perdue
        game_ended, winner = board.game_has_ended()
        if game_ended:
            if winner == ctx['p1']:
                if verbose:
                    print(f"Score for winning : 1000")
                return 1000
            else:
                if verbose:
                    print(f"Score for lossing : -1000")
                return -1000

        #Différence en nombre d'étudiant
        p1_nb_students = sum(valeur == ctx['p1_student'] for ligne in board.board for valeur in ligne)
        p2_nb_students = sum(valeur == ctx['p2_student'] for ligne in board.board for valeur in ligne)
        score_diff_students = (p1_nb_students-p2_nb_students)*100

        #Distance du maitre de P1 par rapport au temple de P2 (positif)
        master_col, master_row = board.get_master_position(player_position=ctx['p1'])
        temple_col, temple_row = ctx['p2_temple']
        distance = abs(master_col - temple_col) + abs(master_row - temple_row)
        score_distance_master_current_player = (8 - distance) * 20

        #Distance du maitre de P2 par rapport au temple de P1 (négtif)
        master_col, master_row = board.get_master_position(player_position=ctx['p2'])
        temple_col, temple_row = ctx['p1_temple']
        distance = abs(master_col - temple_col) + abs(master_row - temple_row)
        score_distance_master_next_player = ((8 - distance) * 20)*-1
        
        score = score_diff_students+score_distance_master_current_player+score_distance_master_next_player

        if verbose:
            print(f"Score student difference : {score_diff_students}")
            print(f"Score distance current player master from opponent temple : {score_distance_master_current_player}")
            print(f"Score distance opponent player master from current player temple : {score_distance_master_next_player}")
            print(f"Score total : {score}")

        return score

    # Heuristique agressive : priorité aux captures et au contrôle du centre.
    # - Bonus important pour capturer des pièces adverses
    # - Bonus pour contrôle du centre (cases autour de (2,2))
    # - Accepte plus facilement les sacrifices
    def heuristic_aggressive(board:Board, from_current_player_point_of_view:bool=True, verbose:bool=False):
        ctx = HeuristicEvaluation._get_evaluation_context(board, from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = board.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = HeuristicEvaluation._count_pieces(board, ctx)

        # Différence d'étudiants (bonus augmenté pour captures)
        score_captures = (len(p1_students) - len(p2_students)) * 150

        # Contrôle du centre (cases (1,1) à (3,3))
        center_control = 0
        center_squares = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
        for col, row in center_squares:
            cell = board.board[col][row]
            if cell in [ctx['p1_student'], ctx['p1_master']]:
                center_control += 15
            elif cell in [ctx['p2_student'], ctx['p2_master']]:
                center_control -= 15

        # Distance agressive du maître vers temple adverse
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_attack = (8 - distance) * 25
        else:
            score_attack = 0

        # Pénalité réduite pour maître adverse proche (on accepte le risque)
        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_defense = (8 - distance) * -10  # Pénalité réduite
        else:
            score_defense = 0

        score = score_captures + center_control + score_attack + score_defense

        if verbose:
            print(f"[Aggressive] Captures: {score_captures}, Centre: {center_control}, Attack: {score_attack}, Defense: {score_defense}, Total: {score}")

        return score

    
    # Heuristique défensive : protection du maître et structure solide.
    # - Priorité à bloquer l'adversaire
    # - Bonus pour étudiants protégeant le maître
    # - Malus important si le maître est exposé
    def heuristic_defensive(board:Board, from_current_player_point_of_view:bool=True, verbose:bool=False):
        ctx = HeuristicEvaluation._get_evaluation_context(board, from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = board.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = HeuristicEvaluation._count_pieces(board, ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Protection du maître : compter les étudiants adjacents
        protection_score = 0
        if p1_master_pos:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                adj_col, adj_row = p1_master_pos[0] + dx, p1_master_pos[1] + dy
                if 0 <= adj_col < 5 and 0 <= adj_row < 5:
                    if board.board[adj_col][adj_row] == ctx['p1_student']:
                        protection_score += 25

        # Malus si maître exposé (peu de protection)
        if p1_master_pos and protection_score < 25:
            protection_score -= 30

        # Distance du maître adverse par rapport à notre temple (très importante)
        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_threat = (8 - distance) * -30  # Pénalité forte
        else:
            score_threat = 0

        # Notre maître vers temple adverse (importance réduite)
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_advance = (8 - distance) * 10  # Bonus réduit
        else:
            score_advance = 0

        # Bonus pour garder le maître près de notre temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance_home = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_home = (4 - distance_home) * 10 if distance_home <= 4 else 0
        else:
            score_home = 0

        score = score_students + protection_score + score_threat + score_advance + score_home

        if verbose:
            print(f"[Defensive] Students: {score_students}, Protection: {protection_score}, Threat: {score_threat}, Advance: {score_advance}, Home: {score_home}, Total: {score}")

        return score
    
    
    # Heuristique basée sur la mobilité et le contrôle de l'espace.
    # - Nombre de coups légaux disponibles
    # - Contrôle des cases centrales
    #  - Avancement des pièces
    def heuristic_mobility(board:Board, from_current_player_point_of_view:bool=True, verbose:bool=False):
        ctx = HeuristicEvaluation._get_evaluation_context(board, from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = board.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = HeuristicEvaluation._count_pieces(board, ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Mobilité : nombre de coups disponibles
        num_moves = len(board.get_available_moves())
        score_mobility = num_moves * 5

        # Avancement des pièces (bonus pour pièces avancées vers l'adversaire)
        # Row 0 = notre côté, Row 4 = côté adverse
        advancement_score = 0
        for col, row in p1_students:
            advancement_score += row * 8  # Plus avancé = plus de points
        if p1_master_pos:
            advancement_score += p1_master_pos[1] * 5

        # Pénalité pour pièces adverses avancées
        for col, row in p2_students:
            advancement_score -= (4 - row) * 8
        if p2_master_pos:
            advancement_score -= (4 - p2_master_pos[1]) * 5

        # Distance maître vers temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_master = (8 - distance) * 15
        else:
            score_master = 0

        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_master -= (8 - distance) * 15

        score = score_students + score_mobility + advancement_score + score_master

        if verbose:
            print(f"[Mobility] Students: {score_students}, Mobility: {score_mobility}, Advancement: {advancement_score}, Master: {score_master}, Total: {score}")

        return score

    

    # Heuristique positionnelle : structure de pions et contrôle territorial.
    # - Formation groupée des pièces
    # - Contrôle de colonnes/rangées clés
    # - Occupation des diagonales
    def heuristic_positional(board:Board, from_current_player_point_of_view:bool=True, verbose:bool=False):
        
        ctx = HeuristicEvaluation._get_evaluation_context(board, from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = board.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = HeuristicEvaluation._count_pieces(board, ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Cohésion des pièces : bonus si les pièces sont proches les unes des autres
        cohesion_score = 0
        p1_pieces = p1_students + ([p1_master_pos] if p1_master_pos else [])
        for i, pos1 in enumerate(p1_pieces):
            for pos2 in p1_pieces[i+1:]:
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if distance <= 2:
                    cohesion_score += 10
                elif distance <= 3:
                    cohesion_score += 5

        # Contrôle de la colonne centrale (colonne 2)
        central_column = 0
        for row in range(5):
            cell = board.board[2][row]
            if cell in [ctx['p1_student'], ctx['p1_master']]:
                central_column += 12
            elif cell in [ctx['p2_student'], ctx['p2_master']]:
                central_column -= 12

        # Contrôle des rangées avancées
        territory_score = 0
        for col in range(5):
            for row in range(3, 5):  # Rangées 3 et 4 (côté adverse)
                cell = board.board[col][row]
                if cell in [ctx['p1_student'], ctx['p1_master']]:
                    territory_score += 15
            for row in range(0, 2):  # Rangées 0 et 1 (notre côté)
                cell = board.board[col][row]
                if cell in [ctx['p2_student'], ctx['p2_master']]:
                    territory_score -= 15

        # Distance maître vers temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_master = (8 - distance) * 15
        else:
            score_master = 0

        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_master -= (8 - distance) * 15

        score = score_students + cohesion_score + central_column + territory_score + score_master

        if verbose:
            print(f"[Positional] Students: {score_students}, Cohesion: {cohesion_score}, Central: {central_column}, Territory: {territory_score}, Master: {score_master}, Total: {score}")

        return score
    

    #Méthodes "helpers"
    #------------------------------------------------------------------------------------------------------------------------------------

    # Retourne le contexte d'évaluation (joueurs, pièces, temples) selon le point de vue.
    # Utilisé par toutes les variantes d'heuristiques.
    def _get_evaluation_context(board:Board, from_current_player_point_of_view:bool=True):
        """
        Retourne le contexte d'évaluation (joueurs, pièces, temples) selon le point de vue.
        Utilisé par toutes les variantes d'heuristiques.
        """
        p1 = board.current_player
        p2 = PLAYER_TWO_POSITION if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_POSITION

        p1_student = PLAYER_ONE_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        p2_student = PLAYER_TWO_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        p1_master = PLAYER_ONE_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        p2_master = PLAYER_TWO_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER
        p1_temple = PLAYER_ONE_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_TEMPLE
        p2_temple = PLAYER_TWO_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_TEMPLE

        if from_current_player_point_of_view is False:
            p1, p2 = p2, p1
            p1_student, p2_student = p2_student, p1_student
            p1_master, p2_master = p2_master, p1_master
            p1_temple, p2_temple = p2_temple, p1_temple

        return {
            'p1': p1, 'p2': p2,
            'p1_student': p1_student, 'p2_student': p2_student,
            'p1_master': p1_master, 'p2_master': p2_master,
            'p1_temple': p1_temple, 'p2_temple': p2_temple
        }
    
    # Compte les pièces et retourne leurs positions en onction d'un contexte
    def _count_pieces(board:Board, ctx:list):
        """Compte les pièces et retourne leurs positions."""
        p1_students = []
        p2_students = []
        p1_master_pos = None
        p2_master_pos = None

        for col in range(5):
            for row in range(5):
                cell = board.board[col][row]
                if cell == ctx['p1_student']:
                    p1_students.append((col, row))
                elif cell == ctx['p2_student']:
                    p2_students.append((col, row))
                elif cell == ctx['p1_master']:
                    p1_master_pos = (col, row)
                elif cell == ctx['p2_master']:
                    p2_master_pos = (col, row)

        return p1_students, p2_students, p1_master_pos, p2_master_pos