from dnbpy import *


def duel(board_size, p1, p2, pol):
    """
    :param board_size: the game's board size 
    :param p1: the first player
    :param p2: the second player
    :return: the games won, lost and tied, from the perspective of p1
    """
    results = {'won': 0, 'lost': 0, 'tied': 0}
    board_state = init_board_state(board_size)
    for player in range(2):

        for edge_index in range(len(board_state)):
            players = [player, (1 - player)]
            if edge_index % 2 == 0:
                players = [x for x in reversed(players)]
            game = Game(board_size, players)
            current_player = game.get_current_player()
            # select the first edge for the first player
            board_state = game.get_board_state()
            current_player, _ = game.select_edge(edge_index, current_player)
            while not game.is_finished():
                board_state = game.get_board_state()
                if current_player == 0:
                    edge = p1.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, current_player)
                else:
                    edge = p2.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, current_player)
            p1_score = game.get_score(0)
            p2_score = game.get_score(1)
            if p1_score > p2_score:
                results['won'] += 1
            elif p2_score > p1_score:
                results['lost'] += 1
            else:
                results['tied'] += 1
            print(results)
    return results


def tournament(board_size, policy, opponent_pool):
    opponents = opponent_pool.get_opponents()
    tournament_results = {'won': 0, 'lost': 0, 'tied': 0}
    for opponent in opponents:
        results = duel(board_size, policy, opponent)
        tournament_results['won'] += results['won']
        tournament_results['lost'] += results['lost']
        tournament_results['tied'] += results['tied']
    return tournament_results
