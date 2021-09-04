from .misc_functions import init_board_state
from .game import Game


def duel(board_size, player_map):
    """
    :param board_size: the board size (i.e. a 2-tuple)
    :param player_map: a dict, where the key is an int, 0 or 1, representing the player, and the value is the policy
    :return: the resulting game outcomes
    """
    board_state = init_board_state(board_size)
    results = {p: {"won": 0, "lost": 0, "tied": 0} for p in player_map}

    for player in player_map:
        for edge_index in range(len(board_state)):
            players = [player, (1 - player)]
            if edge_index % 2 == 0:
                players = [x for x in reversed(players)]
            game = Game(board_size, players)
            current_player = game.get_current_player()
            # select the first edge for the first player
            current_player, _ = game.select_edge(edge_index, current_player)
            while not game.is_finished():
                state = game.get_board_state()
                edge = player_map[current_player].select_edge(
                    state, game.get_score(current_player), game.get_score(1 - current_player))
                current_player, _ = game.select_edge(edge, current_player)

            p0_score = game.get_score(0)
            p1_score = game.get_score(1)

            if p0_score > p1_score:
                results[0]["won"] += 1
                results[1]["lost"] += 1
            if p1_score > p0_score:
                results[1]["won"] += 1
                results[0]["lost"] += 1
            if p0_score == p1_score:
                results[0]["tied"] += 1
                results[1]["tied"] += 1

    return results
