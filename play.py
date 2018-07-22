import sys

import dnbpy
import ai
from util.initializer_util import read_params


def main():
    print("DNBPy - Play")
    num_players = int(input("How many players?: "))

    if num_players < 2:
        print("Error: there must be at least two players")
        sys.exit(0)

    players = []
    for n in range(num_players):
        player = input("player {} name: ".format(n + 1))
        players.append(player)

    board_rows = int(input("Number of board rows: "))
    if board_rows < 1:
        print("Error: there must be at least one row")
        sys.exit(0)

    board_cols = int(input("Number of board columns: "))
    if board_cols < 1:
        print("Error: there must be at least one column")
        sys.exit(0)

    print("preparing game...")

    td0 = ai.TDZeroPolicy((board_rows, board_cols), learning_rate=0.0, gamma=0.0,
                          table_file_path='resources/td0_2x2_0.28_0.01_0.99_immediate_reward.txt')

    td1 = ai.TDOneTabularPolicy((board_rows, board_cols), learning_rate=0.0, gamma=0.0,
                                table_file_path='resources/td1_2x2_0.6_1.0_0.99_delayed_selfplay100k.txt')

    L1 = ai.Level1HeuristicPolicy((board_rows, board_cols))

    L2 = ai.Level2HeuristicPolicy((board_rows, board_cols))

    pg_params = read_params('resources/pg_2x2_cnn2_vs_L0_L1_L2_batch_01-episode-1496000.txt')
    pg_model = ai.PGPolicyCNN2((board_rows, board_cols), existing_params=pg_params)
    pg_model.set_boltzmann_action(False)
    pg_model.set_epsilon(0.0)

    MCTS_PG = ai.MCTSPolicy2((board_rows, board_cols), num_playouts=1000, default_policy=pg_model)

    game = dnbpy.Game((board_rows, board_cols), players)
    print(game)
    while not game.is_finished():
        current_player = game.get_current_player()

        if current_player == "$L0":
            move = dnbpy.RandomPolicy().select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$TD0":
            move = td0.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$TD1":
            move = td1.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$L1":
            move = L1.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$L2":
            move = L2.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$PG":
            move = pg_model.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$MCTS-PG":
            move = MCTS_PG.select_edge(game.get_board_state(), game.get_score(current_player))
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        else:
            try:
                move = int(input("{} select your move: ".format(current_player)))
                current_player, _ = game.select_edge(move, current_player)
            except Exception:
                print("illegal move selection.. select again")

        print(game)

if __name__ == "__main__":
    main()