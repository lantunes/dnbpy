import sys

import tensorflow as tf

import ai
import dnbpy
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

    L1 = ai.Level1HeuristicPolicy((board_rows, board_cols))
    L2 = ai.Level2HeuristicPolicy((board_rows, board_cols))

    CEP = ai.CausalEntropicPolicy((board_rows, board_cols), max_sample_paths=1000)

    if board_rows == 2 and board_cols == 2:
        td0 = ai.TDZeroPolicy((board_rows, board_cols), learning_rate=0.0, gamma=0.0,
                              table_file_path='resources/td0_2x2_0.28_0.01_0.99_immediate_reward.txt')

        td1 = ai.TDOneTabularPolicy((board_rows, board_cols), learning_rate=0.0, gamma=0.0,
                                    table_file_path='resources/td1_2x2_0.6_1.0_0.99_delayed_selfplay100k.txt')

        # pg_params = read_params('resources/pg_2x2_cnn2_tanh_mcts_exit_03-episode-912000.txt')
        pg_params = read_params('resources/pg_2x2_cnn2_tanh_dnbpy29-episode-427000.txt')
        pg_model = ai.PGPolicyCNN2((board_rows, board_cols), existing_params=pg_params, activation=tf.nn.tanh)
        pg_model.set_boltzmann_action(False)
        pg_model.set_epsilon(0.0)

        # MCTS_PG = ai.MCTSPolicyNetPolicyCpuct((board_rows, board_cols), num_playouts=1000, cpuct=5, default_policy=pg_model)
        MCTS_PG = ai.MCTSPolicy2((board_rows, board_cols), num_playouts=10000, default_policy=pg_model)
    else:
        td0 = None
        td1 = None
        pg_model = None
        MCTS_PG = None

    if board_rows == 3 and board_cols == 3:
        # pg_params2 = read_params('resources/dnbpy43-3x3-tanh-305000.txt')
        # OPP = ai.PGPolicy3x3CNN((board_rows, board_cols), existing_params=pg_params2, activation=tf.nn.tanh)
        # OPP.set_boltzmann_action(False)
        # OPP.set_epsilon(0.0)
        # MCTS_PG2 = ai.MCTSPolicyNetPolicy((board_rows, board_cols), num_playouts=1000, w=100, default_policy=OPP)

        # pg_params3 = read_params('out/dnbpy48/search-weights-433000.txt')
        pg_params3 = read_params('out/dnbpy57/weights-139.txt')
        OPP = ai.PGPolicy3x3CNN((board_rows, board_cols), existing_params=pg_params3, activation=tf.nn.relu)
        OPP.set_boltzmann_action(False)
        OPP.set_epsilon(0.0)
        MCTS_PG2 = ai.MCTSPolicyNetPolicyCpuct((board_rows, board_cols), num_playouts=1000, cpuct=5,
                                               normalize_policy_probs_with_softmax=False)
    else:
        OPP = None
        MCTS_PG2 = None

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
            # move = MCTS_PG.select_edge(game.get_board_state(), game.get_score(current_player), pg_model)
            move = MCTS_PG.select_edge(game.get_board_state(), game.get_score(current_player))
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$OPP":
            move = OPP.select_edge(game.get_board_state())
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$MCTS-PG2":
            move = MCTS_PG2.select_edge(game.get_board_state(), game.get_score(current_player), OPP)
            current_player, _ = game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        elif current_player == "$CEP":
            move = CEP.select_edge(game.get_board_state())
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