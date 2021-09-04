import sys

import dnbpy


def play():
    print("DNBPy - Play")
    num_players = int(input("How many players?: "))

    if num_players < 2:
        print("Error: there must be at least two players")
        sys.exit(0)

    players = []
    for n in range(num_players):
        player = input("player {} name: ".format(n + 1))
        players.append(player)

    if len(set(players)) == 1:
        print("Error: player names must be unique")
        sys.exit(0)

    board_rows = int(input("Number of board rows: "))
    if board_rows < 1:
        print("Error: there must be at least one row")
        sys.exit(0)

    board_cols = int(input("Number of board columns: "))
    if board_cols < 1:
        print("Error: there must be at least one column")
        sys.exit(0)

    minimax_depth = None
    if "$L3" in players:
        val = input("Minimax depth (leave empty for variable depth): ")
        if len(val.strip()) > 0:
            minimax_depth = int(val)
            if minimax_depth < 1:
                print("Error: minimax depth must be greater than 0")
                sys.exit(0)

    num_playouts = 100
    if "$mcts" in players:
        num_playouts = int(input("Number of playouts: "))
        if num_playouts < 1:
            print("Error: number of playouts must be greater than 0")
            sys.exit(0)

    print("preparing game...")

    board_size = (board_rows, board_cols)
    game = dnbpy.Game(board_size, players)
    print(game)

    computer_players = {
        "$random": dnbpy.RandomPolicy(),
        "$L1": dnbpy.Level1HeuristicPolicy(board_size=board_size),
        "$L2": dnbpy.Level2HeuristicPolicy(board_size=board_size),
        "$L3": dnbpy.Level3MinimaxPolicy(board_size=board_size, depth=minimax_depth, update_alpha=True),
        "$mcts": dnbpy.MCTSPolicy(board_size=board_size, num_playouts=num_playouts)
    }

    while not game.is_finished():
        current_player = game.get_current_player()

        if current_player in computer_players:
            # get the first player that isn't the current player
            opp_player = [p for p in players if p != current_player][0]
            move = computer_players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(move, current_player)
            print("player %s selects edge %s" % (current_player, move))
        else:
            try:
                move = int(input("{} select your move: ".format(current_player)))
                game.select_edge(move, current_player)
            except Exception:
                print("illegal move selection.. select again")

        print(game)
