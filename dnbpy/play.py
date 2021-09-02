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

    minimax_depth = 3
    if "$L3" in players:
        minimax_depth = int(input("Minimax depth: "))
        if minimax_depth < 1:
            print("Error: minimax depth must be greater than 0")
            sys.exit(0)

    print("preparing game...")

    game = dnbpy.Game((board_rows, board_cols), players)
    print(game)

    computer_players = {
        "$random": dnbpy.RandomPolicy(),
        "$L1": dnbpy.Level1HeuristicPolicy(board_size=(board_rows, board_cols)),
        "$L2": dnbpy.Level2HeuristicPolicy(board_size=(board_rows, board_cols)),
        "$L3": dnbpy.Level3MinimaxPolicy(board_size=(board_rows, board_cols), depth=minimax_depth),
        "$L4": dnbpy.Level4HeuristicPolicy(board_size=(board_rows, board_cols))
    }

    while not game.is_finished():
        current_player = game.get_current_player()

        if current_player in computer_players:
            move = computer_players[current_player].select_edge(game.get_board_state())
            game.select_edge(move, current_player)
            print("computer player selects edge %s" % move)
        else:
            try:
                move = int(input("{} select your move: ".format(current_player)))
                game.select_edge(move, current_player)
            except Exception:
                print("illegal move selection.. select again")

        print(game)
