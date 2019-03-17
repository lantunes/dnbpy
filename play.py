import sys

import dnbpy


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

    game = dnbpy.Game((board_rows, board_cols), players)
    print(game)
    while not game.is_finished():
        current_player = game.get_current_player()

        if current_player == "$random":
            move = dnbpy.RandomPolicy().select_edge(game.get_board_state())
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
