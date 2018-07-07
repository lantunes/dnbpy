from ai import *

board_size = (5, 5)

# player1 = RandomPolicy()
player1 = Level1HeuristicPolicy(board_size)
# player2 = Level1HeuristicPolicy(board_size)
player2 = Level2HeuristicPolicy(board_size)

print("board size: %sx%s" % board_size)
print("player 1 (%s) vs. player 2 (%s)" % (player1.__class__.__name__, player2.__class__.__name__))

p1_results = {'won': 0, 'lost': 0, 'tied': 0}
p2_results = {'won': 0, 'lost': 0, 'tied': 0}
for trial in range(10000):
    players = ['p1', 'p2']
    if trial % 2 == 0:
        players = [x for x in reversed(players)]
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'p1':
            edge = player1.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 'p1')
        else:
            edge = player2.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 'p2')
    p1_score = game.get_score('p1')
    p2_score = game.get_score('p2')
    if p1_score > p2_score:
        p1_results['won'] += 1
        p2_results['lost'] += 1
    elif p2_score > p1_score:
        p2_results['won'] += 1
        p1_results['lost'] += 1
    else:
        p1_results['tied'] += 1
        p2_results['tied'] += 1

print ("%s: %s; %s: %s" % (player1.__class__.__name__, p1_results, player2.__class__.__name__, p2_results))

"""
Results:

board size: 2x2
player 1 (RandomPolicy) vs. player 2 (Level1HeuristicPolicy)
RandomPolicy: {'lost': 7761, 'tied': 1799, 'won': 440}; Level1HeuristicPolicy: {'lost': 440, 'tied': 1799, 'won': 7761}

board size: 2x2
player 1 (RandomPolicy) vs. player 2 (Level2HeuristicPolicy)
RandomPolicy: {'won': 312, 'tied': 1693, 'lost': 7995}; Level2HeuristicPolicy: {'won': 7995, 'tied': 1693, 'lost': 312}

board size: 2x2
player 1 (Level1HeuristicPolicy) vs. player 2 (Level2HeuristicPolicy)
Level1HeuristicPolicy: {'lost': 5408, 'tied': 3361, 'won': 1231}; Level2HeuristicPolicy: {'lost': 1231, 'tied': 3361, 'won': 5408}

board size: 3x3
player 1 (RandomPolicy) vs. player 2 (Level1HeuristicPolicy)
RandomPolicy: {'tied': 0, 'won': 31, 'lost': 9969}; Level1HeuristicPolicy: {'tied': 0, 'won': 9969, 'lost': 31}

board size: 3x3
player 1 (RandomPolicy) vs. player 2 (Level2HeuristicPolicy)
RandomPolicy: {'lost': 9966, 'won': 34, 'tied': 0}; Level2HeuristicPolicy: {'lost': 34, 'won': 9966, 'tied': 0}

board size: 3x3
player 1 (Level1HeuristicPolicy) vs. player 2 (Level2HeuristicPolicy)
Level1HeuristicPolicy: {'lost': 8409, 'won': 1591, 'tied': 0}; Level2HeuristicPolicy: {'lost': 1591, 'won': 8409, 'tied': 0}

board size: 5x5
player 1 (RandomPolicy) vs. player 2 (Level1HeuristicPolicy)
RandomPolicy: {'lost': 10000, 'won': 0, 'tied': 0}; Level1HeuristicPolicy: {'lost': 0, 'won': 10000, 'tied': 0}

board size: 5x5
player 1 (RandomPolicy) vs. player 2 (Level2HeuristicPolicy)
RandomPolicy: {'lost': 10000, 'tied': 0, 'won': 0}; Level2HeuristicPolicy: {'lost': 0, 'tied': 0, 'won': 10000}

board size: 5x5
player 1 (Level1HeuristicPolicy) vs. player 2 (Level2HeuristicPolicy)
Level1HeuristicPolicy: {'tied': 0, 'won': 461, 'lost': 9539}; Level2HeuristicPolicy: {'tied': 0, 'won': 9539, 'lost': 461}

"""