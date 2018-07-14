from ai import *

board_size = (2, 2)
# n_games = 10000
n_games = 500

# player1 = RandomPolicy()
player1 = Level1HeuristicPolicy(board_size)
# player2 = Level1HeuristicPolicy(board_size)
# player2 = Level2HeuristicPolicy(board_size)
player2 = CausalEntropicPolicy(board_size, max_sample_paths=10000)

print("board size: %sx%s" % board_size)
print("player 1 (%s) vs. player 2 (%s)" % (player1.__class__.__name__, player2.__class__.__name__))

p1_results = {'won': 0, 'lost': 0, 'tied': 0}
p2_results = {'won': 0, 'lost': 0, 'tied': 0}
for trial in range(n_games):
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
# print ("%s: %s; %s: %s" % (player1.__class__.__name__, p1_results, player2.__class__.__name__, p2_results))

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

                                            
    ----------------------------------------------------------------------------------------
            | L0 2x2 | L1 2x2 | L2 2x2 | L0 3x3 | L1 3x3 | L2 3x3 | L0 5x5 | L1 5x5 | L2 5x5
    ----------------------------------------------------------------------------------------
     L0 2x2 |   -    |  4.40% |  3.12% |   -    |   -    |   -    |    -   |   -    |   -     
    ----------------------------------------------------------------------------------------
     L1 2x2 | 77.61% |    -   | 12.31% |   -    |   -    |   -    |    -   |   -    |   -    
    ----------------------------------------------------------------------------------------   
     L2 2x2 | 79.95% | 54.08% |    -   |   -    |   -    |   -    |    -   |   -    |   -    
    ---------------------------------------------------------------------------------------- 
     L0 3x3 |   -    |    -   |    -   |   -    | 0.31%  | 0.34%  |    -   |   -    |   -      
    ----------------------------------------------------------------------------------------
     L1 3x3 |   -    |    -   |    -   | 99.69% |   -    | 15.91% |    -   |   -    |   -       
    ---------------------------------------------------------------------------------------- 
     L2 3x3 |   -    |    -   |    -   | 99.66% | 84.09% |   -    |    -   |   -    |   -
    ---------------------------------------------------------------------------------------- 
     L0 5x5 |   -    |    -   |    -   |   -    |   -    |   -    |    -   |   0%   |   0%
    ---------------------------------------------------------------------------------------- 
     L1 5x5 |   -    |    -   |    -   |   -    |   -    |   -    |  100%  |   -    | 4.61%
    ---------------------------------------------------------------------------------------- 
     L2 5x5 |   -    |    -   |    -   |   -    |   -    |   -    |  100%  | 95.39% |   -
    ---------------------------------------------------------------------------------------- 
    Win rates of L0, L1, and L2 policies versus each other on 2x2, 3x3, and 5x5, 
     after tournaments consisting of 10,000 games each.
    L0: A policy that selects an edge at random.
    L1: A policy that selects an edge randomly that completes a box, if possible. Otherwise, 
        it selects an edge at random.
    L2: A policy that selects an edge randomly that completes a box, if possible. Otherwise, 
        it selects an edge randomly that does not allow the opponent to complete a box. 
        Otherwise, it selects an edge at random.
        
CausalEntropicPolicy(board_size=(2,2), max_sample_paths=1000) vs RandomPolicy: 
RandomPolicy: {'won': 21, 'lost': 364, 'tied': 115}; CausalEntropicPolicy: {'won': 364, 'lost': 21, 'tied': 115}

CausalEntropicPolicy(board_size=(2,2), max_sample_paths=10000) vs RandomPolicy
RandomPolicy: {'won': 1, 'lost': 69, 'tied': 30}; CausalEntropicPolicy: {'won': 69, 'lost': 1, 'tied': 30}

CausalEntropicPolicy(board_size=(2,2), max_sample_paths=1000) vs Level1HeuristicPolicy
Level1HeuristicPolicy: {'won': 371, 'lost': 5, 'tied': 124}; CausalEntropicPolicy: {'won': 5, 'lost': 371, 'tied': 124}

CausalEntropicPolicy(board_size=(2,2), max_sample_paths=10000) vs Level1HeuristicPolicy
Level1HeuristicPolicy: {'won': 84, 'lost': 4, 'tied': 29}; CausalEntropicPolicy: {'won': 4, 'lost': 84, 'tied': 29}
"""