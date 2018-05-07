import ai
import dnbpy

board_size = (2, 2)

td1 = ai.TDOnePolicy(board_size, epsilon=0.0, learning_rate=0.0, gamma=0.0,
                     table_file_path='resources/td1_2x2_0.6_1.0_0.99_delayed_selfplay100k.txt')
ent = ai.CausalEntropicPolicy(board_size, max_sample_paths=10000)
rand = dnbpy.RandomPolicy()


# CausalEntropicPolicy vs RandomPolicy: {'won': 374, 'tied': 102, 'lost': 24}
player_map = {0: ent, 1: rand}

results = {'won': 0, 'lost': 0, 'tied': 0}
for trial in range(500):
    players = [0, 1]
    if trial % 2 == 0:
        players = [x for x in reversed(players)]
    game = dnbpy.Game(board_size, players=players)
    current_player = game.get_current_player()
    while not game.is_finished():
        if current_player == 0:
            edge = player_map[0].select_edge(game.get_board_state())
        else:
            edge = player_map[1].select_edge(game.get_board_state())
        current_player, _ = game.select_edge(edge, player=current_player)
    policy_score = game.get_score(0)
    random_score = game.get_score(1)
    if policy_score > random_score:
        results['won'] += 1
    elif random_score > policy_score:
        results['lost'] += 1
    else:
        results['tied'] += 1
    print(results)