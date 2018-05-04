from ai import *

board_size = (2, 2)
num_episodes = 50000
learning_rate = 1.0
gamma = 0.99
epsilon = 0.18

print("initializing value table for (%s, %s) game..." % (board_size[0], board_size[1]))

p1 = TDZeroPolicy(board_size=board_size, epsilon=epsilon, learning_rate=learning_rate, gamma=gamma)

total_rewards = []
results = {'won': 0, 'lost': 0, 'tied': 0}

for episode_num in range(1, num_episodes + 1):
    players = ['p1', 'random']
    if episode_num % 2 == 0:
        players = [x for x in reversed(players)]
    game = Game(board_size, players)
    random_policy = RandomPolicy()
    current_player = game.get_current_player()
    total_reward = 0
    backups = []
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'random':
            random_edge = random_policy.select_edge(board_state)
            current_player, _ = game.select_edge(random_edge, 'random')
        else:
            edge = p1.select_edge(board_state)
            current_player, boxes_made = game.select_edge(edge, 'p1')
            backups.append(game.get_board_state())
            total_reward += boxes_made
            if len(backups) > 1:
                p1.update_value(boxes_made, backups[-2], backups[-1])
    total_rewards.append(total_reward)
    # analyze results
    p1_score = game.get_score('p1')
    random_score = game.get_score('random')
    if p1_score > random_score:
        results['won'] += 1
    elif random_score > p1_score:
        results['lost'] += 1
    else:
        results['tied'] += 1
    if episode_num % 100 == 0:
        vt = p1.get_value_table()
        print("%s, %s, %s, %s" % (episode_num, sum(total_rewards), results, len([x for x in vt if vt[x] > 0.0 ])))
        total_rewards = []
        results = {'won': 0, 'lost': 0, 'tied': 0}


print(p1.get_value_table())
