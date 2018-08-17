from dnbpy import *
import operator


def get_selected_index(child_state, parent_state):
    diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
    argmax = max(enumerate(diff), key=lambda x: x[1])[0]
    return argmax

def get_best_action(board_state,policy_action_map,anti_policy_action_map):

    state_prob_map = {}
    for state in policy_action_map:
        policy_prob = policy_action_map[state]
        anti_policy_prob = 1-anti_policy_action_map[state]
        state_prob_map[state] = max(policy_prob,anti_policy_prob)

    sorted_map = sorted(state_prob_map.items(),key=operator.itemgetter(1),reverse=True)
    best_state = sorted_map[0][0]
    return get_selected_index([int(i) for i in best_state], board_state)


def duel(board_size, p1, p2,num_input_channels,antip1=None,antip2=None):
    """
    :param board_size: the game's board size 
    :param p1: the first player
    :param p2: the second player
    :return: the games won, lost and tied, from the perspective of p1
    """
    results = {'won': 0, 'lost': 0, 'tied': 0}
    board_state = init_board_state(board_size)
    for player in range(2):

        for edge_index in range(len(board_state)):
            players = [player, (1 - player)]
            if edge_index % 2 == 0:
                players = [x for x in reversed(players)]
            game = Game_With_Box(board_size, players)
            current_player = game.get_current_player()
            # select the first edge for the first player
            board_state = game.get_board_state()
            tensor = game.get_tensor_representation(current_player,num_input_channels)

            current_player, _ = game.select_edge(edge_index, current_player)
            while not game.is_finished():
                board_state = game.get_board_state()
                tensor = game.get_tensor_representation(current_player,num_input_channels)
                if current_player == 0:
                    if not antip1:
                        edge = p1.select_edge(board_state,tensor)
                    #else:
                     #   policy_action_prob = p1.get_action_probs(board_state)
                      #  anti_policy_action_prob = antip1.get_action_probs(board_state)
                       # edge = get_best_action(board_state,policy_action_prob,anti_policy_action_prob)
                    #edge = p1.select_edge(board_state)

                    current_player, _ = game.select_edge(edge, current_player)
                else:
                    if not antip2:
                        edge = p2.select_edge(board_state,tensor)
                    #else:
                     #   policy_action_prob = p2.get_action_probs(board_state)
                      #  anti_policy_action_prob = antip2.get_action_probs(board_state)
                       # edge = get_best_action(board_state, policy_action_prob, anti_policy_action_prob)
                    #edge = p2.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, current_player)
            p1_score = game.get_score(0)
            p2_score = game.get_score(1)
            #if not not antip1 and not not antip2:

            if p1_score > p2_score:
                results['won'] += 1
            elif p2_score > p1_score:
                results['lost'] += 1
            else:
                results['tied'] += 1
    return results


def tournament(board_size, policy, opponent_pool):
    opponents = opponent_pool.get_opponents()
    tournament_results = {'won': 0, 'lost': 0, 'tied': 0}
    for opponent in opponents:
        results = duel(board_size, policy, opponent)
        tournament_results['won'] += results['won']
        tournament_results['lost'] += results['lost']
        tournament_results['tied'] += results['tied']
    return tournament_results