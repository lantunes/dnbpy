from threading import Thread

from dnbpy import *


def evaluate(policy, board_size, num_trials, opponents,num_input_channels):
    final_results = {}
    for opponent_policy in opponents:
        results = {'won': 0, 'lost': 0, 'tied': 0}
        for trial in range(num_trials):
            players = ['policy', 'opponent']
            if trial % 2 == 0:
                players = [x for x in reversed(players)]
            game = Game_With_Box(board_size, players)
            current_player = game.get_current_player()
            while not game.is_finished():
                board_state = game.get_board_state()
                if current_player == 'opponent':
                    edge = opponent_policy.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, 'opponent')
                else:
                    tensor = game.get_tensor_representation(current_player,num_input_channels)
                    edge = policy.select_edge(board_state,tensor)
                    current_player, _ = game.select_edge(edge, 'policy')
            policy_score = game.get_score('policy')
            opponent_score = game.get_score('opponent')
            if policy_score > opponent_score:
                results['won'] += 1
            elif opponent_score > policy_score:
                results['lost'] += 1
            else:
                results['tied'] += 1
        final_results[opponent_policy.__class__.__name__] = results
    return final_results


class EvaluationThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        Thread.join(self)
        return self._return


def evaluate_parallel(policy, board_size, num_trials, opponents):
    final_results = {}
    workers = []
    for opponent_policy in opponents:
        t = EvaluationThread(target=evaluate, args=(policy, board_size, num_trials, [opponent_policy]))
        t.start()
        workers.append(t)
    for t in workers:
        results = t.join()
        final_results.update(results)
    return final_results


