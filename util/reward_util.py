

class DelayedBinaryReward:
    def __init__(self):
        pass

    def compute_reward(self, game, policy_name, opponent_name):
        if game.is_finished() and game.get_score(policy_name) > game.get_score(opponent_name):
            return 1.0
        return 0.0

    def __str__(self):
        return "delayed-binary"


class DelayedShapedReward:
    def __init__(self):
        pass

    def compute_reward(self, game, policy_name, opponent_name):
        if game.is_finished():
            score = game.get_score(policy_name)
            return score / len(game.get_all_boxes())
        return 0.0

    def __str__(self):
        return "delayed-shaped"
