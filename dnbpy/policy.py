
class Policy:
    def select_edge(self, board_state, score=None, opp_score=None):
        raise NotImplementedError
