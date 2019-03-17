import unittest

from dnbpy.ai import MCTSGame


class TestMCTSGame(unittest.TestCase):

    def test_2x2_select_edge(self):
        game = MCTSGame((2, 2), players=['root', 'opponent'])
        self.assertEqual(game.is_finished(), False)
        self.assertEqual(game.get_current_player(), 'root')
        next_player = game.select_edge(2, 'root')
        self.assertEqual(next_player, 'opponent')
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_player = game.select_edge(5, 'opponent')
        self.assertEqual(next_player, 'root')
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_2x2_get_score(self):
        game = MCTSGame((2, 2), players=['root', 'opponent'])
        game.select_edge(0, 'root')
        game.select_edge(2, 'opponent')
        game.select_edge(3, 'root')
        game.select_edge(5, 'opponent')
        # opponent makes a box
        self.assertFalse(game.is_finished())
        game.select_edge(11, 'opponent')
        game.select_edge(6, 'root')
        game.select_edge(10, 'opponent')
        game.select_edge(8, 'root')
        game.select_edge(9, 'opponent')
        game.select_edge(7, 'opponent')
        # opponent makes 2 more boxes for a total of 3
        game.select_edge(4, 'opponent')
        game.select_edge(1, 'root')
        # root makes 1 box
        self.assertTrue(game.is_finished())
        with self.assertRaises(Exception) as e:
            game.select_edge(1, 'root')
        self.assertTrue(("game is finished" in str(e.exception)))
        score = game.get_score('root')
        self.assertEqual(score, 1)
        score = game.get_score('opponent')
        self.assertEqual(score, 3)

    def test_2x2_existing_state(self):
        board_state = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
        game = MCTSGame((2, 2), players=['root', 'opponent'], board_state=board_state, root_player_score=1)
        game.select_edge(3, 'root')
        game.select_edge(5, 'root')
        game.select_edge(10, 'root')
        self.assertTrue(game.is_finished())
        score = game.get_score('root')
        self.assertEqual(score, 4)
        score = game.get_score('opponent')
        self.assertEqual(score, 0)

    def test_2x2_existing_state2(self):
        board_state = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
        game = MCTSGame((2, 2), players=['root', 'opponent'], board_state=board_state, root_player_score=0)
        game.select_edge(10, 'root')
        game.select_edge(5, 'opponent')
        game.select_edge(6, 'opponent')
        game.select_edge(3, 'opponent')
        self.assertTrue(game.is_finished())
        score = game.get_score('root')
        self.assertEqual(score, 0)
        score = game.get_score('opponent')
        self.assertEqual(score, 4)

    def test_2x2_existing_state3(self):
        board_state = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
        game = MCTSGame((2, 2), players=['root', 'opponent'], board_state=board_state, root_player_score=0)
        game.select_edge(10, 'root')
        game.select_edge(3, 'opponent')
        game.select_edge(5, 'root')
        game.select_edge(6, 'root')
        self.assertTrue(game.is_finished())
        score = game.get_score('root')
        self.assertEqual(score, 4)
        score = game.get_score('opponent')
        self.assertEqual(score, 0)
