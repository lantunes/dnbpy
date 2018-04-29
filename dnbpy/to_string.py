class ToString:
    def __init__(self):
        pass

    def apply(self, game):
        rows = game.get_board_size()[0]
        cols = game.get_board_size()[1]
        board_state = game.get_board_state()
        edge_to_boxes = self._get_edge_to_boxes(game.get_all_boxes())
        players = game.get_players()
        cur_index = 0
        board = ""
        for row in range(rows):
            if row == 0:
                for c in range(cols):
                    board += self._print_edges(c, board_state, cur_index)
                    cur_index += 1
                board += '  \n'
                board += self._print_spacer_line(cols, board_state, cur_index)
                board += '\n'

            for c in range(cols + 1):
                edge_val = "|" if self._is_edge_selected(board_state, cur_index) else cur_index
                board += '{0: <3}'.format(edge_val)
                if c < cols:
                    player_for_box = self._get_player_for_box(cur_index, edge_to_boxes, game, players)
                    if player_for_box is not None:
                        board += ' {0: <5.5} '.format(player_for_box)
                    else:
                        board += '       '
                cur_index += 1
            board += '\n'
            board += self._print_spacer_line(cols, board_state, cur_index - (cols + 1))
            board += '\n'

            for c in range(cols):
                board += self._print_edges(c, board_state, cur_index)
                cur_index += 1
            board += '  \n'
            if row != (rows - 1):
                board += self._print_spacer_line(cols, board_state, cur_index)
            board += '\n'
        board += self._print_player_scores(game, players)
        return board

    def _print_edges(self, c, board_state, cur_index):
        if c == 0:
            if self._is_edge_selected(board_state, cur_index):
                return '*---------*'
            return '*    {0: <3}  *'.format(cur_index)
        else:
            if self._is_edge_selected(board_state, cur_index):
                return '---------*'
            return '    {0: <3}  *'.format(cur_index)

    def _print_spacer_line(self, cols, board_state, i):
        line = ""
        for c in range(cols + 1):
            edge_val = "|" if self._is_edge_selected(board_state, i) else ''
            line += '{0: <3}'.format(edge_val)
            if c < cols:
                line += '       '
            i += 1
        return line

    def _is_edge_selected(self, board_state, edge_index):
        return board_state[edge_index] == 1

    def _get_edge_to_boxes(self, boxes):
        edge_to_boxes = {}
        for box in boxes:
            edge_to_boxes[box.get_edges()[1]] = box
        return edge_to_boxes

    def _get_player_for_box(self, edge, edge_to_boxes, game, players):
        box = edge_to_boxes[edge]
        for player in players:
            player_boxes = game.get_boxes(player)
            for player_box in player_boxes:
                if str(box) == str(player_box):
                    return player
        return None

    def _print_player_scores(self, game, players):
        scores = ""
        for i, player in enumerate(players):
            scores += "{0:}: {1:}".format(player, game.get_score(player))
            if i < len(players) - 1:
                scores += ", "
        return scores