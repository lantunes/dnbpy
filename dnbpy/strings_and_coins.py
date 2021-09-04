

class StringsAndCoins:
    """
    A graph of a strings and coins representation of the game. All edges are implicitly bi-directional.
    """
    def __init__(self, game):
        self.nodes = []
        self.edges = []
        self._node_degrees = {}
        self._create_graph(game)

    def _create_graph(self, game):
        boxes = game.get_all_boxes()
        board_state = game.get_board_state()
        num_rows, num_cols = game.get_board_size()

        box_edge_to_boxes = {}
        curr_row = 0
        curr_col = 0
        for box in boxes:
            name = str(box)
            self.add_node(name)
            for box_edge in box.get_edges():
                if box_edge not in box_edge_to_boxes:
                    box_edge_to_boxes[box_edge] = []
                else:
                    # form a link to all the nodes that share this edge if that edge is not selected
                    for existing in box_edge_to_boxes[box_edge]:
                        if board_state[box_edge] == 0:
                            self.add_edge(existing, name)
                box_edge_to_boxes[box_edge].append(name)

            # if we're on the first and/or last box of a row
            if curr_col == 0:
                self._add_ground(box, board_state, name, 1)
            if curr_col == (num_cols - 1):
                self._add_ground(box, board_state, name, 2)

            # if we're on the first or last row
            if curr_row == 0:
                self._add_ground(box, board_state, name, 0)
            if curr_row == (num_rows - 1):
                self._add_ground(box, board_state, name, 3)

            curr_col = (curr_col + 1) % num_cols
            curr_row = (curr_row + 1) if curr_col == 0 else curr_row

    def _add_ground(self, box, state, name, idx):
        ground_edge = box.get_edges()[idx]
        ground_node = "ground-%s" % ground_edge
        self.add_node(ground_node)
        if state[ground_edge] == 0:
            self.add_edge(ground_node, name)

    def add_node(self, node):
        self.nodes.append(node)
        self._node_degrees[node] = 0

    def add_edge(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise Exception("%s is not a node in the graph" % (node1 if node1 not in self.nodes else node2))
        self.edges.append((node1, node2))
        self._node_degrees[node1] += 1
        self._node_degrees[node2] += 1

    def get_node_degree(self, node):
        return self._node_degrees[node]

    def to_adjacency_matrix(self):
        num_nodes = len(self.nodes)
        A = [[0] * num_nodes for _ in range(num_nodes)]
        for n1, n2 in self.edges:
            i = self.nodes.index(n1)
            j = self.nodes.index(n2)
            A[i][j] = 1
            A[j][i] = 1
        return A

    def get_long_chains(self):
        A = self.to_adjacency_matrix()
        chains = []
        excluded = set()
        for i, node in enumerate(self.nodes):
            if self.get_node_degree(node) == 1:
                # we get the first (and only) connection, since we know it is degree 1
                next_idx = self._non_zero_indices(A[i])[0]
                next_node = self.nodes[next_idx]
                prev_idx = i

                visited = []
                if not node.startswith("ground"):
                    visited.append(node)

                while self.get_node_degree(next_node) == 2 and not next_node in excluded:
                    visited.append(next_node)
                    connections = self._non_zero_indices(A[next_idx])
                    new_idx = [i for i in connections if i != prev_idx][0]
                    next_node = self.nodes[new_idx]
                    prev_idx = next_idx
                    next_idx = new_idx

                if len(visited) >= 3:
                    chains.append(visited)
                    [excluded.add(n) for n in visited]

        return chains

    def _non_zero_indices(self, arr):
        return [i for i, v in enumerate(arr) if v != 0]