from .policy import Policy


class Level4HeuristicPolicy(Policy):
    """
    Based on the approach used on the dotsandboxes.org website. It combines heuristic rules with a negamax search
    to estimate the value of a board position.

    The following description of the approach is taken from the website dotsandboxes.org:
    ```
    "I use a modification of the half-edge data structure to represent the board. Each edge of each box is called a
    half-edge allowing double counting; thus an edge between two neighbouring boxes will be represented by two
    half-edges - one for each box that it is part of. So the total number of half-edges is N = Width × Height × 4 and
    each half-edge is assigned an index ranging from 0 to N − 1.

    First, I maintain two arrays of size N:

    Other half-edge
        Defines the index of the other side of the half-edge. Or − 1 if there is no other half (ie. this edge is on the
        boundary so is a border for only one box)

    Next half-edge
        The index of the next half-edge for the box that is this half-edge is associated with. By “next” we could just
        give the one that is anticlockwise. In fact the ordering doesn't matter, so long as the sequence of next
        half-edges goes in a loop and cover all 4 edges of this box

    When considering which move to take I initially look to see if there are any of the following free moves I can take.
    If there are, I take one and do not bother searching the rest of the game tree.

    - If there is a broken chain of length not equal to 3 I will eat a square from it
    - If there is a broken loop of length greater than 4 I will eat a square from it
    - If there is a broken chain and a broken loop then I will eat a square from the broken loop
    - If there are 2 or more broken loops then I will eat a square from one of them
    - If there are 2 or more broken chains then I will eat a square from one of them

    After doing this the game will be in one of three states

    1. There are no boxes with only 1 edge left
    2. There is one box with only 1 edge left which is part of a chain of length 3
    3. There is one box with only 1 edge left which is part of a loop with 4 boxes remaining

    Once this is done I simplify the game structure somewhat. In the half-edge data structure I only include boxes that
    have 3 or more edges left. Any boxes of valence 2 will be collapsed. Any any boxes of valence 1 will be represented
    by a flag indicating which of the three states above the game is in. So the final data structure consists of

    Other half-edge
        As above

    Next half-edge
        As above

    Chain lengths
        Array of size N representing the number of boxes of valence 2 that have been collapsed between this half-edge
        and its other half-edge

    Independents
        Array of independent chains and loops. The value in the array indicates the size

    Independents Are Loops
        Boolean flag indicating if the Independents are loops or chains

    Looney Value
        Value of 0, 2 or 4 indicating the which of the 3 states above the game is in

    Who To Play Next
        Boolean indicating who will be playing next

    Score so far
        Number of boxes the player to play next already has − number of boxes the other player has

    I then perform a depth-first mega-max search of the game tree with alpha-beta pruning. The end evaluate function
    consists of counting the number of boxes by which one person leads. If the previous move was looney (ie. Looney
    Value above is 2 or 4) then it is known that the next player to move can capture at least half the remaining
    squares. So I simply assign 3/4 to that player."
    ```
    """
    def __init__(self, board_size):
        self._board_size = board_size

    def select_edge(self, board_state):
        pass  # TODO