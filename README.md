Dots 'n Boxes Game Engine for Python
====================================
DnBPy is a light-weight Dots 'n Boxes game engine for Python. It is
particularly useful for AI projects, and can be used as an environment
for Reinforcement Learning projects.

## Installation
```
pip install dnbpy
```

## Usage example
```python
import dnbpy

# (2, 2) means: create a 2x2 board
# player identifiers are not limited to strings
game = dnbpy.Game((2, 2), ['player1', 'player2'])

game.select_edge(0, 'player1')
game.select_edge(2, 'player2')
game.select_edge(3, 'player1')
game.select_edge(5, 'player2')

print(game.get_score('player1'))
# 0

print(game.get_score('player2'))
# 1

print(game.is_finished())
# False
```

## The Board State
The board state is represented as a bit vector. The bit vector's length
is equal to the number of edges. The index of a bit in the vector
corresponds to an edge on the board. If the bit has value `1`, the edge
is selected, and if the bit has value `0`, the edge is unselected.

The board edges are indexed as follows (for the 2x2 case):
<!-- language: lang-none -->
    *    0    *    1    *

    2         3         4

    *    5    *    6    *

    7         8         9

    *    10   *    11   *

In the example above, if edges 1 and 5 are selected, the board state
will be `01000100000`. For other board dimensions, the same
convention is followed for edge indexing. The horizontal edges at the
very top of the board are indexed first, left-to-right, followed by the
vertical edges of the first row, etc.

A board is described using the convention `n x m`, where `n` is the
number of rows, and `m` is the number of columns. The 2x3 case is indexed
as follows:
<!-- language: lang-none -->
    *    0    *    1    *    2    *

    3         4         5         6

    *    7    *    8    *    9    *

    10        11        12        13

    *    14   *    15   *    16   *


### Boxes

A board consists of a number of boxes. The boxes on a board can be
accessed:
```python
game = dnbpy.Game((2, 2), ['player1', 'player2'])

boxes = game.get_all_boxes()
print(len(boxes))
# 4
print(str(boxes[0]))
# 0-2-3-5
print(str(boxes[1]))
# 1-3-4-6
print(str(boxes[2]))
# 5-7-8-10
print(str(boxes[3]))
# 6-8-9-11
```

During a game, the game engine keeps track of the boxes that players
capture:
```python
game = dnbpy.Game((2, 2), ['player1', 'player2'])
game.select_edge(0, 'player1')
game.select_edge(2, 'player2')
game.select_edge(3, 'player1')
game.select_edge(5, 'player2')

boxes = game.get_boxes('player1')
print(len(boxes))
# 0

boxes = game.get_boxes('player2')
print(len(boxes))
# 1
print(str(boxes[0]))
# 0-2-3-5
```

## Command-line demo

DnBPy includes a simple command-line demo program. Start the program with:
```
python play.py
```

<img src="https://raw.githubusercontent.com/lantunes/dnbpy/master/resources/screenshot.png" width="25%"/>
