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

game = dnbpy.Game((2, 2), ['player1', 'player2'])

game.select_edge(0, 'player1')
game.select_edge(2, 'player2')
game.select_edge(3, 'player1')
game.select_edge(5, 'player2')

score = game.get_score('player1')
# score is 0

score = game.get_score('player2')
# score is 1

game_finished = game.is_finished()
# game_finished is False
```

## Command-line demo

DnBPy includes a simple command-line demo program. Start the program with:
```
python play.py
```

<img src="https://raw.githubusercontent.com/lantunes/dnbpy/master/resources/screenshot.png" width="25%"/>
