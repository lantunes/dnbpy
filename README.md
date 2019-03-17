Dots 'n Boxes Game Engine for Python
====================================

<img src="https://raw.githubusercontent.com/lantunes/dnbpy/master/resources/screenshot.png" width="25%"/>

Installation:
```
pip install dnbpy
```

Usage example:
```
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
