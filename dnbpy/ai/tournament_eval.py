from dnbpy.ai import *
from dnbpy.util.evaluator import *
import time

board_size = (3, 3)

pg_params = read_params('../out/dnbpy52/weights-225000.txt')
pg_model = PGPolicy3x3CNN(board_size, existing_params=pg_params, activation=tf.nn.relu)
pg_model.set_boltzmann_action(False)
pg_model.set_epsilon(0.0)

start = time.time()
opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
results = evaluate(pg_model, board_size, 1000, opponents)
end = time.time()
print("%s, %s, %s, %s" % (results[RandomPolicy.__name__]['won'],
                          results[Level1HeuristicPolicy.__name__]['won'],
                          results[Level2HeuristicPolicy.__name__]['won'], (end - start)))

start = time.time()
opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
results = evaluate_parallel(pg_model, board_size, 1000, opponents)
end = time.time()
print("%s, %s, %s, %s" % (results[RandomPolicy.__name__]['won'],
                          results[Level1HeuristicPolicy.__name__]['won'],
                          results[Level2HeuristicPolicy.__name__]['won'], (end - start)))

"""
999, 976, 910, 43.395017862319946
997, 964, 900, 48.95261907577515
999, 955, 904, 48.22842812538147
1000, 966, 905, 44.11811399459839


1000, 973, 920, 43.78155589103699
1000, 979, 900, 29.492024898529053

1000, 973, 906, 43.845345973968506
999, 959, 882, 28.479337215423584

1000, 964, 902, 44.76214599609375
1000, 972, 916, 28.796285152435303
"""
