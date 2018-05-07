from os import listdir
from os.path import isfile, join
import csv


def convert_edge_index_3x3(edge_number):
    edge_index_map  = {
        0: 0,
        1: 1,
        2: 2,
        3: 7,
        4: 8,
        5: 9,
        6: 14,
        7: 15,
        8: 16,
        9: 21,
        10: 22,
        11: 23,
        12: 3,
        13: 10,
        14: 17,
        15: 4,
        16: 11,
        17: 18,
        18: 5,
        19: 12,
        20: 19,
        21: 6,
        22: 13,
        23: 20
    }
    return edge_index_map[edge_number]

output_file = open('/Users/luis/dnb-moves.csv', 'w')
path_to_csv_files = "/Users/luis/dotsandboxes.org-data-labeled-3x3-playeragnostic"
csv_files = [f for f in listdir(path_to_csv_files) if isfile(join(path_to_csv_files, f))]
for f in csv_files:
    with open(path_to_csv_files + '/' + f, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            try:
                board_state = [0]*24
                if row[-2] != '[]':
                    raw_indices = [int(x) for x in row[-2][1:len(row[-2])-1].split(' ')]
                    converted_indices = [convert_edge_index_3x3(x) for x in raw_indices]
                    for i in converted_indices:
                        board_state[i] = 1
                board_state_string = ''.join([str(x) for x in board_state])
                print(row)
                output_file.write(', '.join([board_state_string, str(convert_edge_index_3x3(int(row[-1])))]) + '\n')
            except Exception:
                print(row)
                raise
output_file.close()

