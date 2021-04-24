from time import time

import jsonpickle
import numpy as np

from astar import random_start_node, is_solvable, a_star, manhattan_distance, hamming_distance

runs = 20
start_arrays = []
results = []
goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
weights1 = np.arange(1.0, 0., -1 / runs)
weights2 = np.arange(0., 1., 1 / runs)
combined_weights = list(zip(weights1, weights2))

while len(start_arrays) != runs:
    start_node = random_start_node()
    if is_solvable(start_node):
        start_arrays.append(start_node)

for start in start_arrays:
    for weights in combined_weights:
        start_time = time()
        path, expanded_nodes = a_star(start=start, goal=goal_state, heuristics=[manhattan_distance, hamming_distance],
                                      weights=list(weights))
        end_time = time()
        results.append([start, list(weights), expanded_nodes, end_time - start_time])
        print("Still working...")

with open("results.json", "w") as file:
    frozen = jsonpickle.encode(results)
    file.write(frozen)
