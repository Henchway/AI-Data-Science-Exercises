from time import time

import jsonpickle
import numpy as np

from astar import random_start_node, is_solvable, a_star, manhattan_distance, hamming_distance


def initialize(runs):
    start_arrays = []
    weights1 = np.arange(1.0, 0.5, -1 / runs)
    weights2 = np.arange(0., 0.5, 1 / runs)
    combined_weights = list(zip(weights1, weights2))

    while len(start_arrays) != runs:
        start_node = random_start_node()
        if is_solvable(start_node):
            start_arrays.append(start_node)

    combined_array = []

    for array in start_arrays:
        for weights in combined_weights:
            combined_array.append([array, weights])

    return combined_array


def start_run(start, weights, goal):
    start_time = time()
    path, expanded_nodes = a_star(start=start, goal=goal,
                                  heuristics=[manhattan_distance, hamming_distance],
                                  weights=list(weights))
    end_time = time()
    return {"array": start, "weights": list(weights), "expanded_nodes": expanded_nodes,
            "time_elapsed": end_time - start_time, "depth": len(path) - 1}


def write_file(results, append):
    with open(f"{append}_results.json", "w") as file:
        frozen = jsonpickle.encode(results)
        file.write(frozen)


if __name__ == "__main__":
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    combined_array = initialize(runs=10)
    for item in combined_array:
        print(start_run(start=item[0], weights=item[1], goal=goal_state))

