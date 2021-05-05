import itertools
from datetime import date
from time import time

import jsonpickle
import numpy as np

from astar import random_start_node, is_solvable, a_star, manhattan_distance, hamming_distance, generate_metrics


def initialize(number_of_examples):
    """ Creates 'num_of_examples' sets of puzzle tuples combined with 31 different weights per puzzle tuple"""
    start_arrays = []
    weights1 = np.arange(1.0, 0.7, (-1 / 100)).round(decimals=2)
    weights2 = np.arange(0.0, 0.3 + (1 / 100), (1 / 100)).round(decimals=2)
    combined_weights = zip(weights1, weights2)

    while len(start_arrays) != number_of_examples:
        start_node = random_start_node()
        if is_solvable(start_node):
            try:
                start_arrays.index(start_node)
            except ValueError:
                start_arrays.append(start_node)

    return [element for element in itertools.product(start_arrays, combined_weights)]


def start_run(start, weights, heuristics, goal):
    """ Runs A* with the given input parameters. """
    start_time = time()
    current, root = a_star(start=start, goal=goal,
                           heuristics=heuristics,
                           weights=weights)
    end_time = time()
    path, visited_nodes, expanded_nodes, branching_factor = generate_metrics(current, root)

    return {"array": start, "weights": weights, "visited_nodes": visited_nodes, "expanded_nodes": expanded_nodes,
            "branching_factor": branching_factor,
            "time_elapsed_astar": end_time - start_time, "depth": len(path) - 1}


def write_file(results, append):
    """ Persists the results to the file system."""
    with open(f"results_{append}_{date.today()}.json", "w") as file:
        frozen = jsonpickle.encode(results)
        file.write(frozen)


if __name__ == "__main__":
    pass
