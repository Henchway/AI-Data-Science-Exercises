import itertools
import multiprocessing as mp
import random
from datetime import date
from datetime import datetime
from time import time

import jsonpickle
import numpy as np

from astar import manhattan_distance, hamming_distance, a_star, is_solvable, generate_metrics


def random_start_node():
    """Generates a random potentially unsolvable initial start puzzle."""
    init = [x for x in range(0, 9)]
    random.shuffle(init)
    return tuple(init)


def compute(data):
    # Put the call to the A* in here.
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    return start_run(start=data[0], weights=data[1], heuristics=[manhattan_distance, hamming_distance], goal=goal_state)


def process_array(array_with_weights, runs):
    p = mp.Pool()
    res = p.map(compute, array_with_weights)
    write_file(res, runs)


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
    start_time_astar = time()
    current, root = a_star(start=start, goal=goal,
                           heuristics=heuristics,
                           weights=weights)
    end_time_astar = time()
    path, visited_nodes, expanded_nodes, branching_factor = generate_metrics(current, root)

    return {"array": start, "weights": weights, "visited_nodes": visited_nodes, "expanded_nodes": expanded_nodes,
            "branching_factor": branching_factor,
            "time_elapsed_astar": end_time_astar - start_time_astar, "depth": len(path) - 1}


def write_file(results, append):
    """ Persists the results to the file system."""
    with open(f"results_{append}_{date.today()}.json", "w") as file:
        frozen = jsonpickle.encode(results)
        file.write(frozen)


def main():
    examples = 1

    start_time = datetime.now()
    print(f"Start: {start_time}")

    work_to_do = initialize(examples)
    process_array(work_to_do, examples)

    end_time = datetime.now()
    print(f"End: {end_time}")
    print(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
