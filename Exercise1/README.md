# 8 Puzzle

A python3 implementation to calculate solutions for the 8 puzzle using the A* algorithm

## Table of Contents

* General Info
* Team Members
* Modules
* Heuristics
* Functions
* Testing ideas

## General Info

This project is the first exercise of the study course Introduction to AI and Data Science. By using a A* algorithm and
two different heuristic functions a random generated 8 puzzle has to be solved (if solvable). The following parameters
have to be calculated and displayed:

* Branching factor
* expanded nodes
* weighting of the heuristic functions

## Team Members

* Thomas Scheibelhofer
* Harrys Luigi Maria Kavan
* Dominik Hackl
* Matthias Schmid-Kietreiber

## Modules

* numpy
* pandas
* matplotlib.pyplot
* PriorityQueue
* random
* time
* multiprocessing
* jsonpickle
* datetime
* itertools

## Heuristics

The Requirement of the exercise was to use at least two heuristics and to use them on their own and use them combined.
The following heuristic functions have been chosen:

* **Manhattan Distance :** For each cell of the puzzle the distance to their goal state is calculated.
* **Hemming Distance:** Calculates the number of misplaced puzzle pieces.

## Files

* astar.py
* worker.py
* analytics.ipynb

## astar.py

### Usage

The astar.py executes the A* algorithm and prints the results including some statistics, it can be used standalone with
a hardcoded start-puzzle and goal-puzzle which can be set in the main method.

### Functions and classes

#### class Node

* **init(self, name, puzzle, parent, children, heuristic=0)**: initializes the node
* **lt(self, other)**: returns a boolean if the heuristic value between nodes differs
* **eq(self, other)**: returns a boolean if the node is equal with another node
* **hash(self)**: returns a hash value of the puzzle
* **len(self)**: returns the visited nodes for all children of the node

#### expanded_nodes_recursion(node: Node)

This function calculates the number child nodes of the child of the node.

#### calculate_expanded_nodes(root: Node)

This function calculates the number of nodes which have child nodes.

#### calculate_branching_factor(visited_nodes, expanded_nodes)

This function calculates the branching factor of the tree traversed in the algorithm.

#### reconstruct_path(node, expanded_nodes)

This function reconstructs the path by going through the parent nodes.

#### print_path(steps)

Prints the traversed nodes which lead to the solution.

#### a_star(start, goal, heuristics, weights)

This function implements the A* algorithm and is based on the pseudocode found on Wikipedia and pushes the neighbor
nodes of the current node into a priority queue in which they are prioritized by their costs calculated by the heuristic
functions. When the node on top of the queue is the goal node the solution has been found.

#### manhattan_distance(start, goal)

This function calculates the sum of the distances of each puzzle piece to its goal state.

#### build_position_matrix(matrix)

Converts a matrix into a position vector, which represents where an element is found when converting the matrix into a
vector.

#### hamming_distance(start, goal)

This function calculates the number of misplaced puzzle pieces.

#### combine_heuristics(start, goal, heuristics, weights)

This function combines heuristics based on a weighted factor.

#### is_solvable(start)

This function checks whether the puzzle is solvable based on inversions.

#### get_neighbors(parent)

This function is used in the a_star function and calculates the neighbors of the puzzle based on it's current position
of its pieces.

#### generate_metrics(current, root)

This function generates metrics calculated during calculation of the solution of the puzzle.

#### solve(start, goal, heuristics, weights)

This function is the main function of the program and uses the is_solvable, the a_star and the print_path functions to
solve the 8 puzzle and returns if the puzzle is solvable and if it is the time taken, the expanded nodes and the path to
the solution.

## worker.py

To enable the computing of many puzzle solutions needed for the analysis in a reasonable time, this script introduces
multiprocessing.

### Usage

This script can be executed standalone (with dependencies on astar.py) and takes only amount of examples to generate as
input, the results are then persisted to the file system and can be used as input for analysis.

### Functions

#### compute(data)

Used by the multiprocessing, generates the results of the start_run function.

#### process_array(array_with_weights, runs)

This function creates processes with the help of the module multiprocessing and maps the A* algorithm result into a
combined result of all runs.

#### initialize(number_of_examples)

This function creates sets of puzzle tuples with 31 different weights per tuple.

#### start_run(start, weights, heuristics, goal)

Function that uses the A* parameter with the given parameters.

#### write_file(results, append)

Saves the results in json format on the local file system.

#### random_start_node()

This function creates a random start node for the puzzle.

## analytics.ipynb

This jupiter-notebook visualizes the results of the statistical methods to find the best weight ratio for the
heuristics.

## Testing Ideas:

### Unit testing:

- Assert that known numbers of misplaced tiles are calculated correctly by ‘hamming_distance’.
- Assert that known manhattan distance is calculated correctly by ‘manhattan_distance’.

### End-to-End testing:

- Assert that known solvable solutions are true for `is_solvable`.
- Assert that known unsolvable solutions are false for `is_solvable`.
- Assert that known solution has the correct depth for `solve`. 