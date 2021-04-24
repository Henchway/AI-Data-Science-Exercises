import random
import time
from queue import PriorityQueue

import numpy as np


class Node:
    """Used to store tree information of the A* algorithm."""

    def __init__(self, name, puzzle, parent, heuristic=0):
        self.name = name
        self.puzzle = puzzle
        self.parent = parent
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.puzzle == other.puzzle

    def __hash__(self):
        return hash(self.puzzle)


def random_start_node():
    """Generates a random potentially unsolvable initial start puzzle."""
    init = [x for x in range(0, 9)]
    random.shuffle(init)
    return tuple(init)


def reconstruct_path(node, expanded_nodes):
    """Reconstruct the path by going through the parent nodes."""
    steps = []
    while node.parent is not None:
        steps.append(node)
        node = node.parent
    steps.append(node)
    return steps, expanded_nodes


def print_path(steps):
    """Print the arrays of the path."""
    steps.reverse()
    print(f"The puzzle was solved in {len(steps) - 1} steps.")
    for step in steps:
        print("===========")
        print(np.reshape(step.puzzle, (3, 3)))
        print("===========")


def a_star(start, goal, heuristics, weights):
    """Based on the pseudo-code on Wikipedia: https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode"""

    root = Node("root", start, None, heuristic=0)
    open_set = PriorityQueue()
    expanded_nodes = 1

    g_score = {root: 0}
    root.heuristic = g_score[root] + combine_heuristics(start=root.puzzle, goal=goal, heuristics=heuristics,
                                                        weights=weights)
    open_set.put((root.heuristic, root))

    while open_set.qsize() > 0:
        current = open_set.get()[1]
        if current.puzzle == goal:
            return reconstruct_path(current, expanded_nodes)

        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g_score
                neighbor.heuristic = g_score[neighbor] + combine_heuristics(start=neighbor.puzzle, goal=goal,
                                                                            heuristics=heuristics,
                                                                            weights=weights)
                if not any((neighbor.heuristic, neighbor) in item for item in open_set.queue):
                    open_set.put((neighbor.heuristic, neighbor))
                    expanded_nodes += 1

        open_set.task_done()
    return False


def manhattan_distance(start, goal):
    """Calculates the overall manhattan distance of the start node to the goal node."""
    distance = 0
    start = np.reshape(start, (3, 3))
    goal = np.reshape(goal, (3, 3))

    for i, array in enumerate(start):
        for j, number in enumerate(array):
            (row, column) = np.where(goal == number)
            distance += abs(i - row) + abs(j - column)
    return int(distance)


def hamming_distance(start, goal):
    """Counts the misplaced puzzle pieces."""
    misplaced = 0
    start = np.reshape(start, (3, 3))
    goal = np.reshape(goal, (3, 3))
    for i, array in enumerate(start):
        for j, number in enumerate(array):
            if start[i][j] != goal[i][j]:
                misplaced += 1
    return misplaced


def combine_heuristics(start, goal, heuristics, weights):
    """Combines the manhattan distance with the misplaced puzzle pieces."""

    total = 0
    for i, heuristic in enumerate(heuristics):
        total += int(weights[i] * heuristic(start, goal))

    return total


def is_solvable(start) -> bool:
    """
    Source: https://github.com/IsaacCheng9/8-puzzle-heuristic-search/blob/master/src/8_puzzle_specific.py
    Checks whether the 8-puzzle problem is solvable based on inversions.
    Args:
        start: The start state of the board input by the user.
    Returns:
        Whether the 8-puzzle problem is solvable.
    """
    start = np.reshape(start, (3, 3))
    k = start[start != 0]
    num_inversions = sum(
        len(np.array(np.where(k[i + 1:] < k[i])).reshape(-1)) for i in
        range(len(k) - 1))
    return num_inversions % 2 == 0


def get_neighbors(parent):
    parent_puzzle = np.reshape(parent.puzzle, (3, 3))

    (row, col) = np.where(parent_puzzle == 0)
    row = int(row[0])
    col = int(col[0])
    neighbors = []

    if row != 0 and parent.name != "down":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row - 1][col]
        node_puzzle[row - 1][col] = 0
        node = Node(name="up", puzzle=tuple(np.reshape(node_puzzle, (9,))), parent=parent)
        neighbors.append(node)

    if row != len(parent_puzzle) - 1 and parent.name != "up":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row + 1][col]
        node_puzzle[row + 1][col] = 0
        node = Node(name="down", puzzle=tuple(np.reshape(node_puzzle, (9,))), parent=parent)
        neighbors.append(node)

    if col != 0 and parent.name != "right":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row][col - 1]
        node_puzzle[row][col - 1] = 0
        node = Node(name="left", puzzle=tuple(np.reshape(node_puzzle, (9,))), parent=parent)
        neighbors.append(node)

    if col != len(parent_puzzle[0]) - 1 and parent.name != "left":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row][col + 1]
        node_puzzle[row][col + 1] = 0
        node = Node(name="right", puzzle=tuple(np.reshape(node_puzzle, (9,))), parent=parent)
        neighbors.append(node)

    return neighbors


def solve(start, goal, heuristics, weights):
    solvable = is_solvable(initial_state)
    if not solvable:
        print(f"The puzzle {start} is NOT solvable.")
        return False

    start_time = time.time()
    path, expanded_nodes = a_star(start=start, goal=goal, heuristics=heuristics, weights=weights)
    end_time = time.time()

    if path is False:
        print("Something went wrong.")

    print(f"The search took {end_time - start_time} seconds")
    print(f"The number of expanded nodes is: {expanded_nodes}")
    print_path(path)

if __name__ == '__main__':
    # initial_state = (7, 2, 4, 5, 0, 6, 8, 3, 1)
    initial_state = (0, 7, 2, 3, 1, 4, 8, 5, 6)
    # initial_state = random_start_node()
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    solve(start=initial_state, goal=goal_state, heuristics=[manhattan_distance, hamming_distance], weights=[0.8, 0.2])
