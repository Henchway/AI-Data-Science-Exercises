import heapq
import time
from typing import List

import numpy as np


class Node:
    """Used to store tree information of the A* algorithm."""

    def __init__(self, name, puzzle, parent, children, heuristic=0):
        self.name: str = name
        self.puzzle: tuple = puzzle
        self.__puzzle_hash__ = hash(puzzle)
        self.parent: Node = parent
        self.children: List[Node] = children
        self.heuristic: int = heuristic

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def __eq__(self, other: "Node"):
        return self.__puzzle_hash__ == other.__puzzle_hash__

    def __hash__(self):
        return hash(self.puzzle)

    def __len__(self):
        def visited_nodes_recursion(node: Node):
            """Calculate the visited nodes for all children of the given node."""
            visited_nodes = 0
            for child in node.children:
                visited_nodes += 1
                visited_nodes += visited_nodes_recursion(child)
            return visited_nodes

        return 1 + visited_nodes_recursion(self)


def expanded_nodes_recursion(node: Node):
    """Calculate the expanded nodes (nodes which have children of their own)
     for all children of the given node."""
    expanded_nodes = 0
    for child in node.children:
        if child.children:
            expanded_nodes += expanded_nodes_recursion(child) + 1
    return expanded_nodes


def calculate_expanded_nodes(root: Node):
    """" Count all nodes in the tree which have children."""
    expanded_nodes = 1  # 1 to account for the root
    expanded_nodes += expanded_nodes_recursion(root)
    return expanded_nodes


def calculate_branching_factor(visited_nodes, expanded_nodes):
    """ Source for the calculation: Wikipedia
    https://en.wikipedia.org/wiki/Branching_factor """
    return (visited_nodes - 1) / expanded_nodes


def reconstruct_path(node):
    """Reconstruct the path by going through the parent nodes."""
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.append(node)
    return path


def print_path(steps):
    """Print the arrays of the path."""
    steps.reverse()
    for step in steps:
        print("===========")
        print(np.reshape(step.puzzle, (3, 3)))
        print("===========")


def a_star(start, goal, heuristics, weights):
    """Based on the pseudo-code on Wikipedia: https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode"""

    root = Node(name="root", puzzle=start, parent=None, children=[], heuristic=0)
    open_set = []
    heapq.heapify(open_set)

    g_score = {root: 0}
    root.heuristic = g_score[root] + combine_heuristics(start=root.puzzle, goal=goal, heuristics=heuristics,
                                                        weights=weights)
    heapq.heappush(open_set, (root.heuristic, root))

    while len(open_set) > 0:
        current = heapq.heappop(open_set)[1]  # Retrieve first item and extract the node from the tuple
        if current.puzzle == goal:
            return current, root
        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            current.children.append(neighbor)
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g_score
                neighbor.heuristic = g_score[neighbor] + combine_heuristics(start=neighbor.puzzle, goal=goal,
                                                                            heuristics=heuristics,
                                                                            weights=weights)

                if not (neighbor.heuristic, neighbor) in open_set:
                    heapq.heappush(open_set, (neighbor.heuristic, neighbor))
        # open_set.task_done()
    return False


def manhattan_distance(start, goal):
    """
    Calculates the distance of the elements in the matrix.
    """
    m = 3
    start = np.reshape(start, (m, m))
    goal = np.reshape(goal, (m, m))
    pos_matrix_start = build_position_matrix(start)
    pos_matrix_goal = build_position_matrix(goal)

    x_start_pos, y_start_pos = pos_matrix_start % m, pos_matrix_start // m
    x_goal_pos, y_goal_pos = pos_matrix_goal % m, pos_matrix_goal // m

    return np.sum(np.abs(x_start_pos - x_goal_pos) + np.abs(y_start_pos - y_goal_pos))


def build_position_matrix(matrix):
    """Converts a matrix into a position vector, which represents where an element is found when converting the matrix into a vector.

    Args:
        matrix (np.array): An array of n*m shape. It has to be two dimensional and n = m.

    Returns:
        np.array: Returns a vector which displays where in the elements would be found in a reshaped (n*m, 1) matrix.

    Example: 
        The input `[[3,0], [2,1]]` gets converted into `[3,0,2,1]` and the function will return the position of the elements relative to the index in the array.
        Element 0 is found at position 1.
        Element 1 is found at position 3.
        Element 2 is found at position 2.
        Element 3 is found at position 0.
        So this function would return: [1,3,2,0]
    """

    n, m = np.shape(matrix)
    nn = n * n
    pos_matrix = np.empty(nn, dtype=int)
    pos_matrix[matrix.reshape(nn)] = np.arange(nn)
    return pos_matrix


@DeprecationWarning
def manhattan_iterative(start, goal):
    distance = 0
    distance = 0
    start = np.reshape(start, (3, 3))
    goal = np.reshape(goal, (3, 3))
    pos_matrix1 = build_position_matrix
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
    """
    Combines the manhattan distance with the hamming distance.
    Can be used to combine any number of heuristics with the same number of weights.
    """

    total = 0.0
    for i, heuristic in enumerate(heuristics):
        if weights[i] > 0:
            total += weights[i] * heuristic(start, goal)

    return float(total)


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
    """
    Determine the possible neighbors of the current array, and return them as nodes.
    Backsteps (e.g. an 'up' following a 'down') are prohibited.
    """
    parent_puzzle = np.reshape(parent.puzzle, (3, 3))

    (row, col) = np.where(parent_puzzle == 0)
    row = int(row[0])
    col = int(col[0])
    neighbors = []

    if row != 0 and parent.name != "down":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row - 1][col]
        node_puzzle[row - 1][col] = 0
        node = Node(name="up", puzzle=tuple(np.reshape(node_puzzle, (9,))), children=[], parent=parent)
        neighbors.append(node)

    if row != len(parent_puzzle) - 1 and parent.name != "up":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row + 1][col]
        node_puzzle[row + 1][col] = 0
        node = Node(name="down", puzzle=tuple(np.reshape(node_puzzle, (9,))), children=[], parent=parent)
        neighbors.append(node)

    if col != 0 and parent.name != "right":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row][col - 1]
        node_puzzle[row][col - 1] = 0
        node = Node(name="left", puzzle=tuple(np.reshape(node_puzzle, (9,))), children=[], parent=parent)
        neighbors.append(node)

    if col != len(parent_puzzle[0]) - 1 and parent.name != "left":
        node_puzzle = parent_puzzle.copy()
        node_puzzle[row][col] = node_puzzle[row][col + 1]
        node_puzzle[row][col + 1] = 0
        node = Node(name="right", puzzle=tuple(np.reshape(node_puzzle, (9,))), children=[], parent=parent)
        neighbors.append(node)

    return neighbors


def generate_metrics(current, root):
    """ Generates metrics about the current run of astar."""
    path = reconstruct_path(current)
    visited_nodes = len(root)
    expanded_nodes = calculate_expanded_nodes(root)
    branching_factor = calculate_branching_factor(visited_nodes=visited_nodes, expanded_nodes=expanded_nodes)
    return path, visited_nodes, expanded_nodes, branching_factor


def solve(start, goal, heuristics, weights):
    """
    Checks if the given puzzle is solveable and if so, proceeds to calculate astar,
    then prints metrics as well as the chosen path.
    """
    solvable = is_solvable(start)
    if not solvable:
        print(f"The puzzle {start} is NOT solvable.")
        return False

    start_time_search = time.time()
    current, root = a_star(start=start, goal=goal, heuristics=heuristics,
                           weights=weights)
    end_time_search = time.time()

    start_time_stats = time.time()
    path, visited_nodes, expanded_nodes, branching_factor = generate_metrics(current, root)
    end_time_stats = time.time()

    if path is False:
        print("Something went wrong.")
    else:
        print_path(path)
        print(f"Heuristics used: {[heuristic.__name__ for heuristic in heuristics]}")
        print(f"Heuristics ratio:{weights}")
        print(f"The search took {end_time_search - start_time_search} seconds")
        print(f"Generating metrics took {end_time_stats - start_time_stats} seconds")
        print(f"The number of visited nodes is: {visited_nodes}")
        print(f"The number of expanded nodes is: {expanded_nodes}")
        print(f"The effective branching factor is: {branching_factor}")
        print(f"The puzzle was solved in {len(path) - 1} steps.")


def main():
    initial_state = (8, 7, 6, 0, 4, 1, 2, 5, 3)
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    solve(start=initial_state, goal=goal_state, heuristics=[manhattan_distance, hamming_distance], weights=[1.0, 0.0])


if __name__ == '__main__':
    main()
