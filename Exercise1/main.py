from copy import deepcopy
from typing import List

from anytree import Node, RenderTree

# https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

initial_state = [[7, 2, 4], [5, None, 6], [8, 3, 1]]
goal_state = [[None, 1, 2], [3, 4, 5], [6, 7, 8]]


def count_misplaced(current_state: List[List[int]], goal: List[List[int]]):
    misplaced = 0
    for i, array in enumerate(current_state):
        for j, number in enumerate(array):
            if current_state[i][j] != goal[i][j]:
                misplaced += 1
    return misplaced


def manhattan_distance(current_state: List[List[int]], goal: List[List[int]]):
    distance = 0
    for i, array in enumerate(current_state):
        for j, number in enumerate(array):
            (row, column) = find_index(x=number, goal=goal)
            distance += abs(i - row) + abs(j - column)
    return distance


def find_index(x, goal: List[List[int]]):
    for row, i in enumerate(goal):
        try:
            column = i.index(x)
        except ValueError:
            continue
        return row, column
    return -1


def build_tree(current_state, depth, root=None, parent=None):
    (row, col) = find_index(None, goal=current_state)
    if root is None:
        root = Node("root", array=current_state)
    if parent is None:
        parent = root

    def add_up():
        node = deepcopy(current_state)
        node[row][col] = node[row - 1][col]
        node[row - 1][col] = None
        return node

    def add_down():
        node = deepcopy(current_state)
        node[row][col] = node[row + 1][col]
        node[row + 1][col] = None
        return node

    def add_left():
        node = deepcopy(current_state)
        node[row][col] = node[row][col - 1]
        node[row][col - 1] = None
        return node

    def add_right():
        node = deepcopy(current_state)
        node[row][col] = node[row][col + 1]
        node[row][col + 1] = None
        return node

    if row != 0 and parent.name != "down":
        Node("up", parent=parent, array=add_up())

    if row != len(current_state) - 1 and parent.name != "up":
        Node("down", parent=parent, array=add_down())

    if col != 0 and parent.name != "right":
        Node("left", parent=parent, array=add_left())

    if col != len(current_state[0]) - 1 and parent.name != "left":
        Node("right", parent=parent, array=add_right())

    for child in parent.children:
        if parent.depth < depth:
            build_tree(current_state=child.array, depth=depth, root=root, parent=child)

    return root


distance = manhattan_distance(current_state=initial_state, goal=goal_state)
print(f"Manhattan Distance: {distance}")

misplaced = count_misplaced(current_state=initial_state, goal=goal_state)
print(f"Misplaces pieces: {misplaced}")

tree = build_tree(current_state=initial_state, depth=9)
for pre, fill, node in RenderTree(tree):
    print("%s%s" % (pre, node.name))



# from ida import iterative_deepening_a_star
# iterative_deepening_a_star(0, heuristic=manhattan_distance, start=initial_state, goal=goal_state)
