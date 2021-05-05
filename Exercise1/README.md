# 8 Puzzle  
A python3 implementation to calculate solutions for the 8 puzzle using the A* algorithm  

## Table of Contents  
* General Info
* Team Members
* Modules
* Heuristics
* Functions

## General Info  
This project is the first exercise of the study course Introduction to AI and Data Science. By using a A* algorithm and two different heuristic functions a random generated 8 puzzle has to be solved (if solvable). The following parameters have to be calculated and displayed:  
* Branching factor
* expanded nodes
* weighting of the heuristic functions  

## Team Members  
* Dominik Hackl
* Harrys Luigi Maria Kavan
* Thomas Scheibelhofer
* Matthias Schmid-Kietreiber  

## Modules  
* numpy
* PriorityQueue
* random
* time
* multiprocessing
* jsonpickle
* datetime
* matplotlib.pyplot
* pandas  

## Heuristics  
The Requirement of the exercise was to use at least two heuristics and to use them on their own and use them combined. The following heuristic functions have been chosen:  

* **Manhattan Distance :** For each cell of the puzzle the distance to their goal state is calculated.  
* **Hemming Distance:** Calculates the number of misplaced puzzle pieces.  

## Functions  
### a_star(start, goal, heuristics, weights)  
This function implements the A* algorithm and is based on the pseudo code found on Wikipedia and pushes the neighbor nodes of the current node into a priority queue in which they are prioritized by their costs calculated by the heuristic functions. When the node on top of the queue is the goal node the solution has been found.  

### reconstruct_path(node, expanded_nodes)  
This function reconstructs the path by going through the parent nodes.  

### print_path(steps)  
Prints the traversed nodes which lead to the solution.  

### manhattan_distance(start, goal)  
This function calculates the sum of the distances of each puzzle piece to its goal state.  

### hamming_distance(start, goal)  
This function calculates the number of misplaced puzzle pieces.  

### is_solvable(start)  
This function checks whether the puzzle is solvable based on inversions.  

### get_neighbors(parent)  
This function is used in the a_star function and calculates the neighbors of the puzzle based on it's current position of its pieces.  

### solve(start, goal, heuristics, weights)  
This function is the main function of the program and uses the is_solvable, the a_star and the print_path functions to solve the 8 puzzle and returns if the puzzle is solvable and if it is the time taken, the expanded nodes and the path to the solution.  

