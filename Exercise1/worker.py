import multiprocessing as mp

from statistics import start_run, write_file, initialize


def compute(data):
    # Put the call to the A* in here.
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    return start_run(start=data[0], weights=data[1], goal=goal_state)


def process_array(array_with_weights, runs):
    p = mp.Pool()
    res = p.map(compute, array_with_weights)
    write_file(res, runs)


if __name__ == "__main__":
    runs = 10
    work_to_do = initialize(runs)
    process_array(work_to_do, runs)
