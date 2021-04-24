import multiprocessing as mp

from statistics import initialize, start_run, write_file


def init():
    print("starting worker")


def compute(data):
    # Put the call to the A* in here.
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    return start_run(start=data[0], weights=data[1], goal=goal_state)


if __name__ == "__main__":
    runs = 100
    work_to_do = initialize(runs)
    p = mp.Pool(initializer=init)
    res = p.map(compute, work_to_do)
    write_file(res, runs)
