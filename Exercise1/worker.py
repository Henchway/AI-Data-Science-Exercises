import multiprocessing as mp
from datetime import datetime

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
    runs = 50

    start_time = datetime.now()
    print(f"Start: {start_time}")

    work_to_do = initialize(runs)
    process_array(work_to_do, runs)

    end_time = datetime.now()
    print(f"End: {end_time}")
    print(f"Duration: {end_time - start_time}")
