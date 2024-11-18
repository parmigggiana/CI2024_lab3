"""
Solve lab3
"""

import csv
from pathlib import Path

import pandas as pd
from matplotlib import cm
from math import sqrt
from tqdm import trange

from custom_heuristics import improved_manhattan
from path_search import Solver
from puzzle import Board

SIZE = 4
HISTORY_PATH = "history.txt"
FORCE_HYPERPARAMETERS_SEARCH = True
PLOT = True


random_board = Board(SIZE, 0)
instances = [
    {  # 6x6 is very slow, 5x5 is mostly reasonable
        "starting_board": random_board,
        "algorithm": "astar",
        "heuristic": "manhattan",
        "plot": False,
    },
    {  # very rarely worse than manhattan, most of the time finds a ~2x better solution in ~2x the speed
        "starting_board": random_board,
        "algorithm": "astar",
        "heuristic": improved_manhattan(SIZE, (1, 1 / sqrt(SIZE), 0.5 / sqrt(SIZE))),
        "plot": False,
    },
    # {  # can handle up to random 4x4 in reasonable time (~10x slower than manhattan and ~6x worse solution)
    #     "starting_board": random_board,
    #     "algorithm": "astar",
    #     "heuristic": "hamming",
    #     "plot": False,
    # },
    # { # Quite slower for random boards, probably can't handle anything beyond 3x3
    #     "starting_board": random_board,
    #     "algorithm": "astar",
    #     "heuristic": "dijkstra",
    #     "plot": False,
    # },
    # { # Unusable for random boards
    #     "starting_board": Board([[1, 2, 3], [4, 0, 5], [7, 8, 6]]),
    #     "algorithm": "bfs",
    #     "plot": False,
    # },
    # { # Unusable for random boards
    #     "starting_board": Board([[1, 2, 3], [4, 0, 5], [7, 8, 6]]),
    #     "algorithm": "dfs",
    #     "plot": False,
    # },
]


def print_path(path):
    size = path[0].size
    if (len(path)) > 40:
        # Just print first and last 20 steps
        path = path[:20] + path[-20:]
    max_digits = len(str(size**2))
    s = ["" for _ in range(size)]
    for count, node in enumerate(path):
        for i in range(size):
            if i == size // 2:
                sep = "   ->  "
                if count == 20:
                    sep = "  ...  "
            else:
                sep = "       "
            s[i] += " ".join(f"{num:>{max_digits}}" for num in node[i]) + sep

    while len(s[0]) > 80:
        for i in range(size):
            print(s[i][:80])
            s[i] = s[i][80:]
        print()
    for i in range(size):
        print(s[i][:-6])


def main():
    for instance in instances:
        print(
            f"Running {instance['algorithm']}"
            + (
                f" with {instance['heuristic']}"
                if instance["algorithm"] == "astar"
                else ""
            )
        )

        path, quality, cost = Solver(**instance).run()

        print(f"Explored {cost} nodes")
        print(f"Solution reached in {quality} steps")
        print_path(path)
        print()


def find_best_hyperparameters():
    import numpy as np

    ranges = [
        (1, 2),
        (0, 2),
        (0, 2),
    ]
    # TODO add multiprocessing
    print("Starting hyperparameters search - press Ctrl+C to stop")
    try:
        for i in trange(50):
            if i % 10 == 0:
                random_board = Board(np.random.randint(3, 6), 42)
            weights = [np.random.uniform(*r) for r in ranges]
            instance = {
                "starting_board": random_board,
                "algorithm": "astar",
                "heuristic": improved_manhattan(random_board.size, weights),
                "plot": False,
            }
            _, quality, cost = Solver(**instance).run()

            with open(HISTORY_PATH, "a+") as f:
                writer = csv.writer(f)
                writer.writerow((*weights, quality, cost, random_board.size))
    except KeyboardInterrupt:
        print("Hyperparameters search stopped")


def plot_hyperparameters(history: pd.DataFrame):
    import matplotlib.pyplot as plt

    # Normalize the data separately for each Size
    for size in history.iloc[:, 5].unique():
        # Ignore weights, only normalize columns 3 and 4
        history.loc[history.iloc[:, 5] == size, ["Quality", "Cost"]] = (
            history.loc[history.iloc[:, 5] == size, ["Quality", "Cost"]]
            - history.loc[history.iloc[:, 5] == size, ["Quality", "Cost"]].min()
        ) / (
            history.loc[history.iloc[:, 5] == size, ["Quality", "Cost"]].max()
            - history.loc[history.iloc[:, 5] == size, ["Quality", "Cost"]].min()
        )

    fig = plt.figure()
    cost_ax: plt.axes = fig.add_subplot(121, projection="3d")
    cost_ax.set_title("Cost")
    cost_ax.set_xlabel("Manhattan")
    cost_ax.set_ylabel("Conflicts")
    cost_ax.set_zlabel("Inversions")

    qual_ax: plt.axes = fig.add_subplot(122, projection="3d")
    qual_ax.set_title("Quality")
    qual_ax.set_xlabel("Manhattan")
    qual_ax.set_ylabel("Conflicts")
    qual_ax.set_zlabel("Inversions")

    img1 = qual_ax.scatter(
        history.iloc[:, 0],
        history.iloc[:, 1],
        history.iloc[:, 2],
        c=history.iloc[:, 3],
        cmap=cm.viridis,
    )
    img2 = cost_ax.scatter(
        history.iloc[:, 0],
        history.iloc[:, 1],
        history.iloc[:, 2],
        c=history.iloc[:, 4],
        cmap=cm.viridis,
    )
    fig.colorbar(img1, ax=qual_ax)
    fig.colorbar(img2, ax=cost_ax)
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    if not Path(HISTORY_PATH).exists():
        with open(HISTORY_PATH, "w") as f:
            csv.writer(f).writerow(
                [
                    "Manhattan_W",
                    "Conflicts_W",
                    "Inversions_W",
                    "Quality",
                    "Cost",
                    "Size",
                ]
            )
        find_best_hyperparameters()
    elif FORCE_HYPERPARAMETERS_SEARCH:
        find_best_hyperparameters()

    with open(HISTORY_PATH, "r") as f:
        history = pd.read_csv(
            f, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: int}
        )

    if PLOT:
        plot_hyperparameters(history)
