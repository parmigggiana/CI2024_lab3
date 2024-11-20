import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Iterable

import matplotlib.pyplot as plt
from custom_heuristics import improved_manhattan
import numpy as np
import pandas as pd
from path_search import Solver
from puzzle import Board
import tqdm
from matplotlib import cm

FILENAME = "history.csv"
ITERATIONS = 100
BOARD_SIZE = 4


def explore_parameters(iters, filename, board_size, seed=0):
    ranges = [
        (0.5, 2),
        (0, 2),
        (0, 2),
    ]
    size = board_size

    futures: list = []
    with tqdm.tqdm(desc=f"Solving {iters} random problems", total=iters) as pbar:
        with ThreadPoolExecutor() as executor:
            for i in range(iters):
                if i % 5 == 0:
                    if board_size is None:
                        size = np.random.randint(3, 6)
                    random_board = Board(size, seed)

                weights: list[float] = [np.random.uniform(*r) for r in ranges]

                t = executor.submit(thread_main(weights, random_board))
                futures.append(t)

            for future in as_completed(fs=futures):
                quality, cost, weights, size = future.result()
                with open(filename, "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow((*weights, quality, cost, size))
                pbar.update(1)


def plot_history(filename, board_size=None):
    with open(filename, "r") as f:
        history = pd.read_csv(
            f, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: int}
        )

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
        filtered_history = history.loc[history.iloc[:, 5] == size]

        if (
            size == board_size
            or (isinstance(board_size, Iterable) and size in board_size)
            or board_size is None
        ):
            fig = plt.figure(size)
            fig.suptitle("Lower is better")
            fig.supxlabel(f"{filtered_history.shape[0]} samples")
            fig.canvas.manager.set_window_title(f"{size}x{size} board")
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

            cost_ax.scatter(
                filtered_history.iloc[:, 0],
                filtered_history.iloc[:, 1],
                filtered_history.iloc[:, 2],
                c=filtered_history.iloc[:, 4],
                cmap=cm.Spectral,
            )
            img = qual_ax.scatter(
                filtered_history.iloc[:, 0],
                filtered_history.iloc[:, 1],
                filtered_history.iloc[:, 2],
                c=filtered_history.iloc[:, 3],
                cmap=cm.Spectral,
            )
            fig.colorbar(
                mappable=img, ax=[qual_ax, cost_ax], location="bottom", shrink=0.6
            )

    plt.show()


def thread_main(weights, board):
    instance = {
        "starting_board": board,
        "algorithm": "astar",
        "heuristic": improved_manhattan(board.size, weights),
        "plot": False,
    }

    def run_solver():
        _, quality, cost = Solver(**instance).run()
        return quality, cost, weights, board.size

    return run_solver


def run_tests(filename, iterations, board_size):
    if not Path(filename).exists():
        with open(filename, "w") as f:
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
    explore_parameters(
        filename=filename, iters=iterations, board_size=board_size, seed=time.time_ns()
    )


if __name__ == "__main__":
    if not any(x in ["-s", "--skip", "SKIP"] for x in sys.argv):
        run_tests(
            FILENAME,
            ITERATIONS,
            BOARD_SIZE,
        )
    if any(x in ["-p", "--plot", "PLOT"] for x in sys.argv):
        plot_history(FILENAME, board_size=BOARD_SIZE)
