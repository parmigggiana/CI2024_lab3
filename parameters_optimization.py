import csv
import ctypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mpltern  # noqa
import numpy as np
import pandas as pd
import tqdm
from matplotlib import cm

from custom_heuristics import improved_manhattan
from path_search import Solver
from puzzle import Board

FILENAME = "history.csv"
ITERATIONS = 874
BOARD_SIZE = 4
TIMEOUT = 10  # Some problems take too long to solve or don't converge, likely due to the heuristic being not acceptable with the given weights. Safe values on my machine are 10 for 4x4 and 300 for 5x5


def explore_parameters(iters, filename, board_size, seed=0):
    ranges = [
        (0, 1),
        (0, 1),
        (0, 1),
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
                weights = weights / np.sum(weights)

                t = executor.submit(thread_main(weights, random_board))
                futures.append(t)

            try:
                for future in as_completed(fs=futures, timeout=TIMEOUT * iters):
                    quality, cost, weights, size = future.result()
                    with open(filename, "a+") as f:
                        writer = csv.writer(f)
                        writer.writerow((*weights, quality, cost, size))
                    pbar.update(1)
            except TimeoutError:  # Just kill the running threads
                n = len([f.cancelled() for f in futures])
                executor.shutdown(wait=False)
                for t in executor._threads:
                    exc = ctypes.py_object(SystemExit)
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(t.ident), exc
                    )

    if n:
        print(f"Skipped {n} iterations due to timeout")


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
            cost_ax = fig.add_subplot(121, projection="ternary")
            cost_ax.set_title(label="Cost")
            cost_ax.set_llabel("Conflicts")
            cost_ax.set_rlabel("Inversions")
            cost_ax.set_tlabel("Manhattan")
            cost_ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )
            cost_ax.scatter(
                filtered_history.iloc[:, 0],
                filtered_history.iloc[:, 1],
                filtered_history.iloc[:, 2],
                c=filtered_history.iloc[:, 4],
                cmap=cm.Spectral,
                alpha=0.8,
            )

            qual_ax: plt.axes = fig.add_subplot(122, projection="ternary")
            qual_ax.set_title("Quality")
            qual_ax.set_llabel("Conflicts")
            qual_ax.set_rlabel("Inversions")
            qual_ax.set_tlabel("Manhattan")
            qual_ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )

            img = qual_ax.scatter(
                filtered_history.iloc[:, 0],
                filtered_history.iloc[:, 1],
                filtered_history.iloc[:, 2],
                c=filtered_history.iloc[:, 3],
                cmap=cm.Spectral,
                alpha=0.8,
            )

            fig.colorbar(
                mappable=img, ax=[qual_ax, cost_ax], location="bottom", shrink=0.6
            )

            # Event handler for clicking on either plot
            def on_click(event):
                if event.inaxes in [
                    qual_ax,
                    cost_ax,
                ]:  # Check if the click is on either subplot
                    ax = event.inaxes  # Determine which subplot was clicked
                    # Get ternary coordinates from Cartesian (mpltern provides this automatically)
                    a, b, c = ax.transProjection.inverted().transform(
                        [event.xdata, event.ydata]
                    )
                    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:  # Valid point check
                        fig.suptitle(
                            f"Mahattan: {a:.2f}\nConflicts: {b:.2f}\nInversions: {c:.2f}",
                            fontsize=12,
                        )
                        fig.canvas.draw()  # Redraw the plot to update the annotations

            # Connect the event to the handler
            fig.canvas.mpl_connect("button_press_event", on_click)

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
