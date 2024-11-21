import csv
import ctypes
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
ITERATIONS = 20000
BOARD_SIZE = 3
TIMEOUT = 1  # Some problems take too long to solve or don't converge, likely due to the heuristic being not acceptable with the given weights. Safe values on my machine are 5 for 4x4 and 30 for 5x5


def process_main(weights, board, name):
    instance = {
        "starting_board": board,
        "algorithm": "astar",
        "heuristic": improved_manhattan(board.size, weights),
        "plot": False,
    }
    _, quality, cost = Solver(**instance).run()
    return quality, cost, weights, board.size, name


def explore_parameters(samples, filename, board_size, timeout=None):
    ranges = [
        (0, 1),
        (0, 1),
        (0, 1),
    ]
    futures: list = []
    n = None
    with tqdm.tqdm(
        desc=f"Trying {samples} random weights", total=samples * board_size
    ) as pbar:
        with ProcessPoolExecutor() as executor:
            for i in range(samples):
                weights: list[float] = [np.random.uniform(*r) for r in ranges]
                weights = weights / np.sum(weights)

                for _ in range(
                    board_size
                ):  # higher board size has more variability, so we try each weight set a bit more times to average out the board's difficulty
                    random_board = Board(board_size)
                    f = executor.submit(
                        process_main, weights=weights, board=random_board, name=i
                    )
                    futures.append(f)

            try:
                for future in as_completed(fs=futures, timeout=timeout * samples):
                    quality, cost, weights, size, name = future.result()
                    with open(filename, "a+") as f:
                        writer = csv.writer(f)
                        writer.writerow((*weights, quality, cost, size, name))
                    pbar.update(1)
            except TimeoutError:  # Just kill the running threads
                n = len([f.cancelled() for f in futures])
                executor.shutdown(wait=False)
                for f in executor._processes:
                    exc = ctypes.py_object(SystemExit)
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(f.ident), exc
                    )

    if n:
        print(f"Skipped {n} iterations due to timeout")


def plot_history(filename, board_size=None):
    with open(filename, "r") as f:
        history = pd.read_csv(
            f, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: int, 6: str}
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
        filtered_history = filtered_history.groupby("Name").mean()
        if (
            size == board_size
            or (isinstance(board_size, Iterable) and size in board_size)
            or board_size is None
        ):
            fig = plt.figure(size, figsize=(25, 22))
            # fig.suptitle("Lower is better")
            # fig.supxlabel(f"{filtered_history.shape[0]} samples")
            fig.text(
                0.5,
                0.7,
                f"{filtered_history.shape[0]} samples",
                ha="center",
                va="center",
            )
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

            cb = fig.colorbar(
                mappable=img, ax=[qual_ax, cost_ax], location="bottom", shrink=0.6
            )
            cb.set_label("Lower is better")
            point = fig.text(
                0.5,
                0.5,
                "",
                ha="center",
                va="center_baseline",
                fontsize=12,
                weight="bold",
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
                        # Change the text in variable `point`
                        point.set_text(
                            f"Mahattan: {a:.2f}\nConflicts: {b:.2f}\nInversions: {c:.2f}"
                        )
                        fig.canvas.draw()

            # Connect the event to the handler
            fig.canvas.mpl_connect("button_press_event", on_click)
            plt.tight_layout(rect=[0.05, 0.27, 0.95, 0.95])
            plt.savefig(
                f"plots/{board_size}x{board_size}.jpg",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.3,
            )
    # plt.show()


if __name__ == "__main__":
    if not any(x in ["-s", "--skip", "SKIP"] for x in sys.argv):
        if not Path(FILENAME).exists():
            with open(FILENAME, "w") as f:
                csv.writer(f).writerow(
                    [
                        "Manhattan_W",
                        "Conflicts_W",
                        "Inversions_W",
                        "Quality",
                        "Cost",
                        "Size",
                        "Name",
                    ]
                )
        explore_parameters(
            filename=FILENAME,
            samples=ITERATIONS,
            board_size=BOARD_SIZE,
            timeout=TIMEOUT,
        )

    if any(x in ["-p", "--plot", "PLOT"] for x in sys.argv):
        plot_history(FILENAME, board_size=BOARD_SIZE)
