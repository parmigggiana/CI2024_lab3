"""
Solve lab3
"""

from custom_heuristics import improved_manhattan
from path_search import Solver
from puzzle import Board

SEED = 0

instances = [
    {
        "starting_board": Board(3, SEED),
        "algorithm": "astar",
        "heuristic": "manhattan",
        "plot": False,
    },
    {
        "starting_board": Board(3, SEED),
        "algorithm": "astar",
        "heuristic": improved_manhattan(3),
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


if __name__ == "__main__":
    main()
