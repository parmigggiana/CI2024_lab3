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
    {
        "starting_board": Board(4, SEED),
        "algorithm": "astar",
        "heuristic": "manhattan",
        "plot": False,
    },
    {
        "starting_board": Board(4, SEED),
        "algorithm": "astar",
        "heuristic": improved_manhattan(4),
        "plot": False,
    },
    {
        "starting_board": Board(5, SEED),
        "algorithm": "astar",
        "heuristic": "manhattan",
        "plot": False,
    },
    {
        "starting_board": Board(5, SEED),
        "algorithm": "astar",
        "heuristic": improved_manhattan(5),
        "plot": False,
    },
    {
        "starting_board": Board(6, SEED),
        "algorithm": "astar",
        "heuristic": improved_manhattan(6),
        "plot": False,
    },
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
