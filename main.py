"""
Solve lab3
"""

from path_search import Solver
from puzzle import Board

random_board = Board(3, 0)
instances = [
    {
        "starting_board": random_board,
        "algorithm": "astar",
        "heuristic": "manhattan",
        "plot": False,
    },
    {
        "starting_board": random_board,
        "algorithm": "astar",
        "heuristic": "hamming",
        "plot": False,
    },
    {
        "starting_board": random_board,
        "algorithm": "astar",
        "heuristic": "dijkstra",
        "plot": False,
    },
    # {
    #     "starting_board": random_board,
    #     "algorithm": "bfs",
    #     "plot": False,
    # },
    # {
    #     "starting_board": random_board,
    #     "algorithm": "dfs",
    #     "plot": False,
    # },
]


def print_path(path):
    size = path[0].size
    s = ["" for _ in range(size)]
    for node in path:
        for i in range(size):
            if i == size // 2:
                sep = "  ->  "
            else:
                sep = "      "
            s[i] += " ".join(map(str, node[i])) + sep

    print("\n".join([line[:-6] for line in s]))


def main():
    # random board, unsolvable with DFS or BFS
    # p = Board(SIZE, 0)  # get a radom board, unsolvable with DFS or BFS

    # easy test case
    # p = Board([[1, 2, 3], [4, 0, 5], [7, 8, 6]])  # very easy test case

    # easy for BFS and A*, unsolvable with DFS
    p = Board([[1, 2, 3], [4, 8, 5], [7, 0, 6]])

    for instance in instances:
        print(
            f"Running {instance['algorithm']}"
            + (
                f" with {instance['heuristic']}"
                if instance["algorithm"] == "astar"
                else ""
            )
        )
        # print("Starting board:")
        # print_path(
        #     [
        #         p,
        #     ]
        # )
        path, quality, cost = Solver(**instance).run()
        print(f"Explored {cost} nodes")
        print(f"Solution reached in {quality} steps")
        print_path(path)
        print()


if __name__ == "__main__":
    main()
