"""
Solve lab3 by BFS
"""

from path_search import BFSSolver, DFSSolver
from puzzle import Board

SIZE = 3


def print_path(path):
    # This is a helper function
    # Based on SIZE, print the path in a more readable way
    # For example, if SIZE = 3, the path [[[1, 2, 3], [4, 5, 6], [7, 0, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 0]]] will be printed as:
    # 1 2 3      1 2 3
    # 4 5 6  ->  4 5 6
    # 7 0 8      7 8 0
    s = ["" for _ in range(SIZE)]
    for node in path:
        for i in range(SIZE):
            if i == SIZE // 2:
                sep = "  ->  "
            else:
                sep = "      "
            s[i] += " ".join(map(str, node[i])) + sep

    print("\n".join([line[:-6] for line in s]))


def main():
    # p = Board(SIZE, 0)  # get a radom board, unsolvable with DFS or BFS
    # p = Board([[1, 2, 3], [4, 0, 5], [7, 8, 6]])  # very easy test case
    p = Board([[1, 2, 3], [4, 8, 5], [7, 0, 6]])  # hard for DFS, easy for BFS

    path, quality, cost = BFSSolver(p).run()
    print("Found a solution with BFS")
    print(f"Explored {cost} nodes")
    print(f"Solution reached in {quality} steps")
    print_path(path)
    print()

    path, quality, cost = DFSSolver(p).run()
    print("Found a solution with DFS")
    print(f"Explored {cost} nodes")
    print(f"Solution reached in {quality} steps")
    print_path(path)


if __name__ == "__main__":
    main()
