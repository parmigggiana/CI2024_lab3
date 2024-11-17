"""
Solve lab3 by BFS
"""

from time import sleep
import numpy as np

from puzzle import Action, Board

SIZE = 3


def main():
    explored_nodes = set()

    solution = np.arange(SIZE**2)
    solution = np.roll(solution, -1).reshape(SIZE, SIZE)

    frontier = []
    cost = 0

    p = Board(SIZE, 0)  # get a radom board
    frontier.append(p)
    print(p)
    while not np.array_equal(p.m, solution) and frontier:
        p = frontier.pop(0)
        explored_nodes.add(p)
        cost += 1
        valid_actions = np.nonzero(p.valid_actions)[0]
        for action in valid_actions:

            new: Board = p.act(Action(action))
            if new not in explored_nodes and new not in frontier:
                frontier.append(new)

        # print()
        # print(f"{cost = }")
        # print(f"{p = }")
        # print(f"{frontier = }")
        # print(f"{explored_nodes = }")
        # sleep(1)
    print(cost)
    print(p.quality)
    print(p)
    print(explored_nodes)
    print(Board(solution))
    print(Board(solution) in explored_nodes)


if __name__ == "__main__":
    main()
