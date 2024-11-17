from enum import Enum
from typing import Self

import numpy as np


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DIRECTIONS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


class InvalidMoveError(Exception):
    pass


class Board:
    def __init__(self, board: np.ndarray | list[list[int]] | int, seed=0):
        rng = np.random.Generator(np.random.PCG64(seed))
        if isinstance(board, int):
            if board < 2:
                raise ValueError("Board size must be at least 2x2")
            self.size = board
            self.m = None
            while (
                self.m is None
            ):  # Randomly generate a board until a solvable one is found
                self.m = rng.permutation(self.size**2).reshape(self.size, self.size)
                self.position = tuple(x[0] for x in np.nonzero(self.m == 0))
                if not self.solvable():
                    self.m = None
        elif isinstance(board, list) and (
            not isinstance(board[0], list)
            or not all(len(row) == len(board) for row in board)
        ):
            raise ValueError("Board must be square")
        elif isinstance(board, list):
            self.size = len(board)
            self.m = np.array(board)
            self.position = tuple(x[0] for x in np.nonzero(self.m == 0))
        elif isinstance(board, np.ndarray):
            self.size = board.shape[0]
            self.m = board
            self.position = tuple(x[0] for x in np.nonzero(self.m == 0))
        else:
            raise ValueError("Invalid board")

        self.valid_actions = self._get_valid_actions()

    def _get_valid_actions(self) -> list[Action]:
        actions = np.zeros((4,), dtype=bool)

        if self.position[0] != 0:
            actions[Action.UP.value] = True

        if self.position[0] != self.size - 1:
            actions[Action.DOWN.value] = True

        if self.position[1] != 0:
            actions[Action.LEFT.value] = True

        if self.position[1] != self.size - 1:
            actions[Action.RIGHT.value] = True

        return actions

    def act(self, action: Action) -> Self:
        if not self.valid_actions[action.value]:
            raise InvalidMoveError(f"Trying to move {action} in {self}")

        new_pos = (
            self.position[0] + DIRECTIONS[action][0],
            self.position[1] + DIRECTIONS[action][1],
        )

        new_board = self.m.copy()
        new_board[self.position], new_board[new_pos] = (
            new_board[new_pos],
            new_board[self.position],
        )
        b = Board(new_board)
        return b

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.m, other.m)

    def __hash__(self):
        return hash(self.m.tobytes())

    # def __str__(self):
    #     s = ""
    #     for row in self.m:
    #         s += str(row)[1:-1]
    #         s += "\n"
    #     return s[:-1]

    def __str__(self):
        return str(self.m)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.m)

    def __getitem__(self, key):
        return self.m[key]

    def manhattan_distance(self, other: Self) -> int:
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.m[i][j] != 0:
                    x, y = np.nonzero(other.m == self.m[i][j])
                    x, y = x[0], y[0]
                    distance += abs(x - i) + abs(y - j)
        return distance

    def hamming_distance(self, other: Self) -> int:
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.m[i][j] != other.m[i][j]:
                    distance += 1
        return distance

    def solvable(self):
        inversions = 0
        for i in range(self.size**2 - 1):
            x, y = divmod(i, self.size)
            for j in range(i + 1, self.size**2):
                a, b = divmod(j, self.size)
                if self[a][b] != 0 and self[x][y] != 0 and self.m[x][y] > self.m[a][b]:
                    inversions += 1
        if self.size % 2 == 0:
            inversions += self.position[0] + 1
        return inversions % 2 == 0
