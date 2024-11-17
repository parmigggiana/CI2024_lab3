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
            if board < 3:
                raise ValueError("Board size must be at least 3x3")
            board = np.arange(board**2).reshape(board, board)
            board = rng.permutation(board)
        elif isinstance(board, list) and (
            not isinstance(board[0], list)
            or not all(len(row) == len(board) for row in board)
        ):
            raise ValueError("Board must be square")

        if isinstance(board, list):
            board = np.array(board)

        self.m = board
        self.size = board.shape[0]
        self.position = tuple(x[0] for x in np.nonzero(board == 0))
        self.valid_actions = self._get_valid_actions()
        self.quality = 0

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
        b.quality = self.quality + 1
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
