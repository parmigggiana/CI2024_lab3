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
            board = np.arange(board**2).reshape(board, board)
            board = rng.permutation(board)

        if isinstance(board, list):
            board = np.array(board)

        self.m = board
        self.size = board.shape[0]
        self.position = tuple(x[0] for x in np.nonzero(board == 0))
        self.valid_actions = self._get_valid_actions()

    def _get_valid_actions(self) -> list[Action]:
        actions = np.zeros((4,), dtype=bool)

        if self.position[0] != 0:
            actions[Action.UP.value] = True

        if self.position[1] != 0:
            actions[Action.LEFT.value] = True

        if self.position[0] != self.size - 1:
            actions[Action.DOWN.value] = True

        if self.position[1] != self.size - 1:
            actions[Action.RIGHT.value] = True

        return actions

    def act(self, action: Action) -> Self:
        if not self.valid_actions[action.value]:
            raise InvalidMoveError

        new_pos = (
            self.position[0] + DIRECTIONS[action][0],
            self.position[1] + DIRECTIONS[action][1],
        )

        new_board = self.m.copy()
        new_board[self.position], new_board[new_pos] = (
            new_board[new_pos],
            new_board[self.position],
        )
        return Board(new_board)

    def __hash__(self):
        return hash(self.m.tobytes())

    def __str__(self):
        return str(self.m)

    def __repr__(self):
        return str(self)
