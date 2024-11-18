import numpy as np

from puzzle import Board


def improved_manhattan(board_size, weights=[1, 1, 1]):
    solution = np.arange(board_size**2)
    solution = np.roll(solution, -1).reshape(board_size, board_size)
    solution = Board(solution)

    def manhattan(board: Board):
        return board.manhattan_distance(solution)

    def conflicts(board: Board):
        conflicts = 0
        for i in range(board.size):
            for j in range(board.size):
                if board[i][j] == 0:
                    continue
                for k in range(i, board.size):
                    for l in range(j, board.size):
                        if board[k][l] == 0:
                            continue
                        if board[i][j] > board[k][l] and (i == k or j == l):
                            conflicts += 1
        return conflicts

    def inversions(board: Board):
        inversions = 0
        for i in range(board.size):
            for j in range(board.size):
                if board[i][j] == 0:
                    continue
                for k in range(i, board.size):
                    for l in range(j, board.size):
                        if board[k][l] != 0 and board[i][j] > board[k][l]:
                            inversions += 1
        return inversions

    return (
        lambda board: manhattan(board) * weights[0]
        + conflicts(board) * weights[1]
        + inversions(board) * weights[2]
    )
