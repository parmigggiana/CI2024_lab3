import numpy as np

from puzzle import Board


#   0 1 2
# 0 1 2 3
# 1 4 5 6
# 2 7 8 9
def improved_manhattan(board_size, weights=[1, 1, 1]):
    solution = np.arange(board_size**2)
    solution = np.roll(solution, -1).reshape(board_size, board_size)
    solution = Board(solution)

    def manhattan(board: Board):
        return board.manhattan_distance(solution)

    def linear_conflicts(board: Board):
        conflicts = 0
        for row in range(board.size):
            for column in range(board.size):
                if board[row][column] == 0:
                    continue
                # check if board[i][j] is in the correct column
                if board[row][column] % board.size == column - 1:
                    for i in range(row, board.size):
                        if board[i][column] == 0:
                            continue
                        if (
                            board[row][column] % board.size != column - 1
                            and board[row][column] > board[i][column]
                        ):
                            conflicts += 1

                # check if board[i][j] is in the correct row
                if (board[row][column] - 1) // board.size == row:
                    for i in range(column, board.size):
                        if board[row][i] == 0:
                            continue
                        if (board[row][column] - 1) // board.size != row and board[row][
                            column
                        ] > board[row][i]:
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
        + linear_conflicts(board) * weights[1]
        + inversions(board) * weights[2]
    )
