import numpy as np

from puzzle import Board


#   0 1 2
# 0 1 2 3
# 1 4 5 6
# 2 7 8 9
def improved_manhattan(board_size, weights=None):
    solution = np.arange(board_size**2)
    solution = np.roll(solution, -1).reshape(board_size, board_size)
    solution = Board(solution)

    if weights is None:
        match board_size:
            case 3:
                weights = [1, 0, 2]
            case 4:
                weights = [0.9, 0.25, 1.5]
            case 5:
                weights = [1, 1, 1]

    def manhattan(board: Board):
        return board.manhattan_distance(solution)

    def linear_conflicts(board: Board):
        size = board.size
        conflicts = 0

        # Check for row conflicts
        for row in range(size):
            for col in range(size):
                tile = board[row][col]
                if tile == 0:
                    continue

                # Check if the tile is in the correct row
                if (tile - 1) // size == row:
                    for i in range(col + 1, size):
                        other_tile = board[row][i]
                        if other_tile == 0:
                            continue
                        if (other_tile - 1) // size == row and tile > other_tile:
                            conflicts += 1

                # Check if the tile is in the correct column
                if tile % size == col - 1:
                    for i in range(row + 1, size):
                        other_tile = board[i][col]
                        if other_tile == 0:
                            continue
                        if other_tile % size == col and tile > other_tile:
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
