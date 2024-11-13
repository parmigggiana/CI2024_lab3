import numpy as np

from puzzle import Action, Board

SIZE = 3


def main():
    solution = np.arange(SIZE**2)
    solution = np.roll(solution, -1).reshape(SIZE, SIZE)

    p = Board(SIZE, 0)  # get a radom board
    print(p)
    while True:
        action = Action(np.random.choice(np.nonzero(p.valid_actions)[0]))
        print()
        print(action)
        p = p.act(action)
        print(p)


if __name__ == "__main__":
    main()
