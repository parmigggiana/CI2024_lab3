import numpy as np
from puzzle import Action, Board
from graph import Graph
from priorityqueue import PriorityQueue


class Solver:
    def __init__(
        self,
        starting_board: Board,
        algorithm: str = "astar",
        heuristic="manhattan",
        plot=False,
    ):
        algorithm = algorithm.lower()

        assert algorithm in ["bfs", "dfs", "astar"]
        if algorithm == "astar":
            assert callable(heuristic) or heuristic in [
                "manhattan",
                "hamming",
                "dijkstra",
            ]
        assert starting_board.solvable(), "Puzzle is not solvable"

        self.board = starting_board
        self.solution = np.arange(self.board.size**2)
        self.solution = Board(
            np.roll(self.solution, -1).reshape(self.board.size, self.board.size)
        )
        self.graph = Graph()
        self.algorithm = algorithm
        self.plot = plot

        if heuristic:
            match heuristic.lower():
                case "manhattan":
                    self.heuristic = lambda x: x.manhattan_distance(self.solution)
                case "hamming":
                    self.heuristic = lambda x: x.hamming_distance(self.solution)
                case "dijkstra":
                    self.heuristic = lambda x: 0
                case _:
                    self.heuristic = heuristic

    def run(self):
        frontier = PriorityQueue()
        explored_nodes = set()

        current_board = self.board
        self.cost = 0

        frontier.push(current_board, 0)
        self.graph.add_node(current_board)
        while frontier:
            if current_board == self.solution:
                break
            current_board = frontier.pop()
            explored_nodes.add(current_board)
            valid_actions = np.nonzero(current_board.valid_actions)[0]
            self.cost += 1
            for action in valid_actions:
                new: Board = current_board.act(Action(action))
                if new not in explored_nodes and new not in frontier:
                    priority = self.get_priority(new)
                    frontier.push(new, priority)
                    self.graph.add_node(new, current_board)
                self.graph.add_edge(current_board, new)
        else:
            raise ValueError("No solution found")

        if self.plot:
            self.graph.draw()
        path = self.graph.find_path(current_board)
        return path, len(path) - 1, self.cost

    def get_priority(self, board: Board):
        match self.algorithm:
            case "bfs":
                return -self.cost
            case "dfs":
                return 0
            case "astar":
                return -1 - self.heuristic(board)
