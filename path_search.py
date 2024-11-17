import numpy as np
from puzzle import Action, Board
from graph import Graph


class BFSSolver:
    def __init__(self, starting_board: Board):
        self.board = starting_board
        self.solution = np.arange(self.board.size**2)
        self.solution = np.roll(self.solution, -1).reshape(
            self.board.size, self.board.size
        )
        self.graph = Graph()

    def run(self):
        frontier = []
        explored_nodes = set()
        p = self.board
        frontier.append(p)
        self.graph.add_node(p)
        cost = 0
        while frontier:
            if np.array_equal(p.m, self.solution):
                break
            p = frontier.pop(0)
            explored_nodes.add(p)
            valid_actions = np.nonzero(p.valid_actions)[0]
            cost += 1
            for action in valid_actions:
                new: Board = p.act(Action(action))
                if new not in explored_nodes and new not in frontier:
                    frontier.append(new)
                    self.graph.add_node(new, p)
                self.graph.add_edge(p, new)
        else:
            raise ValueError("No solution found")

        # self.graph.draw()
        path = self.graph.find_path(p)
        return path, len(path) - 1, cost


class DFSSolver:
    def __init__(self, starting_board: Board):
        self.board = starting_board
        self.solution = np.arange(self.board.size**2)
        self.solution = np.roll(self.solution, -1).reshape(
            self.board.size, self.board.size
        )
        self.graph = Graph()

    def run(self):
        frontier = []
        explored_nodes = set()
        p = self.board
        frontier.append(p)
        self.graph.add_node(p)
        cost = 0
        while frontier:
            if np.array_equal(p.m, self.solution):
                break
            p = frontier.pop()
            explored_nodes.add(p)
            valid_actions = np.nonzero(p.valid_actions)[0]
            cost += 1
            for action in valid_actions:
                new: Board = p.act(Action(action))
                if new not in explored_nodes and new not in frontier:
                    frontier.append(new)
                    self.graph.add_node(new, p)
                self.graph.add_edge(p, new)
        else:
            raise ValueError("No solution found")
        # self.graph.draw()
        path = self.graph.find_path(p)
        return path, len(path) - 1, cost
