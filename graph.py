"""
I used the graph class to implement a custom draw function which I could not obtain with networkx.
It's drawn mostly as a tree, with the additional edges going back.
I also use the find_path method to get the path from the solution to the starting node.

The graph is represented with redundance in two structures:
- neighbors: {node: (level, [neighbours])}
- nodes: [[nodes_at_level0], [nodes_at_level1], ...]

The nodes needs to implement __hash__ and __eq__ methods to be used as keys in the neighbors dict.
"""

from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


class Graph[T]:
    def __init__(self):
        self.neighbors: dict[T, (int, list[T])] = {}  # {node: [neighbours]}
        self.nodes = []
        self.len = 0

    def add_node(self, item: T, parent: T = None):
        if item not in self.neighbors:
            self.neighbors[item] = (None, [])
            if parent:
                n = self.neighbors[parent][0] + 1
                if self.len <= n:
                    self.nodes.append([])
                    self.len += 1
                self.nodes[n].append(item)
            else:
                n = 0
                self.nodes.append(
                    [
                        item,
                    ]
                )
                self.len += 1
            self.neighbors[item] = (n, self.neighbors[item][1])

    def add_edge(self, node1: T, node2: T):
        self.neighbors[node1][1].append(node2)
        self.neighbors[node2][1].append(node1)

    def __contains__(self, item: T):
        print(self.neighbors)
        return item in self.neighbors

    def draw(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        coords = {}
        for i, nodes in enumerate(self.nodes):
            for j, node in enumerate(nodes):
                x = (j + 0.5) / len(nodes)
                y = 0.98 - i / self.len
                ax.text(
                    x,
                    y,
                    node,
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                )
                ax.plot(
                    x,
                    y,
                    "o",
                    color="black",
                    transform=ax.transAxes,
                )
                coords[node] = (x, y)

        for neighbor in self.neighbors:
            for n in self.neighbors[neighbor][1]:
                ax.plot(
                    [coords[neighbor][0], coords[n][0]],
                    [coords[neighbor][1], coords[n][1]],
                    "-",
                    color="black",
                    transform=ax.transAxes,
                )
        plt.show()

    def find_path(self, node: T):
        path = [node]
        while self.neighbors[node][0] != 0:
            node = next(
                filter(
                    lambda i: i[1][0] == self.neighbors[node][0] - 1
                    and node in i[1][1],
                    self.neighbors.items(),
                )
            )[0]
            path.append(node)
        return list(reversed(path))
