from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


class PriorityQueue[T]:
    def __init__(self):
        self.queue = []
        self.size = 0

    def push(self, item: T, priority: int):
        self.queue.append((item, priority))
        self.size += 1
        self.queue.sort(key=lambda x: x[1])

    def pop(self):
        if self.size == 0:
            raise IndexError("pop from empty queue")
        self.size -= 1
        return self.queue.pop()[0]

    def __len__(self):
        return self.size

    def __contains__(self, item: T):
        return item in self.queue

    def __iter__(self):
        return iter(self.queue)

    def __str__(self):
        return str(self.queue)

    def __repr__(self):
        return str(self.queue)

    def __getitem__(self, index: int):
        return self.queue[index]

    def __setitem__(self, index: int, value: T):
        self.queue[index] = value

    def __delitem__(self, index: int):
        del self.queue[index]
        self.size -= 1
