# Mystic Puzzle Problem

This repo contains a solution to solving the Mystic Puzzle problem of configurabile sizes by using path exploration algorithms.
The implemented algorithms are:

- Depth First Search
- Breadth First Search
- A\*

DFS and BFS can't handle random problems in reasonable time.

> [!NOTE] 
> Through the whole repository I use the terms quality for the number of steps in the final solution and cost for the total number of states evaluated. 
> This means in all cases lower is better for both.

## Heuristics

For the A\* algorithm (which pretty much always outperforms the others) there's a couple different heuristics implemented:

- Manhattan Distance
- Hemming Distance (~10x slower than manhattan and results are ~6x worse, but it can solve random 4x4 boards)
- Dijkstra (Works very slowly for random 3x3 boards, too long for bigger ones)
- Weighted Manhattan + Linear Conflicts + Inversions

## Weights Optimization

Depending on the size of the board the 3 factors in the custom heuristic are weighed differently, with the exact weights chosen by running a large number of experiments with random normalized weights on random boards, saved in the file `history.csv`.
This can be done by running

```sh
python parameters_optimization.py
```
after configuring the constants `ITERATIONS` and `BOARD_SIZE` in the file.

The results can be visualized, with normalization, by running the same file again with option `-p` or `--plot`. If you don't want to run new tests, also add `-s` or `--skip`
By clicking somewhere on the plot you can see the precise values for each of the weights

After evaluating the outputs, which can be found in `plots/`, I chose the weights `(0.15, 0.75, 0.10)` for board size 3 and `(0.30, 0.55, 0.15)` for bigger boards

## Examples

You can run some samples by calling

```sh
python main.py
```
which has some pre-configured instances.

## Benchmarking

To get an estimation of the overall performance of the heuristic with the chosen weights, you can run

```sh
python benchmarking.py
```

These are the results from running on my machine:

```
Solving 5000 random 3x3 problems: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:13<00:00, 373.14it/s]
3x3 Benchmark:
        Avg Quality: 43.46
        Avg Cost: 77.57
Solving 1000 random 4x4 problems: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:06<00:00, 15.05it/s]
4x4 Benchmark:
        Avg Quality: 157.84
        Avg Cost: 589.20
Solving 50 random 5x5 problems: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:47<00:00,  3.36s/it]
5x5 Benchmark:
        Avg Quality: 406.42
        Avg Cost: 4464.56
```

## Implementation details

- `puzzle.Board` - Contains all the logic for a generic n-puzzle state, including the generation of a random problem (by default using a permutation and checking for solvability, optionally with a number of random steps from the solution)
- `path_search.Solver` - Contains all the logic for the path optimization, with different algorithms and heuristics in the case of A*
- `priorityqueue.PriorityQueue` - Implements a simple priority queue for use in the Solver
- `graph.Graph` - Implements a graph data structure. It's NOT memory efficient, but it holds the data in such a way to resemble a tree-structure, which can be seen by calling the `draw` method. It's useless for big graphs, it was intended to be used for debugging the path search algorithms in the early stage.
- `custom_heuristics.py` - Defines two more heuristics, specific to the n-puzzle problem. The third function, `improved_manhattan`, combines these two with the manhattan distance. It contains default weights which can be overwritten.
- `parameters_optimization.py` - Can be ran to randomly generate weights for `improved_manhattan`, test them against random boards and save the result in the file `history.csv`. By adding the `-p` parameter it plots the results (both on-screen and in a file in `plots/`)
- `benchmarking.py` - Can be ran to test the default weights against many random problems and prints average quality and cost for each tested board size.
- `main.py` - Contains a couple of samples to compare the proposed heuristic with the default weights against simple manhattan distance, complete with pretty-printing of the found paths (long paths are truncated in the middle)