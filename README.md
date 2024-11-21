# Mystic Puzzle Problem
This repo contains a solution to solving the Mystic Puzzle problem of configurabile sizes by using path exploration algorithms.
The implemented algorithms are:
- Depth First Search
- Breadth First Search 
- A*

DFS and BFS can't handle random problems in reasonable time. 

## Heuristics
For the A* algorithm (which pretty much always outperforms the others) there's a couple different heuristics implemented:
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
by configuring the constants `ITERATIONS` and `BOARD_SIZE` in the file.

The results can be visualized, with normalization, by running the same file again with option `-p` or `--plot`. If you don't want to run new tests, also add `-s` or `--skip`
By clicking somewhere on the plot you can see the precise values for each of the weights

On my machine, I run 
- 20k 3x3 samples in about 3:25 minutes and chose the weights `(0.15, 0.78, 0.08)`
- 1k 4x4 iterations
- TBD

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
TBD 
```

## Implementation details
TBD