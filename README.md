# Mystic Puzzle Problem
This repo contains a solution to solving the Mystic Puzzle problem of configurabile sizes by using path exploration algorithms.
The implemented algorithms are:
- Depth First Search
- Breadth First Search 
- A*

## Heuristics
For the A* algorithm (which pretty much always outperforms the others) there's a couple different heuristics implemented:
- Manhattan Distance
- Hemming Distance
- Weighted Manhattan + Linear Conflicts + Inversions

## Weights Optimization
Depending on the size of the board the 3 factors in the custom heuristic are weighed differently, with the exact weights chosen by running a large number of experiments with random weights on random boards, saved in the file `history.csv`. 
This can be done by running
```sh
python parameters_optimization.py
```
by configuring the constants `ITERATIONS` and `BOARD_SIZE` in the file.

The results can be visualized, with normalization, by running the same file again with option `-p` or `--plot`. If you don't want to run new tests, also add `-s` or `--skip`

## Examples
You can run some samples by calling
```sh
python main.py
```
which has some pre-configured instances.

