# Mystic Puzzle Problem

This repo contains a solution to solving the Mystic Puzzle problem of configurabile sizes by using path exploration algorithms.
The implemented algorithms are:

- Depth First Search
- Breadth First Search
- A\*

DFS and BFS can't handle random problems in reasonable time.

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

by configuring the constants `ITERATIONS` and `BOARD_SIZE` in the file.

The results can be visualized, with normalization, by running the same file again with option `-p` or `--plot`. If you don't want to run new tests, also add `-s` or `--skip`
By clicking somewhere on the plot you can see the precise values for each of the weights

On my machine, I run

- 20k 3x3 samples
- 1k 4x4 iterations
- TBD

After evaluating the outputs, which can be found in `plots/`, I chose the weights `(0.15, 0.75, 0.10)`

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

TBD
