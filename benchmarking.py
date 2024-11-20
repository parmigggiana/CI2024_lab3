from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from custom_heuristics import improved_manhattan
from path_search import Solver
from puzzle import Board

SEED = 0


def run_benchmark(iters, board_size, seed=0):
    futures: list = []
    avg_quality = 0
    avg_cost = 0
    with tqdm.tqdm(
        desc=f"Solving {iters} random {board_size}x{board_size} problems", total=iters
    ) as pbar:
        with ThreadPoolExecutor() as executor:
            for _ in range(iters):
                random_board = Board(board_size, seed)

                t = executor.submit(
                    Solver(
                        starting_board=random_board,
                        algorithm="astar",
                        heuristic=improved_manhattan(board_size),
                    ).run
                )
                futures.append(t)

            for future in as_completed(fs=futures):
                _, quality, cost = future.result()
                avg_quality += quality / iters
                avg_cost += cost / iters
                pbar.update(1)

    return avg_quality, avg_cost


def main():
    quality_3, cost_3 = run_benchmark(1000, 3, SEED)
    print("3x3 Benchmark:")
    print(f"\tAvg Quality: {quality_3:.2f}\n\tAvg Cost: {cost_3:.2f}")
    quality_4, cost_4 = run_benchmark(100, 4, SEED)
    print("4x4 Benchmark:")
    print(f"\tAvg Quality: {quality_4:.2f}\n\tAvg Cost: {cost_4:.2f}")
    quality_5, cost_5 = run_benchmark(10, 5, SEED)
    print("5x5 Benchmark:")
    print(f"\tAvg Quality: {quality_5:.2f}\n\tAvg Cost: {cost_5:.2f}")


if __name__ == "__main__":
    main()
