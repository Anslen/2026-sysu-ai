"""
Entry point for the TSP Genetic Algorithm solver.

Usage:
    python src/main.py [tsp_file]

Defaults to data/qa194.tsp if no argument is provided.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ga
from ga import GeneticAlgorithm
from tee_logger import TeeLogger
from tsp_data import read_tsp


def main() -> None:
    filepath: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/qa194.tsp")

    logger = TeeLogger()

    logger.log(f"Loading TSP data from {filepath} ...")
    dist_matrix, dimension, _ = read_tsp(str(filepath))
    logger.log(f"Cities: {dimension}  |  Dist matrix: {dist_matrix.shape}")

    logger.log_header(
        command=" ".join(sys.argv),
        input_file=str(filepath),
        cities=dimension,
        pop_size=ga.POPULATION_SIZE,
        max_gen=ga.MAX_GENERATIONS,
        mutation_rate=ga.MUTATION_RATE,
        crossover_method=ga.CROSSOVER_METHOD,
        mutation_method=ga.MUTATION_METHOD,
        tournament_size=ga.TOURNAMENT_SIZE,
        elitism_count=ga.ELITISM_COUNT,
        report_interval=ga.REPORT_INTERVAL,
        seed=ga.SEED,
        thread_workers=ga.THREAD_WORKERS,
    )

    ga_instance: GeneticAlgorithm = GeneticAlgorithm(
        dist_matrix, dimension, logger=logger
    )
    best = ga_instance.run()

    logger.log(f"\nBest tour ({len(best.genes)} cities):")
    logger.log(str(best.genes))

    logger.log_footer(
        best_tour=str(best.genes),
        best_distance=best.compute_distance(dist_matrix),
    )
    logger.close()


if __name__ == "__main__":
    main()
