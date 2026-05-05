"""
Entry point for the TSP Genetic Algorithm solver.

Usage:
    python src/main.py [tsp_file]
    python src/main.py --plot <convergence.csv>

Defaults to data/qa194.tsp if no argument is provided.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ga
from ga import GeneticAlgorithm
from plotting import plot_convergence, plot_from_csv
from tee_logger import TeeLogger
from tsp_data import read_tsp


def _run_ga(filepath: Path) -> None:
    """Execute the GA pipeline: load data → run → plot → show."""
    logger: TeeLogger = TeeLogger()

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

    # Generate convergence plot from the CSV recorded during this run
    csv_path: Path = logger.log_dir / "convergence.csv"
    png_path: Path = logger.log_dir / "convergence.png"
    logger.log(f"\nGenerating convergence plot: {png_path}")
    plot_convergence(csv_path, png_path)
    logger.log("Plot saved.")
    logger.close()

    # Display the interactive plot window
    plot_from_csv(csv_path)


def _plot_from_records(csv_path: Path) -> None:
    """Load and display a previously saved convergence CSV without running GA."""
    resolved: Path = csv_path.resolve()
    if not resolved.exists():
        print(f"File not found: {resolved}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading convergence data from: {resolved}")
    plot_from_csv(resolved)


def main() -> None:
    # --plot mode: display saved convergence records
    if len(sys.argv) >= 3 and sys.argv[1] == "--plot":
        csv_path: Path = Path(sys.argv[2])
        _plot_from_records(csv_path)
        return

    filepath: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/qa194.tsp")
    _run_ga(filepath)


if __name__ == "__main__":
    main()
