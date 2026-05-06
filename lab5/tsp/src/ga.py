"""
Genetic Algorithm for the Traveling Salesman Problem.

Uses tournament selection, order crossover, inversion mutation,
vectorized numpy evaluation, and elitism to evolve a population of
Chromosomes toward shorter tours.  Evolution is parallelised across
threads with per-thread random generators.

All tunable parameters are module-level constants for easy adjustment.
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from chromosome import (
    Chromosome,
    DistMatrix,
    _rng,
    crossover_registry,
    mutation_registry,
    seed_rng,
)

if TYPE_CHECKING:
    from tee_logger import TeeLogger

# ---------------------------------------------------------------------------
# Global GA constants — tweak these to tune the algorithm
# ---------------------------------------------------------------------------

POPULATION_SIZE: int = 200
"""Number of chromosomes in the population."""

MUTATION_METHODS: dict[str, float] = {
    "insertion": 0.15,
    "inversion": 0.10,
}
"""Mutation methods with per‑offspring independent application probabilities.
Each method is applied independently — an offspring may undergo multiple
or no mutations in a single generation."""

TOURNAMENT_SIZE: int = 3
"""Number of individuals competing in each tournament selection."""

ELITISM_COUNT: int = 2
"""Number of top chromosomes preserved unchanged each generation."""

MAX_GENERATIONS: int = 30000
"""Maximum number of generations to evolve."""

REPORT_INTERVAL: int = 500
"""Print a progress report every N generations."""

CROSSOVER_METHOD: str = "erx"
"""Key of the crossover operator in crossover_registry."""

SEED: int | None = None
"""Random seed for reproducibility.  Set to an int to fix, or None to skip."""

THREAD_WORKERS: int = os.cpu_count() or 4
"""Number of worker threads for parallel evolution.  Defaults to all CPU cores."""


# ---------------------------------------------------------------------------
# GeneticAlgorithm
# ---------------------------------------------------------------------------


class GeneticAlgorithm:
    """Orchestrates the GA lifecycle: init → evolve → report."""

    def __init__(self, dist_matrix: DistMatrix, dimension: int,
                 logger: TeeLogger | None = None) -> None:
        """
        Args:
            dist_matrix: n×n distance matrix shared by all chromosomes.
            dimension: Number of cities (n).
            logger: Optional TeeLogger for dual console+file output.
                    If None, output goes to stdout via print() only.
        """
        if SEED is not None:
            random.seed(SEED)
        seed_rng(SEED)

        self._logger: TeeLogger | None = logger
        self._dist_matrix: DistMatrix = dist_matrix
        self._dimension: int = dimension
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=THREAD_WORKERS
        )

        self._population: list[Chromosome] = []
        self._fitnesses: npt.NDArray[np.float64] = np.empty(0)
        self._best_chromosome: Chromosome | None = None
        self._best_distance: float = float("inf")
        self._best_generation: int = 0
        self._start_time: float = 0.0

        self._validate_operators()
        self._initialize_population()

    # -- validation --------------------------------------------------------

    def _validate_operators(self) -> None:
        """Ensure the configured crossover / mutation methods exist."""
        if CROSSOVER_METHOD not in crossover_registry:
            raise ValueError(
                f"Crossover method '{CROSSOVER_METHOD}' is not registered. "
                f"Available: {list(crossover_registry.keys())}"
            )
        for method in MUTATION_METHODS:
            if method not in mutation_registry:
                raise ValueError(
                    f"Mutation method '{method}' is not registered. "
                    f"Available: {list(mutation_registry.keys())}"
                )

    # -- population init ---------------------------------------------------

    def _initialize_population(self) -> None:
        """Create a random starting population and evaluate it."""
        self._population = [
            Chromosome.random(self._dimension)
            for _ in range(POPULATION_SIZE)
        ]
        self._evaluate()
        self._update_best(0)
        if self._logger is not None:
            self._logger.record_generation(0, self._best_distance)

    # -- evaluation --------------------------------------------------------

    def _compute_distances(self) -> npt.NDArray[np.float64]:
        """Vectorized: compute tour distance for all chromosomes at once."""
        genes_matrix: npt.NDArray[np.int64] = np.array(
            [c.genes for c in self._population], dtype=np.int64
        )
        rolled: npt.NDArray[np.int64] = np.roll(genes_matrix, -1, axis=1)
        return np.asarray(
            np.sum(self._dist_matrix[genes_matrix, rolled], axis=1),
            dtype=np.float64,
        )

    def _evaluate(self) -> None:
        """Compute fitness for all chromosomes via vectorized numpy."""
        distances: npt.NDArray[np.float64] = self._compute_distances()
        self._fitnesses = np.where(distances > 0, 1.0 / distances, np.inf)

    def _update_best(self, generation: int) -> None:
        """Check current population for a new best solution."""
        best_idx: int = int(np.argmax(self._fitnesses))
        best_dist: float = float(
            self._population[best_idx].compute_distance(self._dist_matrix)
        )
        if best_dist < self._best_distance:
            self._best_distance = best_dist
            self._best_chromosome = self._population[best_idx].__copy__()
            self._best_generation = generation

    # -- selection ---------------------------------------------------------

    def _tournament_select(self) -> Chromosome:
        """Tournament selection: pick TOURNAMENT_SIZE random individuals,
        return the one with highest fitness.  Uses thread-local RNG."""
        rng: random.Random = _rng()
        candidates: list[int] = rng.choices(
            range(POPULATION_SIZE), k=TOURNAMENT_SIZE
        )
        best_idx: int = max(candidates, key=lambda i: self._fitnesses[i])
        return self._population[best_idx]

    # -- evolution ---------------------------------------------------------

    def _evolve_generation(self) -> None:
        """Produce one generation in parallel: elitism + chunked crossover + mutation."""
        sorted_indices: npt.NDArray[np.int64] = np.argsort(-self._fitnesses)
        sorted_pop: list[Chromosome] = [self._population[i] for i in sorted_indices]

        new_pop: list[Chromosome] = []

        # Elitism — copy the best ELITISM_COUNT chromosomes unchanged
        for i in range(min(ELITISM_COUNT, POPULATION_SIZE)):
            new_pop.append(sorted_pop[i].__copy__())

        offspring_needed: int = POPULATION_SIZE - len(new_pop)
        pairs_needed: int = (offspring_needed + 1) // 2

        if pairs_needed == 0:
            self._population = new_pop
            return

        # Chunked parallel offspring production
        workers: int = min(THREAD_WORKERS, pairs_needed)
        chunk_size: int = math.ceil(pairs_needed / workers)

        def _produce_offspring(count: int) -> list[Chromosome]:
            """Worker: produce `count` offspring pairs in one chunk.
            Uses thread-local RNG shared across all pairs in this chunk."""
            rng: random.Random = _rng()
            result: list[Chromosome] = []
            for _ in range(count):
                p1: Chromosome = self._tournament_select()
                p2: Chromosome = self._tournament_select()
                c1, c2 = p1.crossover(p2, CROSSOVER_METHOD)
                for method, rate in MUTATION_METHODS.items():
                    if rng.random() < rate:
                        c1.mutate(method)
                    if rng.random() < rate:
                        c2.mutate(method)
                result.append(c1)
                result.append(c2)
            return result

        futures = [
            self._executor.submit(
                _produce_offspring,
                min(chunk_size, pairs_needed - i * chunk_size),
            )
            for i in range(workers)
        ]

        for future in futures:
            for child in future.result():
                if len(new_pop) < POPULATION_SIZE:
                    new_pop.append(child)

        self._population = new_pop

    # -- output ------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Write *message* via the attached logger (if any), else print to stdout."""
        if self._logger is not None:
            self._logger.log(message)
        else:
            print(message)

    # -- reporting ---------------------------------------------------------

    def _report(self, generation: int) -> None:
        """Output current progress, computing average distance vectorized."""
        avg_dist: float = float(np.mean(self._compute_distances()))
        elapsed: float = time.time() - self._start_time
        self._log(
            f"[Gen {generation:>5d}] "
            f"Best: {self._best_distance:>10.2f} "
            f"| Avg: {avg_dist:>10.2f} "
            f"| Time: {elapsed:.1f}s"
        )

    # -- main loop ---------------------------------------------------------

    def run(self) -> Chromosome:
        """
        Run the genetic algorithm.

        Returns the best Chromosome found.
        """
        self._start_time = time.time()

        self._log(
            f"GA start — Pop: {POPULATION_SIZE}  "
            f"MaxGen: {MAX_GENERATIONS}  "
            f"Mutations: {MUTATION_METHODS}  "
            f"Crossover: {CROSSOVER_METHOD}  "
            f"Thr: {THREAD_WORKERS}"
        )
        self._log(f"Initial best distance: {self._best_distance:.2f}")

        for gen in range(1, MAX_GENERATIONS + 1):
            self._evolve_generation()
            self._evaluate()
            self._update_best(gen)

            if self._logger is not None:
                self._logger.record_generation(gen, self._best_distance)

            if gen % REPORT_INTERVAL == 0:
                self._report(gen)

        # Final report
        elapsed: float = time.time() - self._start_time
        self._log(
            f"\n--- Finished after {MAX_GENERATIONS} generations "
            f"({elapsed:.1f}s) ---"
        )
        self._log(
            f"Best distance: {self._best_distance:.2f}  "
            f"(found at generation {self._best_generation})"
        )

        return self._best_chromosome  # type: ignore[return-value]
