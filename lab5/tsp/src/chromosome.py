"""
Chromosome class for TSP genetic algorithm.

Encoding: permutation of city indices representing the travel order.
Fitness: inverse of total tour distance (shorter tour → higher fitness).
Crossover: Order Crossover (OX).
Mutation: Inversion Mutation.

Extensibility: use register_crossover / register_mutation decorators
to add new operators without modifying the Chromosome class.
"""

from __future__ import annotations

import random
import threading as _t
from typing import Callable

import numpy as np
import numpy.typing as npt

# Type aliases (those referencing Chromosome are defined after the class)
Genes = tuple[int, ...]
DistMatrix = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Thread-local RNG — enables thread-safe parallel random operations
# ---------------------------------------------------------------------------

_rng_local = _t.local()
_rng_seed_source: random.Random = random.Random()


def _rng() -> random.Random:
    """Return a thread-local Random instance for thread-safe parallel use."""
    rng: random.Random | None = getattr(_rng_local, "instance", None)
    if rng is None:
        seed: int = _rng_seed_source.randrange(0, 2**31)
        _rng_local.instance = random.Random(seed)
        rng = _rng_local.instance
    return rng


def seed_rng(seed: int | None) -> None:
    """
    Re-seed the RNG source for reproducibility.

    Call after setting the global random seed.  Each thread's local RNG
    will derive a deterministic sequence from this source.
    """
    global _rng_seed_source
    _rng_seed_source = random.Random(seed)


# ---------------------------------------------------------------------------
# Registry dictionaries — populated via decorators at module level
# ---------------------------------------------------------------------------

crossover_registry: dict[str, CrossoverFn] = {}
mutation_registry: dict[str, MutationFn] = {}


def register_crossover(name: str) -> Callable[[CrossoverFn], CrossoverFn]:
    """
    Decorator that registers a crossover function under the given name.

    Usage:
        @register_crossover("ox")
        def ox_crossover(parent1, parent2): ...
    """

    def decorator(func: CrossoverFn) -> CrossoverFn:
        crossover_registry[name] = func
        return func

    return decorator


def register_mutation(name: str) -> Callable[[MutationFn], MutationFn]:
    """
    Decorator that registers a mutation function under the given name.

    Usage:
        @register_mutation("inversion")
        def inversion_mutation(chromosome): ...
    """

    def decorator(func: MutationFn) -> MutationFn:
        mutation_registry[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Chromosome class
# ---------------------------------------------------------------------------


class Chromosome:
    """
    Represents a candidate TSP solution as a permutation of city indices.

    Attributes:
        genes: The permutation (list of city indices).
    """

    __slots__ = ("genes",)

    genes: Genes

    def __init__(self, genes: Genes) -> None:
        """
        Initialize a Chromosome.

        Args:
            genes: A permutation of city indices, e.g. (0, 3, 1, 2).
        """
        self.genes: Genes = tuple(genes)

    @classmethod
    def random(cls, n: int) -> Chromosome:
        """
        Factory method: create a chromosome with a random permutation.

        Args:
            n: Number of cities.

        Returns:
            A new Chromosome with a random city order.
        """
        lst: list[int] = list(range(n))
        _rng().shuffle(lst)
        return cls(tuple(lst))

    # -- tour evaluation ---------------------------------------------------

    def compute_distance(self, dist_matrix: DistMatrix) -> float:
        """
        Total tour distance (closed loop, returns to start city).

        Args:
            dist_matrix: n×n distance matrix to compute distance against.
        """
        total: float = 0.0
        genes: Genes = self.genes
        n: int = len(genes)
        for i in range(n - 1):
            total += float(dist_matrix[genes[i], genes[i + 1]])
        total += float(dist_matrix[genes[-1], genes[0]])
        return total

    def compute_fitness(self, dist_matrix: DistMatrix) -> float:
        """
        Fitness value: inverse of total tour distance.

        Args:
            dist_matrix: n×n distance matrix to compute fitness against.
        """
        d: float = self.compute_distance(dist_matrix)
        return 1.0 / d if d > 0 else float("inf")

    # -- crossover --------------------------------------------------------

    def crossover(
        self, other: Chromosome, method: str = "ox"
    ) -> tuple[Chromosome, Chromosome]:
        """
        Perform crossover with another chromosome.

        Args:
            other: The second parent chromosome.
            method: Name of the crossover operator (registered key).

        Returns:
            A tuple of two offspring Chromosomes.
        """
        op: CrossoverFn | None = crossover_registry.get(method)
        if op is None:
            raise ValueError(
                f"Unknown crossover method '{method}'. "
                f"Available: {list(crossover_registry.keys())}"
            )
        return op(self, other)

    # -- mutation ---------------------------------------------------------

    def mutate(self, method: str = "inversion") -> None:
        """
        Apply a mutation operator to this chromosome (in-place).

        Args:
            method: Name of the mutation operator (registered key).
        """
        op: MutationFn | None = mutation_registry.get(method)
        if op is None:
            raise ValueError(
                f"Unknown mutation method '{method}'. "
                f"Available: {list(mutation_registry.keys())}"
            )
        op(self)

    # -- dunder methods ---------------------------------------------------

    def __repr__(self) -> str:
        genes: Genes = self.genes
        if len(genes) <= 10:
            return f"Chromosome({genes})"
        return f"Chromosome({genes[:5]}...{genes[-5:]}, n={len(genes)})"

    def __copy__(self) -> Chromosome:
        """Shallow copy — tuple is immutable so no copy is needed."""
        return Chromosome(self.genes)


# Type aliases referencing Chromosome (must come after class definition)
CrossoverFn = Callable[[Chromosome, Chromosome], tuple[Chromosome, Chromosome]]
MutationFn = Callable[[Chromosome], None]


# ---------------------------------------------------------------------------
# Built-in crossover operator: Order Crossover (OX)
# ---------------------------------------------------------------------------


@register_crossover("ox")
def ox_crossover(parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Order Crossover (OX).

    Procedure:
      1. Pick two random cut points.
      2. Child inherits the middle segment from parent1.
      3. Remaining positions are filled with cities from parent2
         in the order they appear (skipping those already present),
         starting from after the second cut point (wrapping around).

    Returns two offspring (parent1⊕parent2 and parent2⊕parent1).
    """
    genes1: Genes = parent1.genes
    genes2: Genes = parent2.genes
    n: int = len(genes1)
    rng: random.Random = _rng()

    def _ox_worker(p: Genes, q: Genes) -> Genes:
        a: int = rng.randrange(0, n)
        b: int = rng.randrange(0, n)
        if a > b:
            a, b = b, a
        b += 1  # now segment is [a, b)

        child: list[int | None] = [None] * n
        child[a:b] = p[a:b]  # inherit middle segment from p

        # Build the filling order from q
        fill: list[int] = [city for city in q[b:] + q[:b] if city not in child]
        idx: int = 0
        for i in range(b, n + a):
            wrapped: int = i % n
            if child[wrapped] is None:
                child[wrapped] = fill[idx]
                idx += 1

        return tuple(child)  # type: ignore[return-value]

    return (
        Chromosome(_ox_worker(genes1, genes2)),
        Chromosome(_ox_worker(genes2, genes1)),
    )


# ---------------------------------------------------------------------------
# Built-in mutation operator: Inversion Mutation
# ---------------------------------------------------------------------------


@register_mutation("inversion")
def inversion_mutation(chromosome: Chromosome) -> None:
    """
    Inversion Mutation.

    Select two random positions and reverse the subsequence between them
    (inclusive of both endpoints).  Replaces the genes tuple with a new one.
    """
    genes: Genes = chromosome.genes
    n: int = len(genes)
    rng: random.Random = _rng()
    a: int = rng.randrange(0, n)
    b: int = rng.randrange(0, n)
    if a > b:
        a, b = b, a
    rev: Genes = genes[a:b + 1][::-1]
    chromosome.genes = genes[:a] + rev + genes[b + 1:]
