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
# Built-in crossover operator: Edge Recombination Crossover (ERX)
# ---------------------------------------------------------------------------


@register_crossover("erx")
def erx_crossover(parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Edge Recombination Crossover (ERX).

    Builds an edge map from both parents — for each city, the set of
    cities adjacent to it in either parent tour.  Offspring are
    constructed by greedily selecting the neighbor with the fewest
    remaining edges (min‑degree heuristic), which maximises the
    preservation of parental edges.

    Returns two offspring generated from two independent random walks
    through the same edge map.
    """
    genes1: Genes = parent1.genes
    genes2: Genes = parent2.genes
    n: int = len(genes1)
    rng: random.Random = _rng()

    # Build the undirected edge map from both parent tours
    edge_map: dict[int, set[int]] = {i: set() for i in range(n)}

    def _collect_edges(tour: Genes) -> None:
        """Record undirected edges from *tour* into *edge_map*."""
        for i in range(n):
            u: int = tour[i]
            v: int = tour[(i + 1) % n]
            edge_map[u].add(v)
            edge_map[v].add(u)

    _collect_edges(genes1)
    _collect_edges(genes2)

    def _remove_city(city: int, em: dict[int, set[int]]) -> None:
        """Remove *city* from every neighbour set in *em*."""
        for neighbors in em.values():
            neighbors.discard(city)

    def _erx_build_child(
        em_base: dict[int, set[int]],
    ) -> Genes:
        """
        Build one offspring via the ERX algorithm.

        At each step the current city's neighbours are examined;
        the one with the fewest remaining edges is chosen (ties broken
        randomly).  If no neighbour remains, a random unvisited city
        is picked.
        """
        em: dict[int, set[int]] = {c: set(adj) for c, adj in em_base.items()}
        visited: set[int] = set()

        current: int = rng.choice(list(em.keys()))
        tour: list[int] = [current]
        visited.add(current)
        _remove_city(current, em)

        for _ in range(n - 1):
            candidates: set[int] = em.get(current, set())
            next_city: int

            if candidates:
                # Min‑degree heuristic: pick candidate with fewest remaining edges
                min_deg: int = min(len(em.get(c, set())) for c in candidates)
                best: list[int] = [
                    c for c in candidates if len(em.get(c, set())) == min_deg
                ]
                next_city = rng.choice(best)
            else:
                # Dead end — pick any unvisited city at random
                remaining: list[int] = [
                    c for c in em if c not in visited
                ]
                if not remaining:
                    break
                next_city = rng.choice(remaining)

            tour.append(next_city)
            visited.add(next_city)
            _remove_city(next_city, em)
            current = next_city

        return tuple(tour)  # type: ignore[return-value]

    return (
        Chromosome(_erx_build_child(edge_map)),
        Chromosome(_erx_build_child(edge_map)),
    )


# ---------------------------------------------------------------------------
# Built-in crossover operator: Partially Mapped Crossover (PMX)
# ---------------------------------------------------------------------------


@register_crossover("pmx")
def pmx_crossover(parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Partially Mapped Crossover (PMX).

    Selects two random cut points and copies the segment between them
    from the first parent to the child.  The remaining positions are
    filled from the second parent; conflicts are resolved via the
    mapping induced by the swapped segment.

    Returns two offspring (parent1⊕parent2 and parent2⊕parent1).
    """
    genes1: Genes = parent1.genes
    genes2: Genes = parent2.genes
    n: int = len(genes1)
    rng: random.Random = _rng()

    a: int = rng.randrange(0, n)
    b: int = rng.randrange(0, n)
    if a > b:
        a, b = b, a
    b += 1  # segment is [a, b)

    def _pmx_child(p: Genes, q: Genes) -> Genes:
        """Build one child using p for the segment and q for the rest.

        Conflicts (values already placed in the segment) are resolved by
        following the position‑based mapping chain: if q[i] is already in
        child, find its position j in the segment of p and replace it with
        q[j]; repeat until an unused value is found.  A visited set
        prevents infinite loops on closed permutation cycles."""
        child: list[int] = [-1] * n
        used: set[int] = set()

        # Position lookup for p within the segment
        pos_in_p: dict[int, int] = {}
        for j in range(a, b):
            child[j] = p[j]
            used.add(p[j])
            pos_in_p[p[j]] = j

        for i in range(n):
            if a <= i < b:
                continue
            val: int = q[i]
            visited: set[int] = {val}
            while val in used:
                j: int | None = pos_in_p.get(val)
                if j is None:
                    break
                val = q[j]
                if val in visited:
                    # Cycle detected — pick any unused city
                    for c in range(n):
                        if c not in used:
                            val = c
                            break
                    break
                visited.add(val)
            child[i] = val
            used.add(val)

        return tuple(child)  # type: ignore[return-value]

    return (
        Chromosome(_pmx_child(genes1, genes2)),
        Chromosome(_pmx_child(genes2, genes1)),
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


# ---------------------------------------------------------------------------
# Built-in mutation operator: Insertion Mutation
# ---------------------------------------------------------------------------


@register_mutation("insertion")
def insertion_mutation(chromosome: Chromosome) -> None:
    """
    Insertion Mutation.

    Randomly selects a city and re‑inserts it at a different position,
    shifting the intermediate cities along the tour.  This is a
    local‑move operator that preserves most of the tour structure while
    exploring alternative orderings of individual cities.
    """
    genes: Genes = chromosome.genes
    n: int = len(genes)
    rng: random.Random = _rng()
    src: int = rng.randrange(0, n)
    dst: int = rng.randrange(0, n - 1)
    if dst >= src:
        dst += 1

    city: int = genes[src]
    if src < dst:
        chromosome.genes = (
            genes[:src] + genes[src + 1:dst + 1] + (city,) + genes[dst + 1:]
        )
    else:
        chromosome.genes = (
            genes[:dst] + (city,) + genes[dst:src] + genes[src + 1:]
        )
