"""
TSP data reading module.

Parses TSPLIB formatted files and builds the distance matrix
for use by the genetic algorithm.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt


CoordsArray = npt.NDArray[np.float64]
DistMatrix = npt.NDArray[np.float64]
ParsedData = Tuple[DistMatrix, int, CoordsArray]


def read_tsp(filepath: str | Path) -> ParsedData:
    """
    Parse a TSPLIB-format file and return a distance matrix.

    Args:
        filepath: Path to the .tsp file.

    Returns:
        A tuple of (distance_matrix, dimension, coordinates).
        - distance_matrix: n×n Euclidean distance matrix.
        - dimension: number of cities.
        - coordinates: n×2 array of (x, y) coordinates.
    """
    with open(filepath, "r") as f:
        lines: list[str] = f.readlines()

    dimension: int = 0
    edge_weight_type: str = ""
    coords: dict[int, tuple[float, float]] = {}
    in_section: bool = False

    for line in lines:
        stripped: str = line.strip()
        if not stripped or stripped == "EOF":
            continue

        if not in_section:
            if stripped.startswith("DIMENSION"):
                dimension = int(stripped.split(":")[-1].strip())
            elif stripped.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = stripped.split(":")[-1].strip()
            elif stripped == "NODE_COORD_SECTION":
                in_section = True
        else:
            parts: list[str] = stripped.split()
            node_id: int = int(parts[0])
            x: float = float(parts[1])
            y: float = float(parts[2])
            coords[node_id - 1] = (x, y)  # 0-index internally

    coords_array: CoordsArray = np.array(
        [coords[i] for i in range(dimension)], dtype=np.float64
    )

    distance_matrix: DistMatrix = build_distance_matrix(
        coords_array, edge_weight_type
    )

    return distance_matrix, dimension, coords_array


def build_distance_matrix(
    coords: CoordsArray,
    edge_weight_type: str = "EUC_2D",
) -> DistMatrix:
    """
    Build an n×n distance matrix from a set of coordinates.

    Args:
        coords: n×2 array of (x, y) coordinates.
        edge_weight_type: Distance metric type (currently only EUC_2D).

    Returns:
        n×n symmetric distance matrix.
    """
    n: int = len(coords)
    dist: DistMatrix = np.zeros((n, n), dtype=np.float64)

    if edge_weight_type == "EUC_2D":
        for i in range(n):
            diff: CoordsArray = coords[i + 1:] - coords[i]
            squared: np.ndarray = diff ** 2
            row_distances: np.ndarray = np.sqrt(
                squared[:, 0] + squared[:, 1]
            )
            dist[i, i + 1:] = row_distances
            dist[i + 1:, i] = row_distances
    else:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}"
        )

    return dist
