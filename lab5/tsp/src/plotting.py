"""
Plotting utilities for TSP GA convergence data.

Reads convergence.csv files and generates publication‑quality
convergence curves.  Supports both head‑less PNG export and
interactive display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt


def _read_csv(csv_path: Path) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Parse a convergence CSV file.

    Returns:
        (generations, best_distances) as numpy arrays.
    """
    with open(str(csv_path), "r", encoding="utf-8") as fh:
        fh.readline()  # skip header line
        lines: list[str] = fh.readlines()

    generations: list[int] = []
    distances: list[float] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts: list[str] = line.split(",")
        if len(parts) < 2:
            continue
        generations.append(int(parts[0]))
        distances.append(float(parts[1]))

    return (
        np.array(generations, dtype=np.int64),
        np.array(distances, dtype=np.float64),
    )


def plot_convergence(
    csv_path: Path,
    output_path: Path,
    *,
    title: str = "TSP Genetic Algorithm — Convergence Curve",
) -> None:
    """
    Read *csv_path*, generate a convergence plot, and save it as a PNG.

    Uses the 'Agg' backend so no display is required — safe for
    head‑less servers.

    Args:
        csv_path: Path to the convergence.csv file.
        output_path: Where to save the PNG image.
        title: Optional plot title.
    """
    original_backend: str = matplotlib.get_backend()
    matplotlib.use("Agg")

    try:
        gens, dists = _read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gens, dists, linewidth=1.0, color="#1f77b4")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Distance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
    finally:
        matplotlib.use(original_backend)


def plot_from_csv(csv_path: Path) -> None:
    """
    Load convergence data from *csv_path* and display an interactive plot.

    Suitable for re‑viewing saved run results.

    Args:
        csv_path: Path to the convergence.csv file.
    """
    gens, dists = _read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(gens, dists, linewidth=1.0, color="#1f77b4")
    plt.xlabel("Generation")
    plt.ylabel("Best Distance")
    plt.title("TSP Genetic Algorithm — Convergence Curve")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    plt.show()
