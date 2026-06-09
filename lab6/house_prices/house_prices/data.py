"""CSV loading for the Kaggle House Prices dataset.

The dataset is loaded as a column-oriented dictionary so that downstream
preprocessing can dispatch on column type (numeric vs. categorical)
without needing a heavy DataFrame dependency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# Sentinel string used by the Kaggle dataset for missing values.
NA_TOKEN = "NA"


@dataclass(frozen=True)
class RawDataset:
    """Column-oriented view of a CSV file.

    Attributes:
        columns: Ordered list of column names exactly as they appear in
            the CSV header.
        rows: ``rows[i]`` is the i-th column, as a list of raw strings
            (``"NA"`` is preserved verbatim and resolved later).
        n_samples: Number of data rows (excluding the header).
    """

    columns: list[str]
    rows: list[list[str]]
    n_samples: int


def load_csv(path: Path) -> RawDataset:
    """Load a CSV file into a :class:`RawDataset`.

    Parameters:
        path: Absolute path to the CSV file.

    Returns:
        A :class:`RawDataset` with one inner list per column.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the CSV is empty or rows have inconsistent
            arities.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc

        # Initialise one column per header entry.
        cols: list[list[str]] = [[] for _ in header]
        n_rows = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"Row {n_rows + 2} in {path} has {len(row)} columns, "
                    f"expected {len(header)}"
                )
            for i, value in enumerate(row):
                cols[i].append(value)
            n_rows += 1

    return RawDataset(columns=header, rows=cols, n_samples=n_rows)
