"""Data preprocessing for the House Prices regression task.

The pipeline performs, in order:

1. Type detection (numeric vs. categorical) using the training rows only.
2. Outlier filtering on the training set ("denoising").
3. Missing value imputation:
   - Categorical NAs are kept as the explicit category ``"None"`` because
     they carry semantic meaning (e.g. "no garage").
   - ``LotFrontage`` is filled per ``Neighborhood`` median; other numeric
     columns use the global median computed on the training split.
4. One-hot encoding of categorical features (training-only vocabulary).
5. Z-score standardisation of numeric features (training-only mean/std).
6. ``log1p`` transform on the target ``SalePrice`` for numerical
   stability (predictions can be inverted with ``expm1``).

All transformations are first ``fit`` on the training rows and then
``transform``-ed onto the validation rows so that no information leaks
across the split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import NA_TOKEN, RawDataset


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

# Columns excluded from feature matrix construction.
_EXCLUDED = {ID_COLUMN, TARGET_COLUMN}


@dataclass(frozen=True)
class PreparedDataset:
    """Numerical feature matrices ready for model training."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    # Statistics retained for diagnostic output.
    n_train: int
    n_val: int
    n_features: int
    n_outliers_removed: int


def _is_numeric_token(token: str) -> bool:
    """Return True when ``token`` parses as a finite float."""
    if token == NA_TOKEN or token == "":
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _detect_numeric_columns(raw: RawDataset) -> list[bool]:
    """Mark each column as numeric if any non-NA token parses as float.

    A column with at least one numeric token (and no non-numeric, non-NA
    token) is treated as numeric. Columns where every value is NA fall
    back to categorical (they will become a single one-hot column).
    """
    is_numeric: list[bool] = []
    for col_values in raw.rows:
        has_num = False
        has_cat = False
        for v in col_values:
            if v == NA_TOKEN or v == "":
                continue
            if _is_numeric_token(v):
                has_num = True
            else:
                has_cat = True
                break
        is_numeric.append(has_num and not has_cat)
    return is_numeric


def _column_to_float(col_values: list[str]) -> np.ndarray:
    """Convert a numeric column to ``float`` with NaN for NA tokens."""
    out = np.empty(len(col_values), dtype=np.float64)
    for i, v in enumerate(col_values):
        if v == NA_TOKEN or v == "":
            out[i] = np.nan
        else:
            out[i] = float(v)
    return out


def _split_indices(
    n: int, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) shuffled with a fixed RNG."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def _filter_outliers(
    numeric_arrays: dict[str, np.ndarray],
    target: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Drop noisy training samples.

    Two rules from the dataset author's recommendations and basic
    log-target z-score filtering are applied:

    * ``GrLivArea > 4000`` and ``SalePrice < 300000`` — known outliers.
    * Samples whose ``log1p(SalePrice)`` z-score exceeds 4 in absolute
      value, i.e. extreme prices that distort the OLS fit.
    """
    keep = np.ones(train_idx.size, dtype=bool)

    # Rule 1: anomalous large-but-cheap houses.
    gr_liv = numeric_arrays.get("GrLivArea")
    if gr_liv is not None:
        bad = (gr_liv[train_idx] > 4000) & (target[train_idx] < 300_000)
        keep &= ~bad

    # Rule 2: log-target z-score filter.
    log_target = np.log1p(target[train_idx])
    mean = float(log_target.mean())
    std = float(log_target.std())
    if std > 0.0:
        z = np.abs((log_target - mean) / std)
        keep &= z <= 4.0

    removed = int((~keep).sum())
    return train_idx[keep], removed


def _fit_numeric(
    train_values: np.ndarray,
) -> tuple[float, float, float]:
    """Compute (median, mean, std) on the non-NaN training values."""
    valid = train_values[~np.isnan(train_values)]
    median = float(np.median(valid)) if valid.size > 0 else 0.0
    # Impute then compute mean/std so that std reflects the imputed data.
    imputed = np.where(np.isnan(train_values), median, train_values)
    mean = float(imputed.mean())
    std = float(imputed.std())
    if std == 0.0:
        std = 1.0  # avoid division by zero for constant columns
    return median, mean, std


def _transform_numeric(
    values: np.ndarray, median: float, mean: float, std: float
) -> np.ndarray:
    """Impute missing values then apply z-score standardisation."""
    imputed = np.where(np.isnan(values), median, values)
    return (imputed - mean) / std


def _fit_lot_frontage_by_neighborhood(
    lot_frontage: np.ndarray,
    neighborhood: list[str],
    train_idx: np.ndarray,
) -> dict[str, float]:
    """Per-neighborhood median used to impute ``LotFrontage``."""
    medians: dict[str, float] = {}
    groups: dict[str, list[float]] = {}
    for i in train_idx:
        v = lot_frontage[i]
        if not np.isnan(v):
            groups.setdefault(neighborhood[i], []).append(float(v))
    for name, vs in groups.items():
        medians[name] = float(np.median(vs))
    return medians


def _categorical_levels(col_values: list[str], train_idx: np.ndarray) -> list[str]:
    """Sorted list of distinct training-set categories (NA -> "None")."""
    seen: set[str] = set()
    for i in train_idx:
        v = col_values[i]
        seen.add("None" if v == NA_TOKEN or v == "" else v)
    return sorted(seen)


def _one_hot_encode(
    col_values: list[str],
    indices: np.ndarray,
    levels: list[str],
) -> np.ndarray:
    """Encode the rows at ``indices`` against ``levels``.

    Unknown categories (present only in validation/test) collapse to all
    zeros, which is the conventional way to keep linear models robust to
    unseen categorical values.
    """
    level_to_idx = {lv: j for j, lv in enumerate(levels)}
    out = np.zeros((indices.size, len(levels)), dtype=np.float64)
    for row_i, src_i in enumerate(indices):
        v = col_values[src_i]
        token = "None" if v == NA_TOKEN or v == "" else v
        j = level_to_idx.get(token)
        if j is not None:
            out[row_i, j] = 1.0
    return out


def prepare_dataset(
    raw: RawDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> PreparedDataset:
    """Run the full preprocessing pipeline on ``raw``.

    Parameters:
        raw: Output of :func:`house_prices.data.load_csv` for the
            training CSV (must contain ``SalePrice``).
        val_ratio: Fraction of samples placed in the validation split.
        seed: RNG seed for the train/validation shuffle.
    """
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(f"Column {TARGET_COLUMN!r} missing from dataset")

    n = raw.n_samples
    train_idx, val_idx = _split_indices(n, val_ratio, seed)

    is_numeric = _detect_numeric_columns(raw)
    col_index = {name: i for i, name in enumerate(raw.columns)}

    # Materialise numeric columns once (reused for outlier filtering).
    numeric_arrays: dict[str, np.ndarray] = {}
    for name, idx in col_index.items():
        if is_numeric[idx]:
            numeric_arrays[name] = _column_to_float(raw.rows[idx])

    target = numeric_arrays[TARGET_COLUMN]
    train_idx, n_outliers_removed = _filter_outliers(numeric_arrays, target, train_idx)

    # Build feature matrix column-by-column.
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    neighborhood = raw.rows[col_index["Neighborhood"]]
    lot_frontage_medians = _fit_lot_frontage_by_neighborhood(
        numeric_arrays["LotFrontage"], neighborhood, train_idx
    )
    overall_lot_frontage_median = float(
        np.median(
            numeric_arrays["LotFrontage"][train_idx][
                ~np.isnan(numeric_arrays["LotFrontage"][train_idx])
            ]
        )
    )

    for idx, name in enumerate(raw.columns):
        if name in _EXCLUDED:
            continue
        if is_numeric[idx]:
            values = numeric_arrays[name].copy()
            if name == "LotFrontage":
                # Group-wise imputation overrides the global median for
                # this column; remaining NaNs (unseen neighborhoods)
                # fall back to the global median below.
                for i in range(values.size):
                    if np.isnan(values[i]):
                        med = lot_frontage_medians.get(
                            neighborhood[i], overall_lot_frontage_median
                        )
                        values[i] = med
            median, mean, std = _fit_numeric(values[train_idx])
            train_blocks.append(
                _transform_numeric(values[train_idx], median, mean, std).reshape(-1, 1)
            )
            val_blocks.append(
                _transform_numeric(values[val_idx], median, mean, std).reshape(-1, 1)
            )
            feature_names.append(name)
        else:
            levels = _categorical_levels(raw.rows[idx], train_idx)
            train_blocks.append(_one_hot_encode(raw.rows[idx], train_idx, levels))
            val_blocks.append(_one_hot_encode(raw.rows[idx], val_idx, levels))
            feature_names.extend(f"{name}={lv}" for lv in levels)

    X_train = np.concatenate(train_blocks, axis=1)
    X_val = np.concatenate(val_blocks, axis=1)

    # Log-transform the target. We store the log-price as the regression
    # target so MSE on the log scale is a stable training objective.
    y_train = np.log1p(target[train_idx])
    y_val = np.log1p(target[val_idx])

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        n_train=int(train_idx.size),
        n_val=int(val_idx.size),
        n_features=X_train.shape[1],
        n_outliers_removed=n_outliers_removed,
    )
