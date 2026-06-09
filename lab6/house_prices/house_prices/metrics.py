"""Hand-coded regression metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (1 - SS_res / SS_tot).

    Returns ``0.0`` when the target has zero variance — predicting the
    mean is then optimal and undefined R² is collapsed to 0.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean = float(y_true.mean())
    ss_tot = float(np.sum((y_true - mean) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot
