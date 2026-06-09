"""Hand-coded perceptron (single-layer linear regressor) for the
House Prices task.

The model implements:

* forward pass        ``y_hat = X @ w + b``
* loss                ``L = 0.5/N * sum((y_hat - y) ** 2) + 0.5 * λ * ||w||^2``
* analytic gradients  ``dw = (1/N) * X.T @ (y_hat - y) + λ * w``
                      ``db = mean(y_hat - y)``
* mini-batch SGD optimiser with per-epoch shuffling

No external ML libraries are used; only ``numpy`` for vector/matrix math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingHistory:
    """Per-epoch training metrics.

    The lists below are guaranteed to all have length ``len(train_loss)``.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_rmse: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)


class Perceptron:
    """Single-layer perceptron used as a linear regressor.

    Parameters:
        n_features: Dimensionality of the input feature vector.
        lr: Learning rate for SGD.
        epochs: Number of full passes over the training set.
        batch_size: Mini-batch size; if larger than ``len(X)`` it
            becomes batch gradient descent.
        l2: L2 regularisation strength (``0.0`` disables it).
        seed: RNG seed used for weight initialisation and shuffling.
    """

    def __init__(
        self,
        n_features: int,
        lr: float = 0.01,
        epochs: int = 200,
        batch_size: int = 64,
        l2: float = 1e-4,
        lr_decay: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.n_features: int = n_features
        self.lr: float = lr
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.l2: float = l2
        # Multiplicative per-epoch decay: lr_t = lr / (1 + lr_decay * t).
        # ``0.0`` keeps the learning rate constant.
        self.lr_decay: float = lr_decay
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # He-style small init keeps initial logits close to 0 so the
        # first MSE measurement is dominated by the variance of y.
        self.w: np.ndarray = self._rng.normal(loc=0.0, scale=0.01, size=(n_features,))
        self.b: float = 0.0

    # ------------------------------------------------------------------
    # Forward / loss / gradients
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ``X @ w + b`` for a feature matrix ``X``."""
        return X @ self.w + self.b

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Plain mean squared error (without the L2 term).

        We report MSE on the log-price scale; downstream code converts
        it to RMSE for human-readable reporting.
        """
        diff = self.predict(X) - y
        return float(0.5 * np.mean(diff * diff))

    def _gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Analytic gradient of the regularised MSE w.r.t. (w, b)."""
        n = X.shape[0]
        diff = self.predict(X) - y  # shape: (n,)
        dw = (X.T @ diff) / n + self.l2 * self.w  # shape: (n_features,)
        db = float(diff.mean())
        return dw, db

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
        log_every: int = 10,
    ) -> TrainingHistory:
        """Train the model and return per-epoch metrics."""
        if X_train.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_train.shape[1]}"
            )

        history = TrainingHistory()
        n = X_train.shape[0]
        indices = np.arange(n)

        for epoch in range(1, self.epochs + 1):
            # Inverse-time decay keeps SGD steps stable late in training,
            # smoothing the loss curve so over-fitting trends are easier
            # to read off the plot.
            lr_t = self.lr / (1.0 + self.lr_decay * (epoch - 1))

            # Shuffle once per epoch so each mini-batch sees a fresh
            # ordering — important for SGD convergence.
            self._rng.shuffle(indices)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                dw, db = self._gradients(Xb, yb)
                self.w -= lr_t * dw
                self.b -= lr_t * db

            train_loss = self._mse(X_train, y_train)
            val_loss = self._mse(X_val, y_val)
            train_rmse = float(np.sqrt(2.0 * train_loss))
            val_rmse = float(np.sqrt(2.0 * val_loss))

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_rmse.append(train_rmse)
            history.val_rmse.append(val_rmse)

            if verbose and (
                epoch == 1 or epoch % log_every == 0 or epoch == self.epochs
            ):
                print(
                    f"  epoch {epoch:>4d}/{self.epochs} | "
                    f"train_mse={train_loss:.5f} val_mse={val_loss:.5f} | "
                    f"train_rmse={train_rmse:.5f} val_rmse={val_rmse:.5f}"
                )

        return history
