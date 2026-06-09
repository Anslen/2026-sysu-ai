"""End-to-end training pipeline.

Run with::

    uv run python -m house_prices

The script:

1. Ensures ``data/train.csv`` exists, extracting the Kaggle ZIP if not.
2. Loads and preprocesses the data.
3. Trains the hand-coded perceptron.
4. Reports per-epoch and final metrics.
5. Saves ``outputs/loss_curve.png`` and ``outputs/history.csv``.
6. Prints an overfitting analysis based on the loss curves.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

from .data import load_csv
from .metrics import mae, r2, rmse
from .model import Perceptron, TrainingHistory
from .plot import plot_loss_curve
from .preprocess import prepare_dataset


# Paths anchored to the project root (parent of the ``house_prices`` pkg).
_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PKG_DIR.parent
DATA_DIR = _PROJECT_DIR / "data"
OUTPUTS_DIR = _PROJECT_DIR / "outputs"

# Fallback location of the original Kaggle archive on the lab machine.
_FALLBACK_ZIP = Path(
    "/mnt/c/Users/Anslen/Downloads/house-prices-advanced-regression-techniques.zip"
)


# ----------------------------------------------------------------------
# Hyper-parameters
# ----------------------------------------------------------------------

LR: float = 0.05
EPOCHS: int = 500
BATCH_SIZE: int = 64
L2: float = 1e-3
LR_DECAY: float = 0.005
SEED: int = 42
VAL_RATIO: float = 0.2


def _ensure_data() -> Path:
    """Return path to ``train.csv``, extracting the ZIP if necessary."""
    train_path = DATA_DIR / "train.csv"
    if train_path.exists():
        return train_path
    if not _FALLBACK_ZIP.exists():
        raise FileNotFoundError(
            f"既找不到 {train_path}，也找不到 ZIP 数据 {_FALLBACK_ZIP}"
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_FALLBACK_ZIP) as z:
        z.extractall(DATA_DIR)
    return train_path


def _save_history_csv(history: TrainingHistory, path: Path) -> None:
    """Persist per-epoch metrics for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse", "train_rmse", "val_rmse"])
        for i, (tl, vl, tr, vr) in enumerate(
            zip(
                history.train_loss,
                history.val_loss,
                history.train_rmse,
                history.val_rmse,
            ),
            start=1,
        ):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}", f"{tr:.6f}", f"{vr:.6f}"])


def _analyse_overfitting(history: TrainingHistory) -> str:
    """Heuristic textual analysis of the loss curves."""
    train = np.array(history.train_loss)
    val = np.array(history.val_loss)

    # Generalisation gap on the final epoch.
    final_gap = float(val[-1] - train[-1])
    relative_gap = final_gap / max(train[-1], 1e-9)

    # Detect the "降-升" inflection: argmin somewhere before the end.
    best_epoch = int(np.argmin(val)) + 1
    last_epoch = len(val)
    val_min = float(val.min())
    val_end = float(val[-1])
    rebound = val_end - val_min
    rebound_ratio = rebound / max(val_min, 1e-9)

    lines: list[str] = []
    lines.append(
        f"  - 最终 train_mse = {train[-1]:.5f}, "
        f"val_mse = {val[-1]:.5f}, gap = {final_gap:+.5f} "
        f"(相对 {relative_gap * 100:+.1f}%)"
    )
    lines.append(
        f"  - 验证集 MSE 最低点出现在第 {best_epoch}/{last_epoch} 个 epoch "
        f"(val_mse_min={val_min:.5f})"
    )
    lines.append(
        f"  - 之后验证集 MSE 反弹幅度 = {rebound:.5f} "
        f"(相对 {rebound_ratio * 100:+.1f}%)"
    )

    if rebound_ratio > 0.05 and best_epoch < last_epoch * 0.9:
        verdict = (
            "结论：验证集 loss 在中途到达最低点后明显反弹，"
            "出现【明显过拟合】。建议减少 epoch、增大 L2 正则或加入早停。"
        )
    elif relative_gap > 0.5 or rebound_ratio > 0.02:
        verdict = (
            "结论：训练集 loss 持续下降而验证集 loss 已停滞或轻微反弹，"
            "出现【轻微过拟合】。当前 L2 正则与训练长度基本合适，"
            "可适度增大正则系数进一步提升泛化能力。"
        )
    else:
        verdict = (
            "结论：训练集与验证集 loss 同步下降并趋于平稳，"
            "差距较小，【未观察到明显过拟合】。"
        )
    lines.append(f"  - {verdict}")
    return "\n".join(lines)


def main() -> None:
    """Run the full training + evaluation pipeline."""
    print("=" * 60)
    print(" 实验任务二：感知机房价预测 (House Prices)")
    print("=" * 60)

    train_csv = _ensure_data()
    print(f"[1/5] 加载数据: {train_csv}")
    raw = load_csv(train_csv)
    print(f"      原始样本数: {raw.n_samples}, 列数: {len(raw.columns)}")

    print("[2/5] 数据预处理 (去噪 / 缺失值 / one-hot / 标准化)")
    prepared = prepare_dataset(raw, val_ratio=VAL_RATIO, seed=SEED)
    print(
        f"      训练集样本: {prepared.n_train} | "
        f"验证集样本: {prepared.n_val} | "
        f"特征维度: {prepared.n_features} | "
        f"过滤异常样本: {prepared.n_outliers_removed}"
    )

    print("[3/5] 训练感知机 (单层线性回归 + MSE + mini-batch SGD)")
    print(
        f"      超参数: lr={LR}, epochs={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, l2={L2}, lr_decay={LR_DECAY}, seed={SEED}"
    )
    model = Perceptron(
        n_features=prepared.n_features,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        l2=L2,
        lr_decay=LR_DECAY,
        seed=SEED,
    )
    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        prepared.X_val,
        prepared.y_val,
        verbose=True,
        log_every=20,
    )

    print("[4/5] 最终指标 (log-price 空间)")
    train_pred = model.predict(prepared.X_train)
    val_pred = model.predict(prepared.X_val)
    print(
        f"      train: RMSE={rmse(prepared.y_train, train_pred):.5f} "
        f"MAE={mae(prepared.y_train, train_pred):.5f} "
        f"R2={r2(prepared.y_train, train_pred):.4f}"
    )
    print(
        f"      val  : RMSE={rmse(prepared.y_val, val_pred):.5f} "
        f"MAE={mae(prepared.y_val, val_pred):.5f} "
        f"R2={r2(prepared.y_val, val_pred):.4f}"
    )

    print("[5/5] 保存产物")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    history_csv = OUTPUTS_DIR / "history.csv"
    loss_png = OUTPUTS_DIR / "loss_curve.png"
    _save_history_csv(history, history_csv)
    plot_loss_curve(history, loss_png)
    print(f"      历史记录: {history_csv}")
    print(f"      Loss 曲线: {loss_png}")

    print()
    print("过拟合分析:")
    print(_analyse_overfitting(history))


if __name__ == "__main__":
    main()
