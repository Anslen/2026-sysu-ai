"""Loss-curve plotting helpers (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script runs on headless setups.
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (must follow ``use`` call)

from .model import TrainingHistory


def _configure_cjk_font() -> None:
    """Pick the first available CJK-capable font for matplotlib.

    matplotlib does not auto-index ``.ttc`` font collections (e.g. the
    Noto CJK packaging used on Debian/Ubuntu), so we explicitly add
    candidate paths before resolving by name.
    """
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidate_paths:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception:
                # Font registration is best-effort; fall back to default.
                pass

    candidate_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidate_names if c in available), None)
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [
            chosen,
            *plt.rcParams["font.sans-serif"],
        ]
    # Keep the minus sign rendering correct under non-Latin fonts.
    plt.rcParams["axes.unicode_minus"] = False


_configure_cjk_font()


def plot_loss_curve(
    history: TrainingHistory,
    output_path: Path,
    title: str = "训练 / 验证集 Loss 曲线",
) -> None:
    """Render train/validation MSE curves to ``output_path`` as PNG."""
    epochs = list(range(1, len(history.train_loss) + 1))

    fig, (ax_loss, ax_rmse) = plt.subplots(
        nrows=1, ncols=2, figsize=(11, 4.2), constrained_layout=True
    )

    ax_loss.plot(epochs, history.train_loss, label="train MSE", color="#1f77b4")
    ax_loss.plot(epochs, history.val_loss, label="val MSE", color="#d62728")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE (log-price)")
    ax_loss.set_title("MSE vs. epoch")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_loss.legend()

    ax_rmse.plot(epochs, history.train_rmse, label="train RMSE", color="#1f77b4")
    ax_rmse.plot(epochs, history.val_rmse, label="val RMSE", color="#d62728")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.set_ylabel("RMSE (log-price)")
    ax_rmse.set_title("RMSE vs. epoch")
    ax_rmse.grid(True, linestyle="--", alpha=0.4)
    ax_rmse.legend()

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
