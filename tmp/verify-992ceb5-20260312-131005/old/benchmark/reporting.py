from csv import DictWriter
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from matplotlib import use

use("Agg")
import matplotlib.pyplot as plt

from common.config import RECONSTRUCTION_NAME, RESULTS_DIR

SummaryRow: TypeAlias = dict[str, float | int]


def summarize_losses(
    scales: np.ndarray,
    losses_by_scale: list[list[torch.Tensor]],
) -> list[SummaryRow]:
    summary: list[SummaryRow] = []
    for scale, losses in zip(scales, losses_by_scale, strict=False):
        if not losses:
            msg = "没有可用于汇总的损失值。"
            raise ValueError(msg)
        loss_array = (
            torch.cat(losses)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        summary.append(
            {
                "scale": float(scale),
                "mean_loss": float(loss_array.mean()),
                "std_loss": float(loss_array.std(ddof=0)),
                "median_loss": float(np.median(loss_array)),
                "min_loss": float(loss_array.min()),
                "max_loss": float(loss_array.max()),
                "num_images": int(loss_array.size),
            },
        )
    return summary


def write_summary_csv(path: Path, summary: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(
            file,
            fieldnames=[
                "scale",
                "mean_loss",
                "std_loss",
                "median_loss",
                "min_loss",
                "max_loss",
                "num_images",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(
    path: Path,
    summary: list[SummaryRow],
    plot_xscale: str,
    dpi: int,
    title: str,
    curve_label: str,
) -> None:
    scales = np.asarray([row["scale"] for row in summary], dtype=np.float64)
    means = np.asarray([row["mean_loss"] for row in summary], dtype=np.float64)
    stds = np.asarray([row["std_loss"] for row in summary], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        scales,
        means,
        color="#0055cc",
        linewidth=2,
        label=curve_label,
    )
    axis.fill_between(
        scales,
        np.clip(means - stds, a_min=0.0, a_max=None),
        means + stds,
        color="#0055cc",
        alpha=0.18,
        label="±1 std",
    )
    axis.set_xlabel("Scale factor")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.set_xscale(plot_xscale)
    axis.grid(visible=True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return RESULTS_DIR / f"{RECONSTRUCTION_NAME}_milo"
