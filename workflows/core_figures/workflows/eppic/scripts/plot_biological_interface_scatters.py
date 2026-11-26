#!/usr/bin/env python3
"""Create publication scatter comparisons for biological EPPIC interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#1F77B4"
RED = "#D62728"
MODEL_LABELS = {
    "esm3_1.4b_catjac": "ESM3-1.4B cat-Jacobian",
    "esm2_650m_catjac": "ESM2-650M cat-Jacobian",
    "esm2_650m_official": "ESM2-650M official head",
    "esm2_650m_refit": "ESM2-650M overlay refit",
    "esmc_300m_catjac": "ESM-C-300M cat-Jacobian",
    "esmc_300m_refit": "ESM-C-300M overlay refit",
    "esmc_600m_catjac": "ESM-C-600M cat-Jacobian",
    "esmc_600m_refit": "ESM-C-600M overlay refit",
    "esmc_6b_catjac": "ESM-C-6B cat-Jacobian",
    "esmc_6b_refit": "ESM-C-6B overlay refit",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    export_root = project_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "scatter_comparisons_per_pdb.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "eppic" / "scatters",
    )
    return parser.parse_args()


def aligned_values(
    frame: pd.DataFrame,
    x_model: str,
    y_model: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.pivot(index="pdb_id", columns="model", values=metric)
    pair = values[[x_model, y_model]].dropna()
    if len(pair) != 40:
        raise ValueError(
            f"{x_model} vs {y_model}, {metric}: expected 40 pairs, found {len(pair)}"
        )
    return pair[x_model].to_numpy(), pair[y_model].to_numpy()


def scatter_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_model: str,
    y_model: str,
    metric: str,
    color: str,
    title: str,
    panel_letter: str,
) -> None:
    x, y = aligned_values(frame, x_model, y_model, metric)

    ax.scatter(
        x,
        y,
        s=25,
        color=color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(MODEL_LABELS[x_model])
    ax.set_ylabel(MODEL_LABELS[y_model])
    ax.set_title(title, fontsize=10)
    ax.text(
        0.04,
        0.95,
        f"Mean x = {x.mean():.2f}\nMean y = {y.mean():.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.text(
        -0.14,
        1.06,
        panel_letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def make_two_panel(
    frame: pd.DataFrame,
    x_model: str,
    y_model: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.15))
    scatter_panel(
        axes[0],
        frame,
        x_model,
        y_model,
        "intra_recovery",
        BLUE,
        "Intrachain contacts",
        "a",
    )
    scatter_panel(
        axes[1],
        frame,
        x_model,
        y_model,
        "inter_recovery",
        RED,
        "Interface contacts",
        "b",
    )
    fig.tight_layout(w_pad=1.9)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_esmc_grid(frame: pd.DataFrame, output: Path) -> None:
    comparisons = (
        ("esmc_300m_catjac", "esmc_300m_refit"),
        ("esmc_600m_catjac", "esmc_600m_refit"),
        ("esmc_6b_catjac", "esmc_6b_refit"),
    )
    letters = iter("abcdef")
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.8))
    for row, (x_model, y_model) in enumerate(comparisons):
        scatter_panel(
            axes[row, 0],
            frame,
            x_model,
            y_model,
            "intra_recovery",
            BLUE,
            "Intrachain contacts",
            next(letters),
        )
        scatter_panel(
            axes[row, 1],
            frame,
            x_model,
            y_model,
            "inter_recovery",
            RED,
            "Interface contacts",
            next(letters),
        )
    fig.tight_layout(h_pad=2.0, w_pad=1.9)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    make_two_panel(
        frame,
        "esm2_650m_catjac",
        "esm3_1.4b_catjac",
        output_dir / "01_catjac_esm3_vs_esm2_650m.pdf",
    )
    make_two_panel(
        frame,
        "esm2_650m_official",
        "esm2_650m_catjac",
        output_dir / "02_esm2_650m_catjac_vs_official.pdf",
    )
    make_two_panel(
        frame,
        "esm2_650m_official",
        "esm2_650m_refit",
        output_dir / "03_esm2_650m_official_vs_refit.pdf",
    )
    make_esmc_grid(frame, output_dir / "04_esmc_catjac_vs_refit.pdf")
    print(f"Wrote four scatter PDFs to {output_dir}")


if __name__ == "__main__":
    main()
