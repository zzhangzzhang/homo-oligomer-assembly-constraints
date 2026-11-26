#!/usr/bin/env python3
"""Plot the two-panel EPPIC scaling figure from publication CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext


COLORS = {
    "bio_inter": "#D62728",
    "bio_intra": "#1F77B4",
    "xtal_inter": "#2CA02C",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    export_root = project_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=export_root
        / "outputs"
        / "reproduced"
        / "eppic"
        / "eppic_scaling_qid35.pdf",
    )
    parser.add_argument(
        "--max-pairformer-qid",
        type=int,
        default=35,
        help="Highest MSA Pairformer QID cutoff shown.",
    )
    return parser.parse_args()


def metric_series(frame: pd.DataFrame) -> tuple[tuple[str, str, pd.DataFrame], ...]:
    return (
        (
            "Biological interface",
            COLORS["bio_inter"],
            frame[frame["set"] == "bio"].assign(
                value=lambda x: x["mean_inter_recovery"]
            ),
        ),
        (
            "Biological intrachain",
            COLORS["bio_intra"],
            frame[frame["set"] == "bio"].assign(
                value=lambda x: x["mean_intra_recovery"]
            ),
        ),
        (
            "Crystal interface",
            COLORS["xtal_inter"],
            frame[frame["set"] == "xtal"].assign(
                value=lambda x: x["mean_inter_recovery"]
            ),
        ),
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(direction="out", length=3)
    ax.set_ylim(0, 0.82)


def plot_scaling(ax: plt.Axes, summary: pd.DataFrame) -> None:
    family_styles = {
        "ESM2": {"linestyle": "-", "marker": "o", "fillstyle": "full"},
        "ESM-C refit": {
            "linestyle": "--",
            "marker": "s",
            "fillstyle": "none",
        },
    }
    for _, color, metric_frame in metric_series(summary):
        for family, style in family_styles.items():
            values = (
                metric_frame[metric_frame["family"] == family]
                .sort_values("size_m")
                .copy()
            )
            ax.plot(
                values["size_m"],
                values["value"],
                color=color,
                linewidth=1.5,
                markersize=4.5,
                markeredgewidth=1.1,
                **style,
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Mean contact recovery at top-X")
    ax.set_title("Protein language models", fontsize=10)
    style_axes(ax)


def plot_pairformer(
    ax: plt.Axes,
    summary: pd.DataFrame,
    max_qid: int,
) -> None:
    plotted = summary.loc[summary["qid"].le(max_qid)].copy()
    if plotted.empty:
        raise ValueError(f"No Pairformer values at or below QID {max_qid}")
    series = (
        (
            "bio",
            "mean_inter_recovery",
            COLORS["bio_inter"],
        ),
        (
            "bio",
            "mean_intra_recovery",
            COLORS["bio_intra"],
        ),
        (
            "xtal",
            "mean_inter_recovery",
            COLORS["xtal_inter"],
        ),
    )
    for set_name, column, color in series:
        values = plotted[plotted["set"] == set_name].sort_values("qid")
        ax.plot(
            values["qid"],
            values[column],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4.5,
        )

    ax.set_xlabel("MSA sequence-identity cutoff (%)")
    ax.set_title("MSA Pairformer", fontsize=10)
    ax.set_xticks(sorted(plotted["qid"].unique()))
    style_axes(ax)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scaling = pd.read_csv(data_dir / "contact_scaling_summary.csv")
    pairformer = pd.read_csv(data_dir / "pairformer_qid_summary.csv")

    expected_families = {"ESM2", "ESM-C refit"}
    if set(scaling["family"]) != expected_families:
        raise ValueError(f"Unexpected model families: {set(scaling['family'])}")
    if set(pairformer["qid"]) != {15, 20, 25, 30, 35, 40, 45, 50}:
        raise ValueError("Unexpected Pairformer QID cutoffs")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    plot_scaling(axes[0], scaling)
    plot_pairformer(axes[1], pairformer, args.max_pairformer_qid)

    outcome_handles = [
        Line2D([0], [0], color=COLORS["bio_inter"], marker="o", lw=1.5),
        Line2D([0], [0], color=COLORS["bio_intra"], marker="o", lw=1.5),
        Line2D([0], [0], color=COLORS["xtal_inter"], marker="o", lw=1.5),
    ]
    family_handles = [
        Line2D([0], [0], color="0.25", marker="o", lw=1.5),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="s",
            markerfacecolor="none",
            linestyle="--",
            lw=1.5,
        ),
    ]
    legend = axes[0].legend(
        outcome_handles,
        ["Biological interface", "Biological intrachain", "Crystal interface"],
        frameon=False,
        loc="upper left",
        handlelength=2.0,
    )
    axes[0].add_artist(legend)
    axes[0].legend(
        family_handles,
        ["ESM2 official head", "ESM-C overlay refit"],
        frameon=False,
        loc="upper right",
        handlelength=2.0,
    )

    axes[0].text(
        -0.15,
        1.05,
        "a",
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    axes[1].text(
        -0.12,
        1.05,
        "b",
        transform=axes[1].transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
