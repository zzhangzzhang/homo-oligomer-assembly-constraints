#!/usr/bin/env python3
"""Plot AF2 query-bias and masking distributions as violins plus jitter."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TM_BLUE = "#1F77B4"
DOCKQ_RED = "#D62728"
RANDOM_SEED = 20260731
FONT_SCALE = 1.5

QUERY_ORDER = [
    "QID 0, query bias on",
    "QID 30, query bias on",
    "QID 0, query bias off",
    "QID 30, query bias off",
]
QUERY_LABELS = [
    "Bias on\nQID 0",
    "Bias on\nQID 30",
    "Bias off\nQID 0",
    "Bias off\nQID 30",
]
QID_ORDER = [0, 30]
QID_LABELS = ["QID 0", "QID 30"]
MASKING_ORDER = [
    "baseline",
    "offdiag",
    "ondiag",
    "inter_expand",
    "intra_expand",
]
MASKING_LABELS = [
    "Baseline",
    "Mask inter\nblocks",
    "Mask intra\nblocks",
    "Inter-exclusive\nexpanded",
    "Intra-exclusive\nexpanded",
]
MASKING_DELTA_ORDER = MASKING_ORDER[1:]
MASKING_DELTA_LABELS = MASKING_LABELS[1:]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    export_root = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "af2",
    )
    return parser.parse_args()


def draw_violin_jitter(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    color: str,
    *,
    is_delta: bool,
    seed_offset: int,
) -> None:
    positions = np.arange(len(groups))
    violin = ax.violinplot(
        groups,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=120,
        bw_method="scott",
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25 if color == DOCKQ_RED else 0.28)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    for position, values in zip(positions, groups):
        jitter = rng.uniform(-0.25, 0.25, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=4.0,
            color=color,
            alpha=0.14,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        median = np.median(values)
        ax.scatter(
            position,
            median,
            s=25,
            color="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_xticks(
        positions,
        labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if is_delta:
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        maximum = max(abs(float(values.min())) for values in groups)
        maximum = max(maximum, max(abs(float(values.max())) for values in groups))
        limit = min(1.0, max(0.05, maximum * 1.05))
        ax.set_ylim(-limit, limit)
    else:
        ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_two_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    order: list,
    labels: list[str],
    metric_columns: tuple[str, str],
    ylabel_suffix: str,
    title: str,
    output_stem: Path,
    is_delta: bool,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.0),
        constrained_layout=True,
    )
    for index, (axis, metric, metric_name, color) in enumerate(
        zip(
            axes,
            metric_columns,
            ("TM-score", "DockQ"),
            (TM_BLUE, DOCKQ_RED),
        )
    ):
        groups = [
            frame.loc[frame[group_column].eq(value), metric]
            .to_numpy(dtype=float)
            for value in order
        ]
        if any(len(group) == 0 for group in groups):
            raise ValueError(f"Missing plot group for {group_column}: {order}")
        draw_violin_jitter(
            axis,
            groups,
            labels,
            color,
            is_delta=is_delta,
            seed_offset=index,
        )
        axis.set_ylabel(f"{metric_name}{ylabel_suffix}")
        axis.set_title(metric_name)
        axis.text(
            -0.08,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=13 * FONT_SCALE,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=13 * FONT_SCALE)
    fig.text(
        0.5,
        -0.055,
        "109 proteins; 25 model/seed predictions per protein. "
        "Dots are repeated predictions; white-centered markers show medians.",
        ha="center",
        fontsize=8.5 * FONT_SCALE,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 10 * FONT_SCALE,
            "axes.titlesize": 10 * FONT_SCALE,
            "axes.labelsize": 12 * FONT_SCALE,
            "xtick.labelsize": 9.5 * FONT_SCALE,
            "ytick.labelsize": 10 * FONT_SCALE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    query = pd.read_csv(
        args.processed_dir / "af2_query_bias_observations.csv"
    )
    query_deltas = pd.read_csv(
        args.processed_dir / "af2_query_bias_paired_deltas.csv"
    )
    masking = pd.read_csv(
        args.processed_dir / "af2_masking_observations.csv"
    )
    masking_deltas = pd.read_csv(
        args.processed_dir / "af2_masking_paired_deltas.csv"
    )

    plot_two_metrics(
        query,
        group_column="condition",
        order=QUERY_ORDER,
        labels=QUERY_LABELS,
        metric_columns=("tm_score", "dockq"),
        ylabel_suffix="",
        title="AlphaFold2-Multimer query bias and MSA identity filtering",
        output_stem=args.output_dir / "figure_af2_query_bias_absolute",
        is_delta=False,
    )
    plot_two_metrics(
        query_deltas,
        group_column="qid",
        order=QID_ORDER,
        labels=QID_LABELS,
        metric_columns=("tm_score_delta", "dockq_delta"),
        ylabel_suffix=" change (bias on − bias off)",
        title="Paired effect of enabling query bias",
        output_stem=args.output_dir / "figure_af2_query_bias_paired_deltas",
        is_delta=True,
    )
    plot_two_metrics(
        masking,
        group_column="strategy",
        order=MASKING_ORDER,
        labels=MASKING_LABELS,
        metric_columns=("tm_score", "dockq"),
        ylabel_suffix="",
        title="AlphaFold2-Multimer covariance-masking strategies",
        output_stem=args.output_dir / "figure_af2_masking_absolute",
        is_delta=False,
    )
    plot_two_metrics(
        masking_deltas,
        group_column="strategy",
        order=MASKING_DELTA_ORDER,
        labels=MASKING_DELTA_LABELS,
        metric_columns=("tm_score_delta", "dockq_delta"),
        ylabel_suffix=" change (strategy − baseline)",
        title="Paired effect of covariance masking",
        output_stem=args.output_dir / "figure_af2_masking_paired_deltas",
        is_delta=True,
    )


if __name__ == "__main__":
    main()
