#!/usr/bin/env python3
"""Plot same-MSA Jacobian similarity for ESM2 and ESM-C by model size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


MODEL_SPECS = (
    ("150m", "ESM2", 150, "ESM2 150M"),
    ("esmc_300m", "ESM-C", 300, "ESM-C 300M"),
    ("esmc_600m", "ESM-C", 600, "ESM-C 600M"),
    ("650m", "ESM2", 650, "ESM2 650M"),
    ("3b", "ESM2", 3_000, "ESM2 3B"),
    ("esmc_6b", "ESM-C", 6_000, "ESM-C 6B"),
    ("15b", "ESM2", 15_000, "ESM2 15B"),
)
MODEL_ORDER = tuple(spec[0] for spec in MODEL_SPECS)
MODEL_LOOKUP = {
    model: {"family": family, "size_m": size_m, "label": label}
    for model, family, size_m, label in MODEL_SPECS
}
BINS = tuple(range(1, 9))
BIN_LABELS = (
    "0.1–0.2",
    "0.2–0.3",
    "0.3–0.4",
    "0.4–0.5",
    "0.5–0.6",
    "0.6–0.7",
    "0.7–0.8",
    "0.8–0.9",
)
MAIN_FIGURE_PROTEINS = {"3o1n"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    export_root = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "derived",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=root / "data" / "sampled_sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "same_msa",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        help="Optional protein subset; defaults to all sampled proteins.",
    )
    return parser.parse_args()


def expected_rows(sample_dir: Path) -> dict[str, int]:
    result = {}
    for path in sorted(sample_dir.glob("*.json")):
        with path.open() as handle:
            data = json.load(handle)
        result[path.stem] = sum(
            len(data["samples"][str(identity_bin)])
            for identity_bin in BINS
        )
    return result


def load_replicates(
    data_dir: Path,
    expected_by_protein: dict[str, int],
) -> pd.DataFrame:
    frames = []
    for model in MODEL_ORDER:
        model_dir = data_dir / "replicates" / model
        for protein, expected in expected_by_protein.items():
            path = model_dir / f"{protein}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing score table: {path}")
            frame = pd.read_csv(path)
            if len(frame) != expected:
                raise ValueError(
                    f"{path}: expected {expected} rows, found {len(frame)}"
                )
            if set(frame["model"]) != {model}:
                raise ValueError(f"{path}: unexpected model labels")
            frames.append(frame)

    replicates = pd.concat(frames, ignore_index=True)
    keys = ["model", "protein", "identity_bin", "replicate"]
    if replicates.duplicated(keys).any():
        raise ValueError("Duplicate model/protein/bin/replicate rows")
    if not np.isfinite(replicates["spearman_r"]).all():
        raise ValueError("Non-finite Spearman values")
    if not replicates["spearman_r"].between(-1, 1).all():
        raise ValueError("Spearman values outside [-1, 1]")
    return replicates


def summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        replicates.groupby(
            ["model", "protein", "identity_bin"],
            as_index=False,
            sort=False,
        )
        .agg(
            n=("spearman_r", "count"),
            mean_similarity=("spearman_r", "mean"),
            sd_similarity=(
                "spearman_r",
                lambda values: np.std(values, ddof=0),
            ),
        )
    )
    summary["family"] = summary["model"].map(
        lambda model: MODEL_LOOKUP[model]["family"]
    )
    summary["parameters_millions"] = summary["model"].map(
        lambda model: MODEL_LOOKUP[model]["size_m"]
    )
    summary["model_label"] = summary["model"].map(
        lambda model: MODEL_LOOKUP[model]["label"]
    )
    return summary.sort_values(
        ["protein", "parameters_millions", "identity_bin"]
    ).reset_index(drop=True)


def model_colors() -> dict[str, tuple[float, float, float, float]]:
    normalization = Normalize(
        vmin=np.log10(min(spec[2] for spec in MODEL_SPECS)),
        vmax=np.log10(max(spec[2] for spec in MODEL_SPECS)),
    )
    colormap = plt.colormaps["viridis"]
    return {
        model: colormap(normalization(np.log10(size_m)))
        for model, _, size_m, _ in MODEL_SPECS
    }


def draw_protein(
    axis: plt.Axes,
    protein: str,
    summary: pd.DataFrame,
    colors: dict[str, tuple[float, float, float, float]],
    *,
    show_title: bool = True,
) -> None:
    protein_data = summary.loc[summary["protein"].eq(protein)]
    for model in MODEL_ORDER:
        spec = MODEL_LOOKUP[model]
        values = (
            protein_data.loc[protein_data["model"].eq(model)]
            .set_index("identity_bin")
            .reindex(BINS)
        )
        means = values["mean_similarity"].to_numpy(dtype=float)
        deviations = values["sd_similarity"].fillna(0).to_numpy(dtype=float)
        axis.errorbar(
            BINS,
            means,
            yerr=deviations,
            color=colors[model],
            linestyle=":" if spec["family"] == "ESM-C" else "-",
            marker="s" if spec["family"] == "ESM-C" else "o",
            linewidth=1.35,
            elinewidth=0.8,
            capsize=1.5,
            markersize=3.2,
            label=spec["label"],
        )
    axis.set_ylim(0, 1)
    axis.set_xlim(0.75, 8.25)
    axis.set_xticks(BINS, BIN_LABELS, rotation=45, ha="right")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    if show_title:
        axis.set_title(protein.upper())


def save_per_protein(
    protein: str,
    summary: pd.DataFrame,
    output_dir: Path,
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.8))
    draw_protein(axis, protein, summary, colors)
    axis.set_xlabel("Sequence-identity bin")
    axis.set_ylabel("Similarity of coevolution")
    axis.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower right",
    )
    figure.tight_layout()
    stem = output_dir / f"same_msa_jacobian_similarity_ESM2_ESMC_{protein}"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def save_overview(
    proteins: list[str],
    summary: pd.DataFrame,
    output_dir: Path,
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for axis, protein in zip(flat_axes, proteins):
        draw_protein(axis, protein, summary, colors)
    for axis in flat_axes[len(proteins) :]:
        axis.set_visible(False)
    for axis in (axis for axis in axes[-1, :] if axis.get_visible()):
        axis.set_xlabel("Sequence-identity bin")
    for axis in axes[:, 0]:
        axis.set_ylabel("Similarity of coevolution")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Same-MSA categorical-Jacobian similarity across ESM2 and ESM-C sizes",
        fontsize=13,
    )
    stem = output_dir / "same_msa_jacobian_similarity_ESM2_ESMC_all_proteins"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    sample_dir = args.sample_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = expected_rows(sample_dir)
    proteins = sorted(args.proteins) if args.proteins else sorted(expected)
    unknown = set(proteins) - set(expected)
    if unknown:
        raise ValueError(f"Unknown proteins: {sorted(unknown)}")

    replicates = load_replicates(data_dir, expected)
    replicates.to_csv(
        data_dir / "same_msa_jacobian_esm2_esmc_replicates.csv",
        index=False,
    )
    summary = summarize(replicates)
    summary.to_csv(
        data_dir / "same_msa_jacobian_esm2_esmc_plot_values.csv",
        index=False,
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = model_colors()
    for protein in proteins:
        save_per_protein(protein, summary, output_dir, colors)
    supplementary_proteins = [
        protein
        for protein in proteins
        if protein.lower() not in MAIN_FIGURE_PROTEINS
    ]
    if supplementary_proteins:
        save_overview(supplementary_proteins, summary, output_dir, colors)
    print(
        f"Wrote {len(proteins)} per-protein plots and an overview containing "
        f"{len(supplementary_proteins)} supplementary proteins to {output_dir}"
    )


if __name__ == "__main__":
    main()
