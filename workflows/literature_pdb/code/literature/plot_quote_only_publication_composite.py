#!/usr/bin/env python3
"""Plot the final four-panel figure from quote-only literature classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quote-only-composite")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CLASS_ORDER = ["Monomer-only", "Multiple states", "Oligomer-only"]
CLASS_COLORS = {
    "Monomer-only": "#1F77B4",
    "Multiple states": "#9467BD",
    "Oligomer-only": "#D62728",
}
RECOVERY_ORDER = ["Zero", "0–0.2", "0.2–0.4", ">0.4"]
ALPHA = 0.85


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color("black")
        axis.spines[side].set_linewidth(1.0)
    axis.tick_params(
        direction="out",
        color="black",
        labelcolor="black",
        top=False,
        right=False,
        length=4,
        width=1,
    )


def recovery_bin(hits: Any, recall: Any) -> str:
    hits = pd.to_numeric(pd.Series([hits]), errors="coerce").iloc[0]
    recall = pd.to_numeric(pd.Series([recall]), errors="coerce").iloc[0]
    if not np.isfinite(hits) or hits <= 0:
        return "Zero"
    if recall <= 0.2:
        return "0–0.2"
    if recall <= 0.4:
        return "0.2–0.4"
    return ">0.4"


def run(
    funnel_path: Path,
    composition_path: Path,
    homo_joined: Path,
    monomer_pdb_table: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 16,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    funnel = pd.read_csv(funnel_path).set_index("stage")
    mini_base = funnel.loc[
        "canonical_exact_member_cards_sent_to_gpt_5_4_mini"
    ]
    mini_candidates = funnel.loc[
        "cards_with_any_mini_state_or_event_candidate"
    ]
    base = np.asarray(
        [
            mini_base["monomer_unique_pdb_uniprot_records"],
            mini_base["homo_unique_pdb_uniprot_records"],
        ],
        dtype=int,
    )
    found = np.asarray(
        [
            mini_candidates["monomer_unique_pdb_uniprot_records"],
            mini_candidates["homo_unique_pdb_uniprot_records"],
        ],
        dtype=int,
    )
    found_pct = 100 * found / base

    composition = pd.read_csv(composition_path)
    composition = composition[
        composition["direct_exact_class"].isin(CLASS_ORDER)
    ].copy()
    class_pivot = (
        composition.pivot(
            index="branch",
            columns="direct_exact_class",
            values="unique_pdb_uniprot_records",
        )
        .reindex(["monomer_linked", "homo_linked"])
        .reindex(columns=CLASS_ORDER)
    )

    homo = pd.read_csv(homo_joined)
    homo = homo[
        homo["literature_class"].isin(CLASS_ORDER)
        & pd.to_numeric(
            homo["intra_recall_topx_qid35"], errors="coerce"
        ).gt(0.4)
    ].copy()
    if homo.duplicated(["pdb_normalized", "uniprot_normalized"]).any():
        raise RuntimeError("duplicate homo PDB×UniProt plotting rows")
    homo["recovery_bin"] = [
        recovery_bin(hits, recall)
        for hits, recall in zip(
            homo["n_bio_hits_topx_qid35"],
            homo["bio_recall_topx_qid35"],
        )
    ]
    homo_counts = (
        homo.groupby(["recovery_bin", "literature_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=RECOVERY_ORDER, columns=CLASS_ORDER, fill_value=0)
    )

    monomer = pd.read_csv(monomer_pdb_table)
    required = {
        "pdb",
        "uniprot",
        "direct_current_paper_exact_class",
        "best_recall_interface_recall",
        "best_recall_pisa_dissociation_energy_kcal_mol",
    }
    missing = required - set(monomer)
    if missing:
        raise ValueError(f"monomer table missing columns: {sorted(missing)}")
    monomer = monomer[
        monomer["direct_current_paper_exact_class"].isin(CLASS_ORDER)
    ].copy()
    scatter = monomer.dropna(
        subset=[
            "best_recall_interface_recall",
            "best_recall_pisa_dissociation_energy_kcal_mol",
        ]
    )
    if scatter.duplicated(["pdb", "uniprot"]).any():
        raise RuntimeError("duplicate monomer PDB×UniProt plotting rows")

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18.0, 5.8),
        gridspec_kw={"width_ratios": [1, 1, 2, 2]},
    )
    x_two = np.arange(2)
    database_labels = ["Monomer", "Oligomer"]

    axes[0].bar(
        x_two,
        found_pct,
        width=0.68,
        color=matplotlib.colors.to_rgba("#BDBDBD", ALPHA),
        edgecolor="#666666",
        linewidth=1,
    )
    axes[0].bar(
        x_two,
        100 - found_pct,
        width=0.68,
        bottom=found_pct,
        color=matplotlib.colors.to_rgba("#FFFFFF", ALPHA),
        edgecolor="#666666",
        linewidth=1,
    )
    axes[0].set_ylabel("PDB records (%)")
    axes[0].set_xlabel("PDB database label")
    axes[0].set_xticks(x_two, database_labels, rotation=35, ha="right")
    axes[0].set_ylim(0, 110)
    for index, total in enumerate(base):
        axes[0].text(index, 102, f"{total:,}", ha="center", va="bottom")

    totals = class_pivot.sum(axis=1).to_numpy()
    bottoms = np.zeros(2)
    for class_name in CLASS_ORDER:
        values = class_pivot[class_name].to_numpy(dtype=float)
        percentages = 100 * values / totals
        axes[1].bar(
            x_two,
            percentages,
            width=0.68,
            bottom=bottoms,
            color=matplotlib.colors.to_rgba(
                CLASS_COLORS[class_name], ALPHA
            ),
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms += percentages
    axes[1].set_ylabel("Literature classification (%)")
    axes[1].set_xlabel("PDB database label")
    axes[1].set_xticks(x_two, database_labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 110)
    for index, total in enumerate(totals):
        axes[1].text(index, 102, f"{int(total):,}", ha="center", va="bottom")

    for class_name in CLASS_ORDER:
        subset = scatter[
            scatter["direct_current_paper_exact_class"].eq(class_name)
        ]
        axes[2].scatter(
            subset["best_recall_interface_recall"],
            subset["best_recall_pisa_dissociation_energy_kcal_mol"],
            color=CLASS_COLORS[class_name],
            s=30,
            alpha=ALPHA,
            linewidths=0,
        )
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Max interface recovery")
    axes[2].set_ylabel(
        "Same-interface PISA dissociation\nenergy (kcal/mol)"
    )
    axes[2].set_title("PDB label: monomer", pad=18)
    axes[2].text(
        0.99,
        0.02,
        f"n={len(scatter):,} PDB records",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
    )

    homo_percent = homo_counts.div(homo_counts.sum(axis=1), axis=0) * 100
    x_recovery = np.arange(len(RECOVERY_ORDER))
    bottoms = np.zeros(len(RECOVERY_ORDER))
    for class_name in CLASS_ORDER:
        values = homo_percent[class_name].fillna(0).to_numpy()
        axes[3].bar(
            x_recovery,
            values,
            width=0.72,
            bottom=bottoms,
            color=matplotlib.colors.to_rgba(
                CLASS_COLORS[class_name], ALPHA
            ),
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms += values
    axes[3].set_ylabel("Literature classification (%)")
    axes[3].set_xlabel("Interface recovery")
    axes[3].set_xticks(x_recovery, RECOVERY_ORDER)
    axes[3].set_ylim(0, 110)
    axes[3].set_title("PDB label: homo-oligomer", pad=18)
    for index, total in enumerate(homo_counts.sum(axis=1)):
        axes[3].text(index, 102, f"{int(total):,}", ha="center", va="bottom")

    for axis in axes:
        clean_axis(axis)
    for label, axis in zip("BCDE", axes):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=20,
            fontweight="bold",
        )

    evidence_handles = [
        Patch(facecolor="#BDBDBD", edgecolor="#666666", label="Found"),
        Patch(facecolor="white", edgecolor="#666666", label="Not found"),
    ]
    class_handles = [
        Patch(facecolor=CLASS_COLORS[name], edgecolor="none", label=name)
        for name in CLASS_ORDER
    ]
    first_legend = figure.legend(
        handles=evidence_handles,
        title="Oligomerisation evidence candidate",
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.01, -0.02),
    )
    figure.add_artist(first_legend)
    figure.legend(
        handles=class_handles,
        title="Literature classification",
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.64, -0.02),
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.86,
        bottom=0.29,
        wspace=0.36,
    )

    prefix = output_dir / "quote_only_four_panel_publication_composite"
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
    ):
        figure.savefig(prefix.with_suffix(suffix), **kwargs)
    plt.close(figure)

    funnel.reset_index().to_csv(
        output_dir / "panel_b_retrieval_funnel.csv", index=False
    )
    composition.to_csv(
        output_dir / "panel_c_literature_composition.csv", index=False
    )
    scatter.to_csv(output_dir / "panel_d_monomer_scatter_rows.csv", index=False)
    homo.to_csv(output_dir / "panel_e_homo_rows.csv", index=False)
    summary = {
        "mini_input_records": base.tolist(),
        "mini_candidate_records": found.tolist(),
        "literature_three_class_records": {
            branch: {
                name: int(class_pivot.loc[branch, name])
                for name in CLASS_ORDER
            }
            for branch in class_pivot.index
        },
        "monomer_scatter_records": len(scatter),
        "homo_plotted_records": len(homo),
        "homo_recovery_bin_counts": {
            name: int(value)
            for name, value in homo_counts.sum(axis=1).items()
        },
        "inputs": {
            str(path): sha256(path)
            for path in (
                funnel_path,
                composition_path,
                homo_joined,
                monomer_pdb_table,
            )
        },
        "style": {
            "alpha": ALPHA,
            "colors": CLASS_COLORS,
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--homo-joined", type=Path, required=True)
    parser.add_argument("--monomer-pdb-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.funnel,
        args.composition,
        args.homo_joined,
        args.monomer_pdb_table,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
