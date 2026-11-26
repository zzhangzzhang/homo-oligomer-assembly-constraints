#!/usr/bin/env python3
"""Prepare and plot Pairformer QID-scan data for Figure 4D and Figure S3.

It uses saved evaluation results and does not run MSA Pairformer or select new
predictions.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    root = Path(__file__).resolve().parents[1]
    export_root = root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "qid_vs_random_eval_masked_all.csv",
    )
    parser.add_argument(
        "--panel-data",
        type=Path,
        default=root / "data" / "qid_scan_panel_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "pairformer_qid_scan" / "qid_scan_panels.pdf",
    )
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "pairformer_qid_scan" / "panels",
    )
    return parser.parse_args()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "pdb", "qid", "base_inter", "base_intra", "rand_inter", "rand_intra"
    }
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    clean = frame.dropna().copy()
    qids = {15, 20, 25, 30, 35, 40, 45, 50}
    grouped = clean.groupby("pdb", sort=True)
    if set(clean["qid"]) != qids:
        raise ValueError("Unexpected QID values")
    complete = grouped["qid"].nunique()
    if len(complete) != 537 or not complete.eq(8).all():
        raise ValueError(
            f"Expected 537 proteins with eight QID values; found {len(complete)}"
        )
    rows = []
    for pdb_id, values in grouped:
        q15 = values.loc[values["qid"].eq(15)].iloc[0]
        rows.append(
            {
                "pdb": pdb_id,
                "qid15_interface_recovery": q15["base_inter"],
                "max_qid_scan_interface_recovery": values["base_inter"].max(),
                "qid15_monomer_recovery": q15["base_intra"],
                "max_qid_scan_monomer_recovery": values["base_intra"].max(),
                "max_count_matched_random_interface_recovery": (
                    values["rand_inter"].max()
                ),
                "max_count_matched_random_monomer_recovery": (
                    values["rand_intra"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def draw(axis, x, y, xlabel, ylabel, color):
    axis.scatter(x, y, s=22, alpha=0.55, color=color, linewidths=0)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1)
    axis.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel=xlabel, ylabel=ylabel)
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)


def main():
    args = parse_args()
    panel = prepare(pd.read_csv(args.input))
    args.panel_data.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.panel_data, index=False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
        }
    )
    panel_specs = (
        (
            panel["qid15_monomer_recovery"],
            panel["max_qid_scan_monomer_recovery"],
            "P@X monomer at QID=15",
            "Max P@X monomer with QID scan",
            "#1F77B4",
            "Monomeric contacts",
            "monomer_qid15_vs_max.pdf",
        ),
        (
            panel["qid15_interface_recovery"],
            panel["max_qid_scan_interface_recovery"],
            "P@X interface at QID=15",
            "Max P@X interface with QID scan",
            "#D62728",
            "Homo-oligomeric interface contacts",
            "interface_qid15_vs_max.pdf",
        ),
        (
            panel["max_count_matched_random_interface_recovery"],
            panel["max_qid_scan_interface_recovery"],
            "Max P@X interface with\ncount matching random subsample",
            "Max P@X interface with QID scan",
            "#D62728",
            "Control",
            "interface_random_vs_qid_scan.pdf",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for axis, (x, y, xlabel, ylabel, color, title, _) in zip(axes, panel_specs):
        draw(axis, x, y, xlabel, ylabel, color)
        axis.set_title(title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)

    args.panel_output_dir.mkdir(parents=True, exist_ok=True)
    for x, y, xlabel, ylabel, color, title, filename in panel_specs:
        figure, axis = plt.subplots(figsize=(4.0, 4.0))
        draw(axis, x, y, xlabel, ylabel, color)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(args.panel_output_dir / filename, bbox_inches="tight")
        plt.close(figure)

    print(f"Wrote {args.panel_data}")
    print(f"Wrote {args.output}")
    print(f"Wrote individual panels to {args.panel_output_dir}")


if __name__ == "__main__":
    main()
