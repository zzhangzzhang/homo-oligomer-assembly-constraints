#!/usr/bin/env python3
"""Reproduce the six final contact-map source figures from saved matrices.

This script performs plotting and deterministic top-X selection only. It does
not run ESM2 or MSA Pairformer inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MIN_SEPARATION = 6
BLUE = "#0057D9"
RED = "#D62728"
GOLD = "#F2C94C"
GRAY = "#A6A6A6"

matplotlib.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def valid_pair_mask(resolved: np.ndarray, length: int) -> np.ndarray:
    return np.outer(resolved.astype(bool), resolved.astype(bool)) & np.triu(
        np.ones((length, length), dtype=bool), k=MIN_SEPARATION
    )


def select_top_x(
    prediction: np.ndarray,
    valid: np.ndarray,
    intra_only: np.ndarray,
    inter_only: np.ndarray,
    overlap: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    reference = intra_only | inter_only | overlap
    top_x = int(np.count_nonzero(reference))
    selected = np.zeros(prediction.shape, dtype=bool)
    if top_x:
        scores = np.where(valid, prediction, -np.inf)
        flat = np.argpartition(scores.ravel(), -top_x)[-top_x:]
        flat = flat[
            np.argsort(scores.ravel()[flat], kind="mergesort")[::-1]
        ]
        rows, columns = np.unravel_index(flat, prediction.shape)
        selected[rows, columns] = True
    union_hits = int(np.count_nonzero(selected & reference))
    return selected, {
        "top_x": top_x,
        "union_hits": union_hits,
        "union_precision_top_x": union_hits / top_x if top_x else float("nan"),
        "intra_only_contacts": int(np.count_nonzero(intra_only)),
        "inter_only_contacts": int(np.count_nonzero(inter_only)),
        "shared_contacts": int(np.count_nonzero(overlap)),
        "intra_only_hits": int(np.count_nonzero(selected & intra_only)),
        "inter_only_hits": int(np.count_nonzero(selected & inter_only)),
        "shared_hits": int(np.count_nonzero(selected & overlap)),
    }


def load_prediction(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path) if path.suffix == ".txt" else np.load(path)
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def load_reference(data_dir: Path, pdb_id: str) -> dict[str, np.ndarray | int]:
    target_dir = data_dir / pdb_id
    intra = np.load(target_dir / "intra.npy").astype(bool)
    inter = np.load(target_dir / "inter.npy").astype(bool)
    resolved = np.load(target_dir / "resolved_mask.npy").astype(bool)
    length = intra.shape[0]
    if intra.shape != (length, length) or inter.shape != (length, length):
        raise ValueError(f"{pdb_id}: invalid contact-matrix shapes")
    if resolved.shape != (length,):
        raise ValueError(f"{pdb_id}: invalid resolved-mask shape")
    valid = valid_pair_mask(resolved, length)
    return {
        "length": length,
        "valid": valid,
        "intra_only": intra & ~inter & valid,
        "inter_only": inter & ~intra & valid,
        "overlap": intra & inter & valid,
    }


def scatter_mask(
    ax: plt.Axes,
    mask: np.ndarray,
    color: str,
    dot_size: float,
    *,
    offset: int = 0,
    rasterized: bool = True,
) -> None:
    rows, columns = np.where(mask)
    if len(rows):
        ax.scatter(
            rows + offset,
            columns + offset,
            s=dot_size,
            c=color,
            edgecolors="none",
            rasterized=rasterized,
        )


def draw_overlay(
    ax: plt.Axes,
    prediction: np.ndarray,
    valid: np.ndarray,
    intra_only: np.ndarray,
    inter_only: np.ndarray,
    overlap: np.ndarray,
    title: str,
    dot_size: float,
    *,
    residue_origin: int = 0,
    rasterize_dots: bool = True,
) -> tuple[np.ndarray, dict[str, int | float]]:
    length = prediction.shape[0]
    upper = np.where(
        np.triu(np.ones((length, length), dtype=bool)), prediction, np.nan
    )
    if residue_origin:
        left = residue_origin + 0.5
        right = residue_origin + length + 0.5
        extent = (left, right, right, left)
        scatter_offset = residue_origin + 1
    else:
        left, right = -0.5, length - 0.5
        extent = None
        scatter_offset = 0
    ax.imshow(
        upper,
        origin="upper",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="none",
        rasterized=True,
        extent=extent,
    )
    selected, metrics = select_top_x(
        prediction, valid, intra_only, inter_only, overlap
    )
    scatter_mask(
        ax,
        selected & ~(intra_only | inter_only | overlap),
        GRAY,
        dot_size,
        offset=scatter_offset,
        rasterized=rasterize_dots,
    )
    scatter_mask(
        ax,
        selected & intra_only,
        BLUE,
        dot_size,
        offset=scatter_offset,
        rasterized=rasterize_dots,
    )
    scatter_mask(
        ax,
        selected & inter_only,
        RED,
        dot_size,
        offset=scatter_offset,
        rasterized=rasterize_dots,
    )
    scatter_mask(
        ax,
        selected & overlap,
        GOLD,
        dot_size,
        offset=scatter_offset,
        rasterized=rasterize_dots,
    )
    ax.set_xlim(left, right)
    ax.set_ylim(right, left)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Residue i")
    ax.set_ylabel("Residue j")
    ax.tick_params(direction="out", length=2.5)
    return selected, metrics


def draw_ground_truth(
    ax: plt.Axes,
    reference: dict[str, np.ndarray | int],
    dot_size: float,
    *,
    rasterize_dots: bool = True,
) -> None:
    scatter_mask(
        ax,
        np.asarray(reference["intra_only"]),
        BLUE,
        dot_size,
        rasterized=rasterize_dots,
    )
    scatter_mask(
        ax,
        np.asarray(reference["inter_only"]),
        RED,
        dot_size,
        rasterized=rasterize_dots,
    )
    scatter_mask(
        ax,
        np.asarray(reference["overlap"]),
        GOLD,
        dot_size,
        rasterized=rasterize_dots,
    )
    length = int(reference["length"])
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(length - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Experimental contacts")
    ax.set_xlabel("Residue i")
    ax.set_ylabel("Residue j")
    ax.tick_params(direction="out", length=2.5)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def save_processed(
    processed_dir: Path,
    stem: str,
    reference: dict[str, np.ndarray | int],
    predictions: dict[str, np.ndarray],
    selections: dict[str, np.ndarray],
    residue_numbers: np.ndarray,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "valid_pair_mask": np.asarray(reference["valid"]),
        "intra_only": np.asarray(reference["intra_only"]),
        "inter_only": np.asarray(reference["inter_only"]),
        "shared": np.asarray(reference["overlap"]),
        "residue_numbers": residue_numbers,
    }
    arrays.update({f"prediction_{key}": value for key, value in predictions.items()})
    arrays.update({f"selected_top_x_{key}": value for key, value in selections.items()})
    np.savez_compressed(processed_dir / f"{stem}.npz", **arrays)


def plot_esm2_3b(
    pdb_id: str,
    data_dir: Path,
    output_dir: Path,
    processed_dir: Path,
    dot_size: float,
) -> dict[str, dict[str, int | float]]:
    reference = load_reference(data_dir, pdb_id)
    prediction = load_prediction(data_dir / pdb_id / "esm2_3b.npy")
    length = int(reference["length"])
    if prediction.shape != (length, length):
        raise ValueError(f"{pdb_id}: prediction shape mismatch")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
    draw_ground_truth(axes[0], reference, dot_size, rasterize_dots=False)
    selected, metrics = draw_overlay(
        axes[1],
        prediction,
        np.asarray(reference["valid"]),
        np.asarray(reference["intra_only"]),
        np.asarray(reference["inter_only"]),
        np.asarray(reference["overlap"]),
        "ESM2-3B contact head",
        dot_size,
        rasterize_dots=False,
    )
    axes[1].set_ylabel("")
    fig.suptitle(pdb_id.upper(), fontsize=12)
    fig.tight_layout()
    stem = f"{pdb_id}_esm2_3b_contact_maps"
    save_figure(fig, output_dir, stem)
    save_processed(
        processed_dir,
        stem,
        reference,
        {"esm2_3b": prediction},
        {"esm2_3b": selected},
        np.arange(1, length + 1),
    )
    return {"esm2_3b": metrics}


def crop_reference(
    reference: dict[str, np.ndarray | int], window: slice
) -> dict[str, np.ndarray | int]:
    return {
        "length": window.stop - window.start,
        "valid": np.asarray(reference["valid"])[window, window],
        "intra_only": np.asarray(reference["intra_only"])[window, window],
        "inter_only": np.asarray(reference["inter_only"])[window, window],
        "overlap": np.asarray(reference["overlap"])[window, window],
    }


def plot_3o1n(
    data_dir: Path, output_dir: Path, processed_dir: Path
) -> dict[str, dict[str, int | float]]:
    pdb_id = "3o1n"
    full_reference = load_reference(data_dir, pdb_id)
    full_length = int(full_reference["length"])
    start, requested_end = 150, 280
    end = min(requested_end, full_length)
    window = slice(start - 1, end)
    origin = window.start
    reference = crop_reference(full_reference, window)
    residue_numbers = np.arange(start, end + 1)
    metrics: dict[str, dict[str, int | float]] = {}

    comparisons = (
        (
            "3o1n_res150-280_esm2_150m_vs_15b",
            (
                ("esm2_150m", "ESM2-150M contact head"),
                ("esm2_15b", "ESM2-15B contact head"),
            ),
        ),
        (
            "3o1n_res150-280_pairformer_qid15_vs_qid35",
            (
                ("pairformer_qid15", "Pairformer QID15"),
                ("pairformer_qid35", "Pairformer QID35"),
            ),
        ),
    )
    for stem, panels in comparisons:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
        predictions: dict[str, np.ndarray] = {}
        selections: dict[str, np.ndarray] = {}
        for ax, (key, title) in zip(axes, panels):
            suffix = ".txt" if key.startswith("pairformer") else ".npy"
            matrix = load_prediction(data_dir / pdb_id / f"{key}{suffix}")
            matrix = matrix[window, window]
            predictions[key] = matrix
            selected, panel_metrics = draw_overlay(
                ax,
                matrix,
                np.asarray(reference["valid"]),
                np.asarray(reference["intra_only"]),
                np.asarray(reference["inter_only"]),
                np.asarray(reference["overlap"]),
                f"{title} ({start}-{end})",
                10,
                residue_origin=origin,
            )
            selections[key] = selected
            metrics[key] = panel_metrics
        axes[1].set_ylabel("")
        fig.tight_layout()
        save_figure(fig, output_dir, stem)
        save_processed(
            processed_dir,
            stem,
            reference,
            predictions,
            selections,
            residue_numbers,
        )
    return metrics


def plot_pairformer_target(
    pdb_id: str,
    comparison_qid: int,
    data_dir: Path,
    output_dir: Path,
    processed_dir: Path,
) -> dict[str, dict[str, int | float]]:
    reference = load_reference(data_dir, pdb_id)
    length = int(reference["length"])
    panel_keys = ("pairformer_qid15", f"pairformer_qid{comparison_qid}")
    predictions = {
        key: load_prediction(data_dir / pdb_id / f"{key}.txt")
        for key in panel_keys
    }
    for key, matrix in predictions.items():
        if matrix.shape != (length, length):
            raise ValueError(f"{pdb_id} {key}: prediction shape mismatch")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
    selections: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, int | float]] = {}
    for ax, key in zip(axes, panel_keys):
        qid = key[len("pairformer_qid") :]
        selected, panel_metrics = draw_overlay(
            ax,
            predictions[key],
            np.asarray(reference["valid"]),
            np.asarray(reference["intra_only"]),
            np.asarray(reference["inter_only"]),
            np.asarray(reference["overlap"]),
            f"MSA-Pairformer QID{qid}",
            10,
        )
        selections[key] = selected
        metrics[key] = panel_metrics
    axes[1].set_ylabel("")
    fig.suptitle(pdb_id.upper(), fontsize=12)
    fig.tight_layout()
    stem = f"{pdb_id}_pairformer_qid15_vs_qid{comparison_qid}"
    save_figure(fig, output_dir, stem)
    save_processed(
        processed_dir,
        stem,
        reference,
        predictions,
        selections,
        np.arange(1, length + 1),
    )
    return metrics


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    export_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_root / "outputs" / "reproduced" / "contact_maps",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=script_dir.parent / "data" / "processed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = {
        "1bjp_esm2_3b_contact_maps": plot_esm2_3b(
            "1bjp", args.data_dir, args.output_dir, args.processed_dir, 60
        ),
        "1r5y_esm2_3b_contact_maps": plot_esm2_3b(
            "1r5y", args.data_dir, args.output_dir, args.processed_dir, 10
        ),
        **{
            f"3o1n_{key}": value
            for key, value in plot_3o1n(
                args.data_dir, args.output_dir, args.processed_dir
            ).items()
        },
        "3epw_pairformer_qid15_vs_qid30": plot_pairformer_target(
            "3epw", 30, args.data_dir, args.output_dir, args.processed_dir
        ),
        "3mkm_pairformer_qid15_vs_qid35": plot_pairformer_target(
            "3mkm", 35, args.data_dir, args.output_dir, args.processed_dir
        ),
    }
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    with (args.processed_dir / "top_x_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"Wrote figures to {args.output_dir}")
    print(f"Wrote processed panel data to {args.processed_dir}")


if __name__ == "__main__":
    main()
