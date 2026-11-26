#!/usr/bin/env python
"""07: TMalign ALL predicted structures vs both experimental conformations.

For each pair, masking condition, model, and seed: compute TM-scores against
both exp structures.  Only 100% masking conditions (no_mask, mask_excl1,
mask_excl2) are processed; 50% masking jobs are skipped.

TMalign calls are parallelised across all available CPUs.
"""

import os
import glob
import subprocess
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Config ---
OUT_DIR = "/path/to/external-data/cfold_clean"
PRED_DIR = f"{OUT_DIR}/predictions"
PDB_DIR = f"{OUT_DIR}/exp_pdb"
RESULT_DIR = f"{OUT_DIR}/tmalign_results"
USALIGN = "/usr/local/bin/USalign"

ALLOWED_MASKS = {"no_mask", "mask_excl1", "mask_excl2"}
N_WORKERS = min(os.cpu_count() or 4, 120)


# --- Functions ---

def run_tmalign(pdb1, pdb2):
    """Run USalign and return TM-score normalized to pdb2 (the reference)."""
    try:
        result = subprocess.run(
            [USALIGN, pdb1, pdb2],
            capture_output=True, text=True, timeout=60,
        )
        tm_scores = []
        for line in result.stdout.split("\n"):
            if line.startswith("TM-score="):
                match = re.search(r"TM-score=\s*([\d.]+)", line)
                if match:
                    tm_scores.append(float(match.group(1)))
        if len(tm_scores) >= 2:
            return tm_scores[1]
        elif tm_scores:
            return tm_scores[0]
    except Exception:
        pass
    return None


def parse_jobname(jobname):
    """Parse jobname -> (pred_pdb, id1, id2, mask_label).

    Formats:
      {pdb}_{id1}_{id2}_mask_excl1        -> mask_excl1
      {pdb}_{id1}_{id2}_mask_excl1_50     -> mask_excl1_50
      {pdb}_{id1}_{id2}_no_mask           -> no_mask
    """
    parts = jobname.split("_")
    if len(parts) < 4:
        return None
    pdb_id = parts[0]
    id1 = parts[1]
    id2 = parts[2]
    mask_label = "_".join(parts[3:])
    return pdb_id, id1, id2, mask_label


def _process_one(pred_pdb, exp1, exp2, pdb_id, id1, id2, mask_label):
    """Score a single predicted PDB against both experimental conformations."""
    fname = os.path.basename(pred_pdb).replace(".pdb", "")
    model_name = "_".join(fname.split("_")[:3])
    seed = fname.split("seed")[-1]

    tm1 = run_tmalign(pred_pdb, exp1)
    tm2 = run_tmalign(pred_pdb, exp2)

    return {
        "pred_pdb_id": pdb_id,
        "pair_id1": id1,
        "pair_id2": id2,
        "mask_condition": mask_label,
        "model": model_name,
        "seed": int(seed),
        "pdb_file": fname,
        "tm_vs_conf1": tm1,
        "tm_vs_conf2": tm2,
        "delta_tm": (tm2 - tm1) if (tm1 is not None and tm2 is not None) else None,
    }


# --- Main ---
if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    out_path = f"{RESULT_DIR}/tmalign_all.csv"

    # Load existing results to skip already-processed PDBs
    already_done = set()
    existing_rows = []
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        prev = prev[prev["mask_condition"].isin(ALLOWED_MASKS)]
        existing_rows = prev.to_dict("records")
        already_done = set(zip(prev["pred_pdb_id"], prev["pair_id1"],
                               prev["pair_id2"], prev["mask_condition"],
                               prev["pdb_file"]))
        print(f"Loaded {len(already_done)} already-processed entries (kept 100% only)")

    job_dirs = sorted(glob.glob(f"{PRED_DIR}/*/pdb/best.npz"))
    print(f"Found {len(job_dirs)} completed jobs")

    # Build flat list of work items, skipping already-done
    work_items = []
    skipped_50 = 0
    skipped_done = 0
    for best_npz in job_dirs:
        pdb_dir = os.path.dirname(best_npz)
        job_dir = os.path.dirname(pdb_dir)
        jobname = os.path.basename(job_dir)

        parsed = parse_jobname(jobname)
        if not parsed:
            print(f"  SKIP unrecognized: {jobname}")
            continue
        pdb_id, id1, id2, mask_label = parsed

        if mask_label not in ALLOWED_MASKS:
            skipped_50 += 1
            continue

        exp1 = f"{PDB_DIR}/{id1}.pdb"
        exp2 = f"{PDB_DIR}/{id2}.pdb"
        if not os.path.exists(exp1) or not os.path.exists(exp2):
            continue

        for pred_pdb in sorted(glob.glob(f"{pdb_dir}/model_*_seed*.pdb")):
            fname = os.path.basename(pred_pdb).replace(".pdb", "")
            key = (pdb_id, id1, id2, mask_label, fname)
            if key in already_done:
                skipped_done += 1
                continue
            work_items.append((pred_pdb, exp1, exp2, pdb_id, id1, id2, mask_label))

    print(f"Skipped {skipped_50} jobs with 50% masking")
    print(f"Skipped {skipped_done} already-processed model PDBs")
    print(f"New TMalign tasks: {len(work_items)}  (with {N_WORKERS} workers)")

    new_rows = []
    if work_items:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_process_one, *item): item for item in work_items}
            for future in tqdm(as_completed(futures), total=len(futures)):
                new_rows.append(future.result())

    all_rows = existing_rows + new_rows
    results = pd.DataFrame(all_rows)
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} results to {out_path}  "
          f"({len(existing_rows)} existing + {len(new_rows)} new)")

    if not results.empty:
        print("\n=== Summary ===")
        for cond in sorted(results["mask_condition"].unique()):
            sub = results[results["mask_condition"] == cond].dropna(subset=["delta_tm"])
            if not sub.empty:
                print(f"  {cond:20s}  mean_delta_tm={sub['delta_tm'].mean():+.4f}  "
                      f"median={sub['delta_tm'].median():+.4f}  "
                      f"mean_tm1={sub['tm_vs_conf1'].mean():.3f}  "
                      f"mean_tm2={sub['tm_vs_conf2'].mean():.3f}  n={len(sub)}")
