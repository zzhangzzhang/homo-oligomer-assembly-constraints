#!/usr/bin/env python
"""05: Compare aligned contact maps to find conformation-exclusive contacts."""

import os
import json
import numpy as np
import pandas as pd
import gemmi
from tqdm import tqdm

# --- Config ---
OUT_DIR = "/path/to/external-data/cfold_clean"
CONFIND_DIR = f"{OUT_DIR}/confind_contacts"
ALN_DIR = f"{OUT_DIR}/alignments"
EXCL_DIR = f"{OUT_DIR}/exclusive_contacts"
CIF_DIR = "/path/to/external-data/pdb_bioassembly_all/"

# --- Functions ---
def read_residue_ids(pdb_id, property_name):
    """Read unique 0-indexed residue IDs from CIF."""
    cif_path = f"{CIF_DIR}{pdb_id.lower()}-assembly1.cif.gz"
    doc = gemmi.cif.read(cif_path)
    block = doc.sole_block()
    ids = [x for x in block.find_loop(property_name)]
    unique = list(dict.fromkeys(ids))
    return [int(x) - 1 for x in unique if x.lstrip("-").isdigit()]

def get_aligned_existing_indices(aln_idx1, aln_idx2, label_ids1, label_ids2):
    """Filter alignment to positions that exist in both structures."""
    set1 = set(label_ids1)
    set2 = set(label_ids2)
    filt1, filt2 = [], []
    for i1, i2 in zip(aln_idx1, aln_idx2):
        if i1 in set1 and i2 in set2:
            filt1.append(i1)
            filt2.append(i2)
    return filt1, filt2

# --- Main ---
if __name__ == "__main__":
    os.makedirs(EXCL_DIR, exist_ok=True)

    with open(f"{ALN_DIR}/cfold_seq_aln.json") as f:
        aln_info = json.load(f)
    df = pd.read_csv(f"{OUT_DIR}/filtered_df.csv")

    done, skip = 0, 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Comparing contacts"):
        id1, id2 = row["ID1"], row["ID2"]
        try:
            # Load contact maps
            cm1 = np.load(f"{CONFIND_DIR}/{id1}_processed.npy")
            cm2 = np.load(f"{CONFIND_DIR}/{id2}_processed.npy")

            # Get structural residue indices
            label1 = read_residue_ids(id1, "_atom_site.label_seq_id")
            label2 = read_residue_ids(id2, "_atom_site.label_seq_id")

            # Filter alignment to positions existing in both structures
            aln1, aln2 = get_aligned_existing_indices(
                aln_info[id1], aln_info[id2], label1, label2
            )

            # Extract aligned sub-matrices
            cm1_aln = cm1[np.ix_(aln1, aln1)]
            cm2_aln = cm2[np.ix_(aln2, aln2)]

            # Exclusive contacts: present in one but not the other
            excl1 = ((cm1_aln == 1) & (cm2_aln == 0)).astype(np.int8)  # conf1-only
            excl2 = ((cm2_aln == 1) & (cm1_aln == 0)).astype(np.int8)  # conf2-only

            # Save
            prefix = f"{EXCL_DIR}/{idx}_{id1}_{id2}"
            np.savez_compressed(
                f"{prefix}.npz",
                excl1=excl1, excl2=excl2,
                cm1_aln=cm1_aln, cm2_aln=cm2_aln,
                aln_idx1=np.array(aln1), aln_idx2=np.array(aln2),
            )
            done += 1
        except Exception as e:
            print(f"  SKIP {id1}-{id2}: {e}")
            skip += 1

    print(f"Done: {done} pairs, {skip} skipped")
