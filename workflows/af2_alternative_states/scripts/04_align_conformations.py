#!/usr/bin/env python
"""04: Align sequences for each pair using hhalign, save index mappings."""

import os
import json
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Config ---
OUT_DIR = "/path/to/external-data/cfold_clean"
ALN_DIR = f"{OUT_DIR}/alignments"
HHALIGN = "/path/to/external-home/hhsuite/bin/hhalign"

# --- Functions ---
def parse_hhalign_output(filename):
    """Parse hhalign output to get aligned sequences and start indices."""
    with open(filename) as f:
        lines = f.readlines()
    starts = [None, None]
    seqs = ["", ""]
    for line in lines:
        parts = line.split()
        if line.startswith("Q "):
            if starts[0] is None:
                starts[0] = int(parts[2]) - 1
            seqs[0] += parts[3]
        if line.startswith("T "):
            if starts[1] is None:
                starts[1] = int(parts[2]) - 1
            seqs[1] += parts[3]
    return seqs, starts

def align_pair(seq1, seq2):
    """Run hhalign and return aligned index arrays for the two sequences."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fas") as q, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".fas") as t, \
         tempfile.NamedTemporaryFile(suffix=".hhr") as o:
        q.write(f">query\n{seq1}\n"); q.flush()
        t.write(f">target\n{seq2}\n"); t.flush()
        os.system(f"{HHALIGN} -glob -hide_cons -i {q.name} -t {t.name} -o {o.name} 2>/dev/null")
        aligned_seqs, starts = parse_hhalign_output(o.name)

    i, j = starts
    idx_i, idx_j = [], []
    for q_char, t_char in zip(*aligned_seqs):
        if q_char != "-" and t_char != "-":
            idx_i.append(i)
            idx_j.append(j)
        if q_char != "-":
            i += 1
        if t_char != "-":
            j += 1
    return np.array(idx_i), np.array(idx_j)

# --- Main ---
if __name__ == "__main__":
    os.makedirs(ALN_DIR, exist_ok=True)

    with open(f"{OUT_DIR}/seq_dict.json") as f:
        seq_dict = json.load(f)
    df = pd.read_csv(f"{OUT_DIR}/filtered_df.csv")

    aln_info = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aligning pairs"):
        id1, id2 = row["ID1"], row["ID2"]
        try:
            seq1 = seq_dict[id1]["1"]
            seq2 = seq_dict[id2]["1"]
            idx1, idx2 = align_pair(seq1, seq2)
            aln_info[id1] = idx1.tolist()
            aln_info[id2] = idx2.tolist()
        except Exception as e:
            print(f"  SKIP {id1}-{id2}: {e}")

    out_path = f"{ALN_DIR}/cfold_seq_aln.json"
    with open(out_path, "w") as f:
        json.dump(aln_info, f)
    print(f"Saved {len(aln_info)} alignment entries to {out_path}")
