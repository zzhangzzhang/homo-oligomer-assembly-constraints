#!/usr/bin/env python
"""01: Read CSV pairs, fix PDB IDs, extract sequences from CIF files."""

import json
import pandas as pd
import gemmi
from tqdm import tqdm

# --- Config ---
CSV_PATH = "/path/to/external-home/sam/homooligomer/cfold_train_test_monomer.csv"
CIF_DIR = "/path/to/external-data/pdb_bioassembly_all/"
OUT_DIR = "/path/to/external-data/cfold_clean"

# --- Functions ---
def fix_pdb_ids(df):
    """Fix PDB IDs mangled by Excel scientific notation (e.g. 5.00E+84 -> 5E84)."""
    for col in ["ID1", "ID2", "struct_clust1", "struct_clust2"]:
        df[col] = df[col].astype(str).replace(r"\.00E\+", "E", regex=True)
    return df

def extract_sequences(cif_path):
    """Extract canonical sequences per entity from a CIF file."""
    doc = gemmi.cif.read(cif_path)
    block = doc.sole_block()
    seqs = {}
    entity_poly = block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"])
    if entity_poly:
        for row in entity_poly:
            seqs[row[0]] = row[1].replace("\n", "").replace(" ", "").replace(";", "")
    return seqs

# --- Main ---
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df = fix_pdb_ids(df)

    # Non-redundant PDB IDs from both columns
    pdb_ids = pd.concat([df["ID1"], df["ID2"]]).unique().tolist()
    print(f"{len(pdb_ids)} unique PDB IDs from {len(df)} pairs")

    # Extract sequences
    seq_dict = {}
    for pdb_id in tqdm(pdb_ids, desc="Extracting sequences"):
        cif_path = f"{CIF_DIR}{pdb_id.lower()}-assembly1.cif.gz"
        try:
            seq_dict[pdb_id] = extract_sequences(cif_path)
        except Exception as e:
            print(f"  SKIP {pdb_id}: {e}")

    # Filter pairs to those with sequences for both IDs
    has_seq = set(seq_dict.keys())
    filtered = df[df["ID1"].isin(has_seq) & df["ID2"].isin(has_seq)].reset_index(drop=True)
    print(f"{len(filtered)} pairs after filtering (both IDs have sequences)")

    # Save
    with open(f"{OUT_DIR}/seq_dict.json", "w") as f:
        json.dump(seq_dict, f, indent=2)
    filtered.to_csv(f"{OUT_DIR}/filtered_df.csv", index=False)
    print(f"Saved seq_dict.json ({len(seq_dict)} entries) and filtered_df.csv")
