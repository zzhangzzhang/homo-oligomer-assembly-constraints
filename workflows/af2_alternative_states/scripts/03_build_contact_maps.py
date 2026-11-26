#!/usr/bin/env python
"""03: Parse confind output into NxN contact matrices, handling residue ID mapping."""

import json
import numpy as np
import gemmi
from tqdm import tqdm

# --- Config ---
OUT_DIR = "/path/to/external-data/cfold_clean"
CONFIND_DIR = f"{OUT_DIR}/confind_contacts"
CIF_DIR = "/path/to/external-data/pdb_bioassembly_all/"

# --- Functions ---
def get_seq_length(seq_dict, pdb_id):
    """Get length of chain 1 sequence."""
    return len(seq_dict[pdb_id]["1"])

def read_residue_ids(cif_path, property_name):
    """Read unique residue IDs from a CIF file (0-indexed)."""
    doc = gemmi.cif.read(cif_path)
    block = doc.sole_block()
    ids = [x for x in block.find_loop(property_name)]
    unique = list(dict.fromkeys(ids))  # preserve order, deduplicate
    return [int(x) - 1 for x in unique if x.lstrip("-").isdigit()]

def parse_confind(pdb_id, seq_len, label_ids, auth_ids):
    """Parse confind .cf.txt into an NxN contact matrix.
    
    Uses auth->label residue ID mapping when auth_ids differ from label_ids.
    """
    cf_path = f"{CONFIND_DIR}/{pdb_id}.cf.txt"
    mat = np.zeros((seq_len, seq_len), dtype=np.int8)
    
    # Build auth->label mapping if needed
    use_mapping = (auth_ids is not None and label_ids is not None)
    if use_mapping:
        # auth_ids and label_ids are both 0-indexed
        auth_to_label = {}
        for a, l in zip(auth_ids, label_ids):
            auth_to_label[a] = l

    with open(cf_path) as f:
        for line in f:
            if not line.startswith("contact"):
                continue
            parts = line.split()
            chain1, res1 = parts[1].split(",")
            chain2, res2 = parts[2].split(",")
            value = float(parts[3])
            if chain1 != "A" or chain2 != "A" or value <= 0.01:
                continue
            
            r1, r2 = int(res1), int(res2)
            
            if use_mapping:
                # Convert auth residue IDs (1-based in file) to 0-indexed label IDs
                r1_key = r1 - 1  # file is 1-based, our maps are 0-based
                r2_key = r2 - 1
                idx1 = auth_to_label.get(r1_key)
                idx2 = auth_to_label.get(r2_key)
                if idx1 is None or idx2 is None:
                    continue
            else:
                idx1 = r1 - 1
                idx2 = r2 - 1
            
            if 0 <= idx1 < seq_len and 0 <= idx2 < seq_len:
                mat[idx1, idx2] = 1
                mat[idx2, idx1] = 1
    
    return mat

# --- Main ---
if __name__ == "__main__":
    with open(f"{OUT_DIR}/seq_dict.json") as f:
        seq_dict = json.load(f)

    pdb_ids = list(seq_dict.keys())
    done, skip = 0, 0

    for pdb_id in tqdm(pdb_ids, desc="Building contact maps"):
        out_path = f"{CONFIND_DIR}/{pdb_id}_processed.npy"
        cf_path = f"{CONFIND_DIR}/{pdb_id}.cf.txt"
        
        if not __import__("os").path.exists(cf_path):
            skip += 1
            continue
        
        try:
            seq_len = get_seq_length(seq_dict, pdb_id)
            cif_path = f"{CIF_DIR}{pdb_id.lower()}-assembly1.cif.gz"
            label_ids = read_residue_ids(cif_path, "_atom_site.label_seq_id")
            auth_ids = read_residue_ids(cif_path, "_atom_site.auth_seq_id")
            
            mat = parse_confind(pdb_id, seq_len, label_ids, auth_ids)
            np.save(out_path, mat)
            done += 1
        except Exception as e:
            print(f"  SKIP {pdb_id}: {e}")
            skip += 1

    print(f"Done: {done} contact maps, {skip} skipped")
