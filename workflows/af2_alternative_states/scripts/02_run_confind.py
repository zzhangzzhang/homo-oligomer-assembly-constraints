#!/usr/bin/env python
"""02: Download PDB files and run confind contact analysis in parallel."""

import os
import json
import subprocess
import requests
from multiprocessing import Pool
from tqdm import tqdm

# --- Config ---
OUT_DIR = "/path/to/external-data/cfold_clean"
PDB_DIR = f"{OUT_DIR}/exp_pdb"
CONFIND_OUT = f"{OUT_DIR}/confind_contacts"
CONFIND_BIN = "/path/to/external-home/sam/homooligomer/confind-msl-bin/confind"
ROTLIB_DIR = "/path/to/external-home/sam/homooligomer/confind-msl-bin/rotlibs/DB-2010/"
N_WORKERS = 32

# --- Functions ---
def download_pdb(pdb_id):
    """Download PDB file from RCSB if not already present."""
    pdb_file = f"{PDB_DIR}/{pdb_id}.pdb"
    if os.path.exists(pdb_file):
        return True
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(pdb_file, "w") as f:
            f.write(resp.text)
        return True
    print(f"  FAIL download {pdb_id}: HTTP {resp.status_code}")
    return False

def run_confind(pdb_id):
    """Run confind on a PDB file."""
    out_file = f"{CONFIND_OUT}/{pdb_id}.cf.txt"
    if os.path.exists(out_file):
        return
    if not download_pdb(pdb_id):
        return
    try:
        subprocess.run([
            CONFIND_BIN,
            "--p", f"{PDB_DIR}/{pdb_id}.pdb",
            "--o", out_file,
            "--rLib", ROTLIB_DIR,
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL confind {pdb_id}: {e}")

# --- Main ---
if __name__ == "__main__":
    os.makedirs(PDB_DIR, exist_ok=True)
    os.makedirs(CONFIND_OUT, exist_ok=True)

    with open(f"{OUT_DIR}/seq_dict.json") as f:
        seq_dict = json.load(f)
    pdb_ids = list(seq_dict.keys())
    print(f"Running confind on {len(pdb_ids)} PDB IDs with {N_WORKERS} workers")

    with Pool(processes=N_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(run_confind, pdb_ids), total=len(pdb_ids)))
    print("Done")
