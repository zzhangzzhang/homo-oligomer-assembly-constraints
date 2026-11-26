"""Build the per-PDB manifest for the full screen.

Filters applied:
    * pdb is in monomers.json (-> monomer manifest) or
      homooligomers.csv (-> homo manifest).
    * pdb has at least one chain that maps (via SIFTS CSV) to a UniProt
      with a pairformer matrix at qid=35 on disk.
    * the pairformer matrix is <= max_L (default 500) x max_L.
    * the AU CIF exists at /path/to/external-data/mmcif_experimental/{xx}/{pdb}.cif.gz.
    * (homo only) the bioassembly CIF exists at
      /path/to/external-data/pdb_bioassembly_all/{pdb}-assembly1.cif.gz.

Outputs:
    OUT_DIR/manifest_monomer.csv
    OUT_DIR/manifest_homo.csv

Each row: pdb, uniprot, L_up, n_chains_in_au_sifts,
          n_chains_mapping_uniprot, pf_path, au_cif_path,
          bioasm_cif_path (homo only).

If a PDB maps to multiple UniProts (rare for our screen), we keep the
UniProt with the largest n_chains_mapping_uniprot, breaking ties by
SIFTS sequence-coverage (RES_END - RES_BEG).

We deliberately do NOT call gemmi here -- only file existence checks
and a single header-line peek on each pairformer file. The full scan
across ~50k PDBs should take a couple of minutes.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


DEFAULT_MONOMERS = "/path/to/external-home/sam/homooligomer/processed_data2/monomers.json"
DEFAULT_HOMO = "/path/to/external-home/sam/homooligomer/processed_data2/homooligomers.csv"
DEFAULT_SIFTS = "/path/to/external-data/pdb_pairformer_screen/pdb_chain_uniprot.csv"
DEFAULT_PAIRFORMER_DIR = "/path/to/external-data/pdb_pairformer_screen/pairformer"
DEFAULT_AU_DIR = "/path/to/external-data/mmcif_experimental"
DEFAULT_BIOASM_DIR = "/path/to/external-data/pdb_bioassembly_all"
DEFAULT_OUT_DIR = "/path/to/external-data/pdb_pairformer_screen/full_screen"
PF_RE = re.compile(r"^(?P<up>.+?)_qid(?P<qid>\d+)_hhfilter\.txt$")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_monomers(path):
    with open(path) as f:
        return set(p.strip().upper() for p in json.load(f))


def load_homo(path):
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = row["PDB ID"].strip().upper()
            try:
                copy = int(row["Copy"])
            except (KeyError, ValueError):
                copy = None
            out[pdb] = copy
    return out


def index_pairformer(pairformer_dir, qid=35):
    """{uniprot: pairformer_path} for the chosen qid."""
    out = {}
    pat = os.path.join(pairformer_dir, f"*_qid{qid}_hhfilter.txt")
    for p in glob.glob(pat):
        m = PF_RE.match(os.path.basename(p))
        if m and int(m.group("qid")) == qid:
            out[m.group("up")] = p
    return out


def pairformer_matrix_size(path):
    """Peek at the first line of the pairformer text file to get the
    matrix width. The file is `np.savetxt`-formatted, one row per line,
    space-separated. We count the floats in line 1.
    """
    try:
        with open(path) as f:
            first = f.readline()
        return len(first.split())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SIFTS CSV scan (segment level)
# ---------------------------------------------------------------------------

def stream_sifts(csv_path):
    with open(csv_path) as f:
        for line in f:
            if not line or line.startswith("#") or line.startswith("PDB,"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) < 9:
                continue
            pdb = parts[0].strip().lower()
            chain = parts[1].strip()
            sp = parts[2].strip()
            try:
                res_beg = int(parts[3])
                res_end = int(parts[4])
            except ValueError:
                continue
            yield pdb, chain, sp, res_beg, res_end


def build_pdb_uniprot_map(sifts_csv, wanted_pdbs):
    """{pdb_upper: {uniprot: dict(n_chains, max_coverage)}}"""
    out = defaultdict(lambda: defaultdict(lambda: {"chains": set(), "max_coverage": 0}))
    for pdb, chain, sp, res_beg, res_end in stream_sifts(sifts_csv):
        up_pdb = pdb.upper()
        if up_pdb not in wanted_pdbs:
            continue
        cov = res_end - res_beg + 1
        rec = out[up_pdb][sp]
        rec["chains"].add(chain)
        if cov > rec["max_coverage"]:
            rec["max_coverage"] = cov
    return out


# ---------------------------------------------------------------------------
# AU CIF path
# ---------------------------------------------------------------------------

def au_cif_path(pdb, au_dir):
    pdb_lower = pdb.lower()
    sub = pdb_lower[1:3]
    return os.path.join(au_dir, sub, f"{pdb_lower}.cif.gz")


def bioasm_cif_path(pdb, bioasm_dir):
    return os.path.join(bioasm_dir, f"{pdb.lower()}-assembly1.cif.gz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monomers", default=DEFAULT_MONOMERS)
    parser.add_argument("--homo", default=DEFAULT_HOMO)
    parser.add_argument("--sifts", default=DEFAULT_SIFTS)
    parser.add_argument("--pairformer-dir", default=DEFAULT_PAIRFORMER_DIR)
    parser.add_argument("--au-dir", default=DEFAULT_AU_DIR)
    parser.add_argument("--bioasm-dir", default=DEFAULT_BIOASM_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--qid", type=int, default=35)
    parser.add_argument("--max-L", type=int, default=500,
                        help="UniProt length cutoff (= pairformer matrix size).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading PDB lists ...")
    monomers = load_monomers(args.monomers)
    homo = load_homo(args.homo)
    homo_set = set(homo.keys())
    wanted = monomers | homo_set
    print(f"  monomers          = {len(monomers):>7}")
    print(f"  homooligomers     = {len(homo_set):>7}")
    print(f"  union             = {len(wanted):>7}")

    print(f"Indexing pairformer files at qid={args.qid} ...")
    pf_index = index_pairformer(args.pairformer_dir, qid=args.qid)
    print(f"  pf UniProts        = {len(pf_index):>7}")

    print("Filtering pairformer matrices by L <= max_L ...")
    pf_kept = {}
    sizes_counter = defaultdict(int)
    for up, p in pf_index.items():
        L = pairformer_matrix_size(p)
        if L is None:
            continue
        sizes_counter["any"] += 1
        if L <= args.max_L:
            pf_kept[up] = (p, L)
            sizes_counter["kept"] += 1
    print(f"  pf UniProts with L <= {args.max_L} = {sizes_counter['kept']} / {sizes_counter['any']}")

    print("Scanning SIFTS CSV for PDB -> UniProt mappings (wanted PDBs only) ...")
    pdb_to_up = build_pdb_uniprot_map(args.sifts, wanted)
    print(f"  PDBs with any SIFTS row = {len(pdb_to_up)}")

    def choose_uniprot(pdb):
        cands = pdb_to_up.get(pdb, {})
        # Keep only UniProts with a pairformer at qid that passed the L filter
        eligible = [(up, rec) for up, rec in cands.items() if up in pf_kept]
        if not eligible:
            return None
        # Pick UniProt with most mapped chains, tie-break by max coverage
        eligible.sort(key=lambda kv: (-len(kv[1]["chains"]), -kv[1]["max_coverage"], kv[0]))
        return eligible[0]

    def emit_manifest(pdb_iter, dst_path, include_bioasm):
        n_total = 0
        n_kept = 0
        n_dropped_no_cif = 0
        n_dropped_no_bio = 0
        n_dropped_no_pf = 0
        with open(dst_path, "w", newline="") as f:
            cols = ["pdb", "uniprot", "L_up", "n_chains_in_au_sifts",
                    "n_chains_mapping_uniprot", "pf_path", "au_cif_path"]
            if include_bioasm:
                cols.append("bioasm_cif_path")
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for pdb in pdb_iter:
                n_total += 1
                chosen = choose_uniprot(pdb)
                if chosen is None:
                    n_dropped_no_pf += 1
                    continue
                up, rec = chosen
                pf_path, L = pf_kept[up]
                au_path = au_cif_path(pdb, args.au_dir)
                if not os.path.exists(au_path):
                    n_dropped_no_cif += 1
                    continue
                row = {
                    "pdb": pdb,
                    "uniprot": up,
                    "L_up": L,
                    "n_chains_in_au_sifts": len({c for rec2 in pdb_to_up[pdb].values() for c in rec2["chains"]}),
                    "n_chains_mapping_uniprot": len(rec["chains"]),
                    "pf_path": pf_path,
                    "au_cif_path": au_path,
                }
                if include_bioasm:
                    bp = bioasm_cif_path(pdb, args.bioasm_dir)
                    if not os.path.exists(bp):
                        n_dropped_no_bio += 1
                        continue
                    row["bioasm_cif_path"] = bp
                w.writerow(row)
                n_kept += 1
        print(f"  wrote {dst_path}")
        print(f"    total considered : {n_total}")
        print(f"    kept             : {n_kept}")
        print(f"    dropped (no pf)  : {n_dropped_no_pf}")
        print(f"    dropped (no cif) : {n_dropped_no_cif}")
        if include_bioasm:
            print(f"    dropped (no bio) : {n_dropped_no_bio}")
        return n_kept

    print("\n=== monomer manifest ===")
    emit_manifest(sorted(monomers),
                  os.path.join(args.out_dir, "manifest_monomer.csv"),
                  include_bioasm=False)

    print("\n=== homo manifest ===")
    emit_manifest(sorted(homo_set),
                  os.path.join(args.out_dir, "manifest_homo.csv"),
                  include_bioasm=True)


if __name__ == "__main__":
    main()
