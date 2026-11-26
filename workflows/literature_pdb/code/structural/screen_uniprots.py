"""Screen all UniProts in uniprot_pdb.csv by MSA presence and query length.

For every UniProt in the SIFTS uniprot_pdb.csv:
    1. Check whether the AlphaFold MSA file (AF-<uid>-F1-msa_v6.a3m or .a3m.gz)
       is on disk in `--msa-dir`.
    2. If present, read the first ('query') sequence from the MSA and record
       its length (uppercase residues only, since a3m insertions are lowercase
       and never appear in the query line).
    3. Aggregate stats and write:
         - uniprot_screen_summary.json / .txt
         - uniprot_lengths.json: {uid: length} for every uid with an MSA
         - uniprots_le500_with_msa.json: list of uids with length <= --max-len
           AND an MSA on disk -- this is the input list for run_pairformer.py.

Replaces the older PDB-centric screen_monomers.py / screen_non_monomer.py
flow with a single flat per-UniProt screen.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


UNIPROT_CSV = str(
    Path(__file__).resolve().parents[2] / "data/upstream/uniprot_pdb_source.csv"
)
MSA_DIR = "/path/to/external-data/pdb_pairformer_screen/alphafold_msas"
OUT_DIR = "generated/screen_results"

DEFAULT_MAX_LEN = 500
DEFAULT_WORKERS = 16


def iter_uniprot_ids(csv_path):
    """Yield UniProt accessions from the SIFTS-style uniprot_pdb.csv.

    Mirrors download_msa.iter_uniprot_ids: skips the leading '#' comment line
    and the 'SP_PRIMARY,PDB' header row.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if not first or first.startswith("#") or first == "SP_PRIMARY":
                continue
            yield first


def resolve_msa_path(msa_dir, uid):
    """Return path to the AF MSA for `uid`, preferring .a3m over .a3m.gz."""
    plain = os.path.join(msa_dir, f"AF-{uid}-F1-msa_v6.a3m")
    gz = plain + ".gz"
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    return None


def _open_a3m(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def query_length(msa_path):
    """Return the length of the FIRST sequence (the query) in the a3m file.

    Reads only as much of the file as needed to find the second '>' header.
    a3m insertions (lowercase) only occur on non-query rows, so we strip them
    defensively. Returns None on parse failure.
    """
    try:
        with _open_a3m(msa_path) as f:
            seen_first_header = False
            seq_chunks = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(">"):
                    if seen_first_header:
                        # We've hit the second header -- query is complete.
                        break
                    seen_first_header = True
                    continue
                if seen_first_header:
                    # Strip a3m insertions just in case; for the query they
                    # shouldn't exist, but this is cheap insurance.
                    seq_chunks.append("".join(c for c in line if c.isupper()))
            if not seq_chunks:
                return None
            return sum(len(c) for c in seq_chunks)
    except (OSError, EOFError, gzip.BadGzipFile) as e:
        print(f"  warn: failed to read {msa_path}: {e}", file=sys.stderr)
        return None


def process_uid(uid, msa_dir):
    """Resolve MSA path and (if present) query length for one UniProt.

    Returns dict: {uid, msa_exists, msa_path, length}.
    """
    msa_path = resolve_msa_path(msa_dir, uid)
    if msa_path is None:
        return {"uid": uid, "msa_exists": False, "msa_path": None, "length": None}
    length = query_length(msa_path)
    return {"uid": uid, "msa_exists": True, "msa_path": msa_path, "length": length}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=UNIPROT_CSV)
    ap.add_argument("--msa-dir", default=MSA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N UniProts (for quick tests).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    uids = list(iter_uniprot_ids(args.csv))
    # Deduplicate while preserving order (uniprot_pdb.csv should already be
    # unique, but guard against accidental duplicates).
    seen = set()
    uids_unique = []
    for u in uids:
        if u in seen:
            continue
        seen.add(u)
        uids_unique.append(u)
    if len(uids_unique) != len(uids):
        print(f"Note: dropped {len(uids) - len(uids_unique)} duplicate rows in CSV.")
    uids = uids_unique
    if args.limit:
        uids = uids[: args.limit]
    n_total = len(uids)
    print(f"Loaded {n_total} UniProt IDs from {args.csv}")
    print(f"MSA dir: {args.msa_dir}")
    print(f"Workers: {args.workers}, max-len: {args.max_len}")

    lengths = {}
    no_msa = []
    parse_failed = []
    n_done = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_uid, u, args.msa_dir): u for u in uids}
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if not res["msa_exists"]:
                no_msa.append(res["uid"])
            elif res["length"] is None:
                parse_failed.append(res["uid"])
            else:
                lengths[res["uid"]] = res["length"]

            if n_done % 5000 == 0 or n_done == n_total:
                elapsed = time.monotonic() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6)
                print(f"  [{n_done:>6}/{n_total}] "
                      f"with_msa={len(lengths):>6}  no_msa={len(no_msa):>6}  "
                      f"failed={len(parse_failed):>4}  "
                      f"rate={rate:.0f}/s  ETA={eta:.0f}s",
                      flush=True)

    n_with_msa = len(lengths) + len(parse_failed)  # MSA file present, regardless of parse outcome
    n_with_msa_parsed = len(lengths)
    n_no_msa = len(no_msa)
    n_failed = len(parse_failed)

    le_max = sorted(uid for uid, L in lengths.items() if L <= args.max_len)
    gt_max = sorted(uid for uid, L in lengths.items() if L > args.max_len)

    summary = {
        "n_total_uniprots": n_total,
        "n_with_msa_file": n_with_msa,
        "n_with_msa_parsed": n_with_msa_parsed,
        "n_no_msa": n_no_msa,
        "n_parse_failed": n_failed,
        "max_len": args.max_len,
        "n_le_max_len": len(le_max),
        "n_gt_max_len": len(gt_max),
        "pct_with_msa": 100.0 * n_with_msa / n_total if n_total else 0.0,
        "pct_le_max_len": 100.0 * len(le_max) / n_total if n_total else 0.0,
    }

    with open(os.path.join(args.out_dir, "uniprot_screen_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "uniprot_lengths.json"), "w") as f:
        json.dump(lengths, f, indent=2, sort_keys=True)

    with open(os.path.join(args.out_dir, "uniprots_le500_with_msa.json"), "w") as f:
        json.dump(le_max, f, indent=2)

    summary_lines = [
        "=== UniProt-level MSA screen ===",
        f"Source CSV:                       {args.csv}",
        f"MSA directory:                    {args.msa_dir}",
        f"Length cutoff (--max-len):        {args.max_len}",
        "",
        f"Total UniProts:                   {n_total:>8}",
        f"  with MSA file on disk:          {n_with_msa:>8} "
        f"({summary['pct_with_msa']:.2f}%)",
        f"    ... query length parsed:      {n_with_msa_parsed:>8}",
        f"    ... parse failed:             {n_failed:>8}",
        f"  without MSA file:               {n_no_msa:>8}",
        "",
        f"Of those with parsed length:",
        f"  length <= {args.max_len}:                  {len(le_max):>8} "
        f"({summary['pct_le_max_len']:.2f}% of all UniProts)",
        f"  length >  {args.max_len}:                  {len(gt_max):>8}",
    ]
    summary_txt = os.path.join(args.out_dir, "uniprot_screen_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print()
    for line in summary_lines:
        print(line)

    print()
    print(f"Wrote: {os.path.join(args.out_dir, 'uniprot_screen_summary.json')}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote: {os.path.join(args.out_dir, 'uniprot_lengths.json')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'uniprots_le500_with_msa.json')}")


if __name__ == "__main__":
    main()
