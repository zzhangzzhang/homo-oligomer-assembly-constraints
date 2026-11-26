"""Frozen Pairformer execution body launched only by run_pairformer.py.

This file intentionally executes at top level to preserve the historical
scientific implementation. Do not import it as a module.
"""

import os, json, random, gzip, shutil, tempfile
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import login
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d
import pandas as pd
import subprocess

if os.environ.get("HHSUITE_BIN"):
    os.environ["PATH"] += os.pathsep + os.environ["HHSUITE_BIN"]

msa_dir = os.environ.get(
    "PAIRFORMER_MSA_DIR",
    "/path/to/external-data/pdb_pairformer_screen/alphafold_msas",
)
out_dir = os.environ.get(
    "PAIRFORMER_OUT_DIR",
    "/path/to/external-data/pdb_pairformer_screen/pairformer",
)
os.makedirs(out_dir, exist_ok=True)

bundled_target_list = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "upstream",
        "uniprots_le500_with_msa.json",
    )
)
uniprot_list = os.environ.get("PAIRFORMER_TARGET_LIST", bundled_target_list)
with open(uniprot_list) as f:
    uniprot_ids = json.load(f)
print(f"Found {len(uniprot_ids)} UniProts in {uniprot_list}")


# Comma-separated override lets targeted reruns avoid recomputing another qid.
# Default preserves the original full-screen behavior.
qid_cutoffs = [
    int(value)
    for value in os.environ.get("PAIRFORMER_QIDS", "15,35").split(",")
    if value.strip()
]
if not qid_cutoffs:
    raise ValueError("PAIRFORMER_QIDS must contain at least one integer cutoff")

# ============================================================
# Sharding: split remaining work across multiple workers.
# Set PAIRFORMER_SHARD="i/N" (e.g. "0/2", "1/2") to run worker i of N.
# Default "0/1" = single worker (original behavior).
# ============================================================
shard_str = os.environ.get("PAIRFORMER_SHARD", "0/1")
shard_id, n_shards = (int(x) for x in shard_str.split("/"))
assert 0 <= shard_id < n_shards, f"Invalid PAIRFORMER_SHARD={shard_str}"


# ============================================================
# Pre-filter: drop UniProts whose qid_*hhfilter.txt files all already exist.
# A uniprot is kept if at least one qid is still missing its output, so
# partial-work uniprots (e.g. qid=15 done but qid=35 OOM'd) are retried.
# This makes the tqdm bar reflect actual remaining work.
# ============================================================
todo = []
n_already_complete = 0
for uid in uniprot_ids:
    missing_qids = [
        q for q in qid_cutoffs
        if not os.path.exists(os.path.join(out_dir, f"{uid}_qid{q}_hhfilter.txt"))
    ]
    if missing_qids:
        todo.append(uid)
    else:
        n_already_complete += 1
print(
    f"Pre-filter: {n_already_complete} UniProts already complete "
    f"(all qids in {qid_cutoffs} done); {len(todo)} still to run."
)

# Deterministic slicing across shards: sort, then take every Nth starting at shard_id.
# This guarantees disjoint work even if shards start at different times.
if n_shards > 1:
    todo.sort()
    n_total_remaining = len(todo)
    todo = todo[shard_id::n_shards]
    print(
        f"Shard {shard_id}/{n_shards}: assigned {len(todo)} of "
        f"{n_total_remaining} remaining UniProts."
    )

uniprot_ids = todo
random.shuffle(uniprot_ids)

device_str = os.environ.get("PAIRFORMER_DEVICE", "cuda:3")
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = MSAPairformer.from_pretrained(device=device)
model.turn_off_query_biasing()
model.eval()

def _open_a3m(msa_file):
    """Open a .a3m or .a3m.gz file as text."""
    if msa_file.endswith(".gz"):
        return gzip.open(msa_file, "rt")
    return open(msa_file, "r")


def read_a3m(msa_file):
    """Parse .a3m (or .a3m.gz) alignment exactly as in your original code."""
    with _open_a3m(msa_file) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    seqs = []
    for line in lines:
        if line.startswith(">"):
            seqs.append("")
        else:
            seqs[-1] += line.strip()
    return seqs


def resolve_msa_path(msa_dir, uniprot):
    """Return path to the source MSA for a UniProt, preferring .a3m then .a3m.gz.

    Returns None if no MSA file exists.
    """
    plain = os.path.join(msa_dir, f"AF-{uniprot}-F1-msa_v6.a3m")
    gz = plain + ".gz"
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    return None


def materialize_for_hhfilter(src_path):
    """Return (path_for_hhfilter, cleanup_fn) for a source MSA.

    hhfilter does not read gzipped input. If `src_path` is `.a3m.gz`,
    decompress to a NamedTemporaryFile and return its path; the caller MUST
    invoke `cleanup_fn()` when done (it deletes the temp file).
    For plain .a3m, the source path is returned and cleanup_fn is a no-op.
    """
    if not src_path.endswith(".gz"):
        return src_path, (lambda: None)
    tmp = tempfile.NamedTemporaryFile(prefix="msa_", suffix=".a3m", delete=False)
    try:
        with gzip.open(src_path, "rb") as fin:
            shutil.copyfileobj(fin, tmp)
    finally:
        tmp.close()
    return tmp.name, (lambda p=tmp.name: os.path.exists(p) and os.unlink(p))


def predict_contacts_from_tokenized(tokenized):
    msa_onehot = torch.nn.functional.one_hot(
        tokenized, num_classes=len(aa2tok_d)
    ).unsqueeze(0).float().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokenized.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = [x.to(device) for x in [mask, msa_mask, full_mask, pairwise_mask]]
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                msa=msa_onehot.to(torch.bfloat16),
                mask=mask,
                msa_mask=msa_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                return_seq_weights=False,
                return_pairwise_repr_layer_idx=None,
                return_msa_repr_layer_idx=None
            )
    return out["predicted_confind_contacts"][0].cpu().numpy()


# ============================================================
# Reload previous results so relaunching doesn't lose data.
# Per-shard filenames so concurrent workers don't clobber each other.
# ============================================================
if n_shards > 1:
    counts_csv = os.path.join(out_dir, f"msa_sequence_counts.shard{shard_id}of{n_shards}.csv")
    samples_json = os.path.join(out_dir, f"samples_log.shard{shard_id}of{n_shards}.json")
else:
    counts_csv = os.path.join(out_dir, "msa_sequence_counts.csv")
    samples_json = os.path.join(out_dir, "samples_log.json")

if os.path.exists(counts_csv):
    counts = pd.read_csv(counts_csv).to_dict("records")
    print(f"Loaded {len(counts)} existing entries from {counts_csv}")
else:
    counts = []

if os.path.exists(samples_json):
    with open(samples_json) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} existing entries from {samples_json}")
else:
    samples = {}

for uniprot in tqdm(uniprot_ids):

    if uniprot not in samples:
        samples[uniprot] = {}

    # Raw AF MSA path (accepts either .a3m or .a3m.gz)
    msa_path = resolve_msa_path(msa_dir, uniprot)
    if msa_path is None:
        print(f"Missing MSA for {uniprot}: AF-{uniprot}-F1-msa_v6.a3m[.gz]")
        continue

    # quick check (handles gzip transparently)
    seqs = read_a3m(msa_path)
    if len(seqs) < 2:
        print(f"Skipping {uniprot}: fewer than 2 sequences")
        continue

    # ============================================================
    # STEP 1 - Remove very similar sequences with 90% identity
    # ============================================================
    filtered_msa = os.path.join(out_dir, f"{uniprot}_id90.a3m")
    if not os.path.exists(filtered_msa):
        print(f"Running -id 90 filter for {uniprot}")
        # hhfilter cannot read .gz; if needed, decompress to a temp file first.
        hhfilter_input, cleanup = materialize_for_hhfilter(msa_path)
        try:
            subprocess.run([
                "hhfilter",
                "-i", hhfilter_input,
                "-o", filtered_msa,
                "-id", "90",
                "-maxseq", "50000",
            ], check=True)
        finally:
            cleanup()
    else:
        print(f"Using existing ID90-filtered MSA for {uniprot}")

    # ============================================================
    # STEP 2 - Loop over QID thresholds (diversity thresholds)
    # ============================================================
    for qid in qid_cutoffs:

        qid_temp = os.path.join(out_dir, f"{uniprot}_qid{qid}_temp.a3m")
        final_512 = os.path.join(out_dir, f"{uniprot}_qid{qid}_selected512.a3m")
        out_txt = os.path.join(out_dir, f"{uniprot}_qid{qid}_hhfilter.txt")

        if os.path.exists(out_txt):
            print(f"{uniprot} QID={qid} already processed")
            continue

        print(f"=== {uniprot} QID={qid} ===")

        # ============================================================
        # STEP 2A - Apply QID filter to ID90-filtered MSA
        # ============================================================
        subprocess.run([
            "hhfilter",
            "-i", filtered_msa,
            "-o", qid_temp,
            "-qid", str(qid),
            "-maxseq", "50000"
        ], check=True)

        # ============================================================
        # STEP 3 - Select EXACTLY 512 diverse sequences (-diff 512)
        # ============================================================
        subprocess.run([
            "hhfilter",
            "-i", qid_temp,
            "-o", final_512,
            "-diff", "512"
        ], check=True)

        print(f"Saved 512-sequence MSA -> {final_512}")

        # ============================================================
        # STEP 4 - Run MSA-Pairformer (NO internal selection)
        # ============================================================
        try:
            msa_obj = MSA(
                msa_file_path=final_512,
                max_seqs=512,
                diverse_select_method="none"   # disable internal hhfilter
            )

            n_seq = msa_obj.n_diverse_seqs
            tokenized = msa_obj.diverse_tokenized_msa

            contacts = predict_contacts_from_tokenized(tokenized)
            np.savetxt(out_txt, contacts, fmt="%.6f")

            counts.append({
                "uniprot": uniprot,
                "qid": qid,
                "n_seq": n_seq
            })
            samples[uniprot][str(qid)] = {"n_seq": n_seq}

        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {uniprot} QID={qid} - skipping")
            torch.cuda.empty_cache()

    # Persist incrementally so progress survives Ctrl-C / OOM / crashes.
    pd.DataFrame(counts).to_csv(counts_csv, index=False)
    with open(samples_json, "w") as f:
        json.dump(samples, f, indent=2)

# ============================================================
# Save logs (final flush)
# ============================================================
pd.DataFrame(counts).to_csv(counts_csv, index=False)
with open(samples_json, "w") as f:
    json.dump(samples, f, indent=2)

print("\nDone - full pipeline completed:", out_dir)
