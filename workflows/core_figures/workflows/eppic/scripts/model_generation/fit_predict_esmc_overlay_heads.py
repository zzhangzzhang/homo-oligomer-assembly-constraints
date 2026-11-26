"""
Refit the ESM-style attention -> contact logistic-regression head for ESMC.

Works for the biohub HuggingFace ESMC family (ESMC-6B / 600M / 300M), all loaded
with `AutoModelForMaskedLM`. Passing `output_attentions=True` returns a tuple of
per-layer attention maps, each shaped (batch, n_heads, L_tok, L_tok), so every
residue pair is described by n_layers * n_heads attention features:
  ESMC-300M = 30 layers x 15 heads, ESMC-600M = 36 x 18, ESMC-6B = 80 x 40.

This mirrors the ESM2 refit workflow (symmetrize + APC + L1 logistic regression),
fitting on ALL 17 proteins (no subsampling). The L1 fit uses the official
LIBLINEAR library so even the widest model (6B, ~7.75e9 matrix entries) avoids
sklearn's int32 wrapper limit.

Two stages:
  fit     - extract features from the 17 training proteins and train the head.
  predict - apply the trained head to the eval sequences, writing one L x L
            probability map per PDB (consumed by fig05_topx_refit_2.ipynb).

Machine-specific paths are command-line arguments. Example:
  python fit_predict_esmc_overlay_heads.py --model 6b --stage all \
      --model-path /path/to/pinned/huggingface/snapshot \
      --train-data /path/to/17_protein_data_overlay.pkl \
      --eval-root /path/to/eppic --output-root /path/to/output
"""

import os
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from liblinear.liblinearutil import problem, parameter, train
from transformers import AutoModelForMaskedLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# --model key -> short tag for output names.
MODELS = {
    "6b": "esmc6b",
    "600m": "esmc600m",
    "300m": "esmc300m",
}

# Training data: the 17 proteins with experimental contact overlays.
TRAIN_DATA_PKL = None
TRAIN_TARGET_KEY = "overlay_contact"   # label used to fit the head

# Eval sequences to predict on (same convention as the ESM2 notebooks).
EVAL_IDS_JSON = None
EVAL_SEQS_JSON = None

SEQSEP_K = 6            # only score residue pairs with |i - j| >= SEQSEP_K
MAX_LEN = 2046         # ESMC context window is 2048 incl. <cls>/<eos>

# Set by main() based on --model / --device.
MODEL_PATH = None
DEVICE = "cuda:0"
WEIGHTS_PATH = None
PRED_DIR = None

# L1 logistic-regression hyper-parameters.
#
# The ESM2 refit used sklearn LogisticRegression(penalty="l1", C=0.15,
# solver="liblinear"). That is exactly what we want, EXCEPT sklearn's liblinear
# wrapper stores the dense entry count (n_samples * n_features) in a 32-bit int,
# which overflows once it exceeds 2**31 (~2.15e9). All 17 proteins give ~2.42M
# pairs and ESMC-6B has 80*40 = 3200 attention features, so the product is
# ~7.75e9 -- 3.6x over the ceiling (this, not RAM, is what forced subsampling).
#
# We call the official LIBLINEAR library directly instead. Solver "-s 6" is the
# same L1-regularized logistic-regression algorithm sklearn wraps (bit-for-bit
# equivalent results), but its CSR path uses 64-bit indexing and handles the
# full matrix. Options below mirror sklearn's defaults: C=0.15, tol/eps=1e-4,
# and bias=1 (fit_intercept with intercept_scaling=1).
LIBLINEAR_C = 0.15
LIBLINEAR_EPS = 1e-4
LIBLINEAR_OPTS = f"-s 6 -c {LIBLINEAR_C} -e {LIBLINEAR_EPS}"
LIBLINEAR_BIAS = 1.0


# --------------------------------------------------------------------------- #
# Attention post-processing (matches esm.modules.ContactPredictionHead)
# --------------------------------------------------------------------------- #
def symmetrize(x):
    """Make the last two dims symmetric: x + x^T."""
    return x + np.swapaxes(x, -1, -2)


def apc(x):
    """Average product correction over the last two dims."""
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg /= a12
    return x - avg


# --------------------------------------------------------------------------- #
# Model loading + feature extraction
# --------------------------------------------------------------------------- #
def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        attn_implementation="eager",  # required for output_attentions
    )
    model = model.to(DEVICE).eval()
    print("Model loaded.")
    return model, tokenizer


@torch.no_grad()
def extract_attention_features(seq, model, tokenizer):
    """
    Return the symmetrized + APC-corrected attention stack for one sequence.

    Output: float32 array of shape (n_layers * n_heads, L, L) where L == len(seq)
    (the <cls>/<eos> positions have been trimmed).
    """
    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    out = model(input_ids, output_attentions=True)

    # tuple(n_layers) of (1, n_heads, L_tok, L_tok) -> (n_layers, n_heads, L_tok, L_tok)
    attn = torch.stack([a.squeeze(0) for a in out.attentions], dim=0)

    # Trim <cls> (first) and <eos> (last) tokens.
    attn = attn[..., 1:-1, 1:-1]

    layers, heads, L, _ = attn.shape
    attn = attn.reshape(layers * heads, L, L).float().cpu().numpy()

    attn = symmetrize(attn)
    attn = apc(attn)
    return attn  # (layers*heads, L, L)


def pair_mask(res_mask, L, seqsep_k=SEQSEP_K):
    """Valid residue pairs: both residues present AND |i - j| >= seqsep_k."""
    res_mask = np.asarray(res_mask, dtype=bool)
    sep = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) >= seqsep_k
    return sep & np.outer(res_mask, res_mask)


def train_l1_logreg(X, y):
    """
    L1 logistic regression via the official LIBLINEAR (solver -s 6).

    Equivalent to sklearn LogisticRegression(penalty="l1", C=0.15,
    solver="liblinear") but without the int32 entry-count limit.

    Returns (coef_, intercept_) with class label 1 as the positive class,
    shaped (1, n_features) and (1,) to match the sklearn convention.
    """
    N, F = X.shape

    # Build a fully-dense CSR directly (every entry is a stored "nonzero").
    # Constructing indptr/indices explicitly avoids scipy's densify scan and
    # keeps everything in 64-bit where it matters.
    data = np.ascontiguousarray(X).ravel()                 # view, float32
    indices = np.tile(np.arange(F, dtype=np.int32), N)     # column of each entry
    indptr = np.arange(N + 1, dtype=np.int64) * F
    Xcsr = sp.csr_matrix((data, indices, indptr), shape=(N, F), copy=False)

    print(f"Fitting LIBLINEAR ({LIBLINEAR_OPTS}) on {N} pairs x {F} features "
          f"(nnz={Xcsr.nnz:,}) ...")
    prob = problem(y.astype(np.float64).tolist(), Xcsr, bias=LIBLINEAR_BIAS)
    model = train(prob, parameter(LIBLINEAR_OPTS))

    w = np.array([model.get_decfun_coef(j, 0) for j in range(1, F + 1)],
                 dtype=np.float64)
    b = model.get_decfun_bias(0)

    # LIBLINEAR's decision function targets whichever label appears first in y.
    # Flip so that positive weights push towards "contact" (label 1).
    if model.get_labels()[0] == 0:
        w, b = -w, -b

    coef_ = w.reshape(1, -1).astype(np.float32)
    intercept_ = np.array([b], dtype=np.float32)
    print(f"Training finished. nonzero coefs: {int(np.count_nonzero(coef_))}/{F}")
    return coef_, intercept_


# --------------------------------------------------------------------------- #
# Stage 1: fit
# --------------------------------------------------------------------------- #
def load_training_data(path):
    """Load either the original trusted pickle or exported per-PDB NPZ files."""
    path = Path(path)
    if path.is_dir():
        data = {}
        for archive_path in sorted(path.glob("*.npz")):
            with np.load(archive_path, allow_pickle=False) as archive:
                data[archive_path.stem] = {
                    "seq": str(archive["sequence"].item()),
                    "mask": archive["residue_mask"],
                    "overlay_contact": archive["overlay_contact"],
                }
        if not data:
            raise FileNotFoundError(f"No training NPZ files under {path}")
        return data
    with path.open("rb") as handle:
        return pickle.load(handle)


def fit(model, tokenizer):
    data = load_training_data(TRAIN_DATA_PKL)

    pdb_ids = list(data.keys())
    print(f"Fitting on all {len(pdb_ids)} proteins: {pdb_ids}")

    X_list, y_list, used_pdb_ids = [], [], []
    for pdb_id in tqdm(pdb_ids, desc="Extracting attention features"):
        entry = data[pdb_id]
        seq = entry["seq"]
        L = len(seq)
        if L > MAX_LEN:
            print(f"[!] {pdb_id} length {L} > {MAX_LEN}, skipping.")
            continue

        attn = extract_attention_features(seq, model, tokenizer)  # (C, L, L)
        target = np.asarray(entry[TRAIN_TARGET_KEY], dtype=np.float32)  # (L, L)
        mask = pair_mask(entry["mask"], L)                              # (L, L)

        X_list.append(attn[:, mask].T.astype(np.float32))  # (n_pairs, C)
        y_list.append(target[mask])                         # (n_pairs,)
        used_pdb_ids.append(pdb_id)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del X_list, y_list
    print(f"Feature matrix: {X.shape}, labels: {y.shape}, "
          f"positives: {int(y.sum())} ({100 * y.mean():.2f}%)")

    coef_, intercept_ = train_l1_logreg(X, y)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(
            {
                "coef_": coef_,          # (1, F)
                "intercept_": intercept_,  # (1,)
                "seqsep_k": SEQSEP_K,
                "n_features": X.shape[1],
                "train_pdb_ids": used_pdb_ids,
                "target_key": TRAIN_TARGET_KEY,
                "liblinear_opts": LIBLINEAR_OPTS,
            },
            f,
        )
    print(f"Saved regression weights -> {WEIGHTS_PATH}")
    return coef_, intercept_


# --------------------------------------------------------------------------- #
# Stage 2: predict
# --------------------------------------------------------------------------- #
def load_weights():
    path = Path(WEIGHTS_PATH)
    if path.is_file():
        with path.open("rb") as handle:
            weights = pickle.load(handle)
        return (
            weights["coef_"].astype(np.float32),
            weights["intercept_"].astype(np.float32),
        )
    safe_path = path.with_suffix(".npz")
    with np.load(safe_path, allow_pickle=False) as weights:
        return (
            weights["coef"].astype(np.float32),
            weights["intercept"].astype(np.float32),
        )


def predict(model, tokenizer, coef, intercept):
    with open(EVAL_IDS_JSON) as f:
        pdb_ids = json.load(f)
    with open(EVAL_SEQS_JSON) as f:
        seqs = json.load(f)

    os.makedirs(PRED_DIR, exist_ok=True)
    coef = coef.reshape(-1)  # (C,)
    bias = float(intercept.reshape(-1)[0])

    for pdb_id in tqdm(pdb_ids, desc="Predicting contact maps"):
        if pdb_id not in seqs:
            print(f"[!] Missing sequence for {pdb_id}, skipping.")
            continue
        seq = seqs[pdb_id]
        L = len(seq)
        if L > MAX_LEN:
            print(f"[!] {pdb_id} length {L} > {MAX_LEN}, skipping.")
            continue

        try:
            attn = extract_attention_features(seq, model, tokenizer)  # (C, L, L)
            feats = attn.reshape(attn.shape[0], L * L).T               # (L*L, C)
            logits = feats @ coef + bias
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = probs.reshape(L, L).astype(np.float32)
            np.save(os.path.join(PRED_DIR, f"{pdb_id}.npy"), probs)
        except torch.cuda.OutOfMemoryError:
            print(f"[!] OOM on {pdb_id} (L={L}), skipping.")
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error on {pdb_id}: {e}")

    print(f"All predictions written to {PRED_DIR}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", choices=list(MODELS), default="6b",
        help="which ESMC variant to fit/predict (default: 6b).",
    )
    parser.add_argument(
        "--stage", choices=["fit", "predict", "all"], default="all",
        help="fit the head, predict eval maps, or both (default: all).",
    )
    parser.add_argument("--device", default="cuda:0", help="torch device.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Exact local Hugging Face snapshot; pin this path in the data manifest.",
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Training pickle or exported per-PDB NPZ directory.",
    )
    parser.add_argument(
        "--eval-root",
        required=True,
        help="EPPIC directory containing IDs and sequences JSON files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for fitted weights and prediction maps.",
    )
    args = parser.parse_args()

    global MODEL_PATH, DEVICE, WEIGHTS_PATH, PRED_DIR
    global TRAIN_DATA_PKL, EVAL_IDS_JSON, EVAL_SEQS_JSON
    tag = MODELS[args.model]
    MODEL_PATH = os.path.abspath(args.model_path)
    DEVICE = args.device
    TRAIN_DATA_PKL = os.path.abspath(args.train_data)
    eval_root = os.path.abspath(args.eval_root)
    output_root = os.path.abspath(args.output_root)
    EVAL_IDS_JSON = os.path.join(eval_root, "all_pdb_ids.json")
    EVAL_SEQS_JSON = os.path.join(eval_root, "sequences.json")
    WEIGHTS_PATH = os.path.join(
        output_root, "weights", f"{tag}_overlay_17_liblinear.pkl"
    )
    PRED_DIR = os.path.join(
        output_root, "predictions", f"{tag}_overlay_17_liblinear"
    )
    print(f"=== ESMC {args.model} | device {DEVICE} | tag {tag} ===")

    model, tokenizer = load_model()

    if args.stage in ("fit", "all"):
        coef, intercept = fit(model, tokenizer)
    else:
        coef, intercept = load_weights()

    if args.stage in ("predict", "all"):
        predict(model, tokenizer, coef, intercept)


if __name__ == "__main__":
    main()
