"""

ESM2-650M (esm2_t33_650M_UR50D) is loaded with fair-esm. It has 33 layers x 20
heads = 660 attention features. Feature extraction (need_head_weights ->
symmetrize -> APC -> trim BOS/EOS -> seqsep mask) matches the fig1401 notebook,
and the L1 fit uses the same official-LIBLINEAR pipeline as the ESMC refits.

Stages:
  fit      - our L1 head on all 17 proteins (official LIBLINEAR, -s 6 -c 0.15).
  predict  - our-head contact maps for the eval sequences.
  official - the shipped ESM2 contact head (model.predict_contacts) on the same
             eval sequences, as a baseline.
  all      - fit + predict + official (default).

All machine-specific paths are command-line arguments. Example:
  python fit_predict_esm2_650m_overlay_head.py --stage all \
      --model-ckpt /path/to/model.pt \
      --train-data /path/to/17_protein_data_overlay.pkl \
      --eval-root /path/to/eppic --output-root /path/to/output
Env:  inter_pred  (fair-esm 2.0.0)
"""

import os
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import esm
from tqdm import tqdm
from liblinear.liblinearutil import problem, parameter, train


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_CKPT = None
TRAIN_DATA_PKL = None
TRAIN_TARGET_KEY = "overlay_contact"

EVAL_IDS_JSON = None
EVAL_SEQS_JSON = None
WEIGHTS_PATH = None
PRED_DIR = None
OFFICIAL_PRED_DIR = None

SEQSEP_K = 6
MAX_LEN = 1022         # ESM2 positional limit is 1024 incl. BOS/EOS
DEVICE = "cuda:0"

LIBLINEAR_C = 0.15
LIBLINEAR_EPS = 1e-4
LIBLINEAR_OPTS = f"-s 6 -c {LIBLINEAR_C} -e {LIBLINEAR_EPS}"
LIBLINEAR_BIAS = 1.0


# --------------------------------------------------------------------------- #
# Attention post-processing
# --------------------------------------------------------------------------- #
def symmetrize(x):
    return x + np.swapaxes(x, -1, -2)


def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg /= a12
    return x - avg


def pair_mask(res_mask, L, seqsep_k=SEQSEP_K):
    res_mask = np.asarray(res_mask, dtype=bool)
    sep = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) >= seqsep_k
    return sep & np.outer(res_mask, res_mask)


# --------------------------------------------------------------------------- #
# Model loading + feature extraction (fair-esm)
# --------------------------------------------------------------------------- #
def load_model():
    print(f"Loading ESM2-650M from {MODEL_CKPT} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(MODEL_CKPT)
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    print("Model loaded.")
    return model, alphabet, batch_converter


@torch.no_grad()
def _tokens(seq, batch_converter):
    _, _, toks = batch_converter([("x", seq)])
    return toks.to(DEVICE)


@torch.no_grad()
def extract_attention_features(seq, model, alphabet, batch_converter):
    """Symmetrized + APC attention stack, shape (n_layers*n_heads, L, L)."""
    toks = _tokens(seq, batch_converter)
    out = model(toks, repr_layers=[], need_head_weights=True)
    attn = out["attentions"].squeeze(0)  # (layers, heads, L_tok, L_tok)

    start = int(alphabet.prepend_bos)
    end = attn.size(-1) - int(alphabet.append_eos)
    attn = attn[..., start:end, start:end]

    layers, heads, L, _ = attn.shape
    attn = attn.reshape(layers * heads, L, L).float().cpu().numpy()
    attn = symmetrize(attn)
    attn = apc(attn)
    return attn


@torch.no_grad()
def official_contacts(seq, model, batch_converter):
    """ESM2's shipped contact head -> (L, L) probability map."""
    toks = _tokens(seq, batch_converter)
    contacts = model.predict_contacts(toks)[0]  # (L, L), BOS/EOS already trimmed
    return contacts.float().cpu().numpy()


# --------------------------------------------------------------------------- #
# L1 fit (official LIBLINEAR, identical to the ESMC pipeline)
# --------------------------------------------------------------------------- #
def train_l1_logreg(X, y):
    N, F = X.shape
    data = np.ascontiguousarray(X).ravel()
    indices = np.tile(np.arange(F, dtype=np.int32), N)
    indptr = np.arange(N + 1, dtype=np.int64) * F
    Xcsr = sp.csr_matrix((data, indices, indptr), shape=(N, F), copy=False)

    print(f"Fitting LIBLINEAR ({LIBLINEAR_OPTS}) on {N} pairs x {F} features "
          f"(nnz={Xcsr.nnz:,}) ...")
    prob = problem(y.astype(np.float64).tolist(), Xcsr, bias=LIBLINEAR_BIAS)
    model = train(prob, parameter(LIBLINEAR_OPTS))

    w = np.array([model.get_decfun_coef(j, 0) for j in range(1, F + 1)], dtype=np.float64)
    b = model.get_decfun_bias(0)
    if model.get_labels()[0] == 0:
        w, b = -w, -b

    coef_ = w.reshape(1, -1).astype(np.float32)
    intercept_ = np.array([b], dtype=np.float32)
    print(f"Training finished. nonzero coefs: {int(np.count_nonzero(coef_))}/{F}")
    return coef_, intercept_


# --------------------------------------------------------------------------- #
# Stages
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


def fit(model, alphabet, batch_converter):
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
        attn = extract_attention_features(seq, model, alphabet, batch_converter)
        target = np.asarray(entry[TRAIN_TARGET_KEY], dtype=np.float32)
        mask = pair_mask(entry["mask"], L)
        X_list.append(attn[:, mask].T.astype(np.float32))
        y_list.append(target[mask])
        used_pdb_ids.append(pdb_id)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del X_list, y_list
    print(f"Feature matrix: {X.shape}, labels: {y.shape}, "
          f"positives: {int(y.sum())} ({100 * y.mean():.2f}%)")

    coef_, intercept_ = train_l1_logreg(X, y)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump({"coef_": coef_, "intercept_": intercept_, "seqsep_k": SEQSEP_K,
                     "n_features": X.shape[1], "train_pdb_ids": used_pdb_ids,
                     "target_key": TRAIN_TARGET_KEY, "liblinear_opts": LIBLINEAR_OPTS}, f)
    print(f"Saved regression weights -> {WEIGHTS_PATH}")
    return coef_, intercept_


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


def _eval_seqs():
    with open(EVAL_IDS_JSON) as f:
        pdb_ids = json.load(f)
    with open(EVAL_SEQS_JSON) as f:
        seqs = json.load(f)
    return pdb_ids, seqs


def predict(model, alphabet, batch_converter, coef, intercept):
    pdb_ids, seqs = _eval_seqs()
    os.makedirs(PRED_DIR, exist_ok=True)
    coef = coef.reshape(-1)
    bias = float(intercept.reshape(-1)[0])

    for pdb_id in tqdm(pdb_ids, desc="Predicting (our head)"):
        seq = seqs.get(pdb_id)
        if seq is None or len(seq) > MAX_LEN:
            print(f"[!] skip {pdb_id} (missing or len>{MAX_LEN})")
            continue
        try:
            attn = extract_attention_features(seq, model, alphabet, batch_converter)
            L = len(seq)
            logits = attn.reshape(attn.shape[0], L * L).T @ coef + bias
            probs = (1.0 / (1.0 + np.exp(-logits))).reshape(L, L).astype(np.float32)
            np.save(os.path.join(PRED_DIR, f"{pdb_id}.npy"), probs)
        except torch.cuda.OutOfMemoryError:
            print(f"[!] OOM on {pdb_id}, skipping.")
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error on {pdb_id}: {e}")
    print(f"Our-head predictions -> {PRED_DIR}")


def predict_official(model, batch_converter):
    pdb_ids, seqs = _eval_seqs()
    os.makedirs(OFFICIAL_PRED_DIR, exist_ok=True)
    for pdb_id in tqdm(pdb_ids, desc="Predicting (official head)"):
        seq = seqs.get(pdb_id)
        if seq is None or len(seq) > MAX_LEN:
            print(f"[!] skip {pdb_id} (missing or len>{MAX_LEN})")
            continue
        try:
            probs = official_contacts(seq, model, batch_converter).astype(np.float32)
            np.save(os.path.join(OFFICIAL_PRED_DIR, f"{pdb_id}.npy"), probs)
        except torch.cuda.OutOfMemoryError:
            print(f"[!] OOM on {pdb_id}, skipping.")
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error on {pdb_id}: {e}")
    print(f"Official-head predictions -> {OFFICIAL_PRED_DIR}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["fit", "predict", "official", "all"],
                        default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-ckpt", required=True,
                        help="Local ESM2-650M checkpoint.")
    parser.add_argument("--train-data", required=True,
                        help="Training pickle or exported per-PDB NPZ directory.")
    parser.add_argument("--eval-root", required=True,
                        help="EPPIC directory containing IDs and sequences JSON files.")
    parser.add_argument("--output-root", required=True,
                        help="Directory for fitted weights and prediction maps.")
    args = parser.parse_args()

    global DEVICE, MODEL_CKPT, TRAIN_DATA_PKL
    global EVAL_IDS_JSON, EVAL_SEQS_JSON
    global WEIGHTS_PATH, PRED_DIR, OFFICIAL_PRED_DIR
    DEVICE = args.device
    MODEL_CKPT = os.path.abspath(args.model_ckpt)
    TRAIN_DATA_PKL = os.path.abspath(args.train_data)
    eval_root = os.path.abspath(args.eval_root)
    output_root = os.path.abspath(args.output_root)
    EVAL_IDS_JSON = os.path.join(eval_root, "all_pdb_ids.json")
    EVAL_SEQS_JSON = os.path.join(eval_root, "sequences.json")
    WEIGHTS_PATH = os.path.join(
        output_root, "weights", "esm2_650m_overlay_17_liblinear.pkl"
    )
    PRED_DIR = os.path.join(
        output_root, "predictions", "esm2_650m_overlay_17_liblinear"
    )
    OFFICIAL_PRED_DIR = os.path.join(
        output_root, "predictions", "esm2_650m_official"
    )
    print(f"=== ESM2-650M | device {DEVICE} ===")

    model, alphabet, batch_converter = load_model()

    coef = intercept = None
    if args.stage in ("fit", "all"):
        coef, intercept = fit(model, alphabet, batch_converter)
    elif args.stage in ("predict",):
        coef, intercept = load_weights()

    if args.stage in ("predict", "all"):
        predict(model, alphabet, batch_converter, coef, intercept)

    if args.stage in ("official", "all"):
        predict_official(model, batch_converter)


if __name__ == "__main__":
    main()
