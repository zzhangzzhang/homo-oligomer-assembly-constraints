"""
DockQ and TM-align analysis utilities for homodimer structure predictions.

Evaluates:
  - DockQ: interface quality by comparing predicted dimer vs reference multimer
           (finds best-matching dimer subunit in the reference)
  - TM-align: monomer fold quality by comparing predicted chain A vs reference chain A

Usage:
    from lib.structural_scoring import run_dockq_analysis, run_tmalign_analysis
"""

import os
import re
import subprocess
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

USALIGN = "/usr/local/bin/USalign"


# ============================================================
# DockQ
# ============================================================

def run_dockq(pred_pdb, native_path, model_chains=None, native_chains=None):
    """
    Run DockQ comparing a predicted dimer against a reference structure.
    
    Tries all permutations of native chain pairs to find the best-matching
    dimer subunit (important when native is a tetramer or higher oligomer).
    
    Args:
        pred_pdb: path to predicted PDB (homodimer with chains A, B)
        native_path: path to native CIF/PDB (may have >2 chains)
        model_chains: list of chain IDs in model (default: auto-detect)
        native_chains: list of chain IDs in native (default: auto-detect)
    
    Returns:
        dict with best DockQ metrics, or None on failure
    """
    try:
        model = load_PDB(pred_pdb)
        native = load_PDB(native_path)
    except Exception as e:
        return None

    if model_chains is None:
        model_chains = [c.id for c in model.get_chains()]
    if native_chains is None:
        native_chains = [c.id for c in native.get_chains()]

    if len(model_chains) < 2:
        return None

    best_dockq = -1.0
    best_metrics = None

    # DockQ.run_on_all_native_interfaces expects chain_map: native_chain_id -> model_chain_id
    # (keys are native PDB/CIF chain names; values are prediction chain names). Our earlier
    # model->native dict caused KeyError on model[...] when native used C,D,...
    for nc1, nc2 in itertools.permutations(native_chains, 2):
        chain_map = {nc1: model_chains[0], nc2: model_chains[1]}
        try:
            result_dict, total_dockq = run_on_all_native_interfaces(
                model, native, chain_map=chain_map
            )
            for iface_key, metrics in result_dict.items():
                dq = metrics.get('DockQ', 0)
                if dq > best_dockq:
                    best_dockq = dq
                    best_metrics = {
                        'DockQ': metrics['DockQ'],
                        'Fnat': metrics['fnat'],
                        'iRMSD': metrics['iRMSD'],
                        'LRMSD': metrics['LRMSD'],
                        'F1': metrics['F1'],
                        'chain_map': str(chain_map),
                    }
        except Exception:
            continue

    return best_metrics


# ============================================================
# TM-align (monomer)
# ============================================================

def run_tmalign(pred_pdb, native_path):
    """
    Run USalign to compute TM-score of first chain (monomer fold quality).
    
    Uses -ter 2 to align only the first chain of each structure.
    
    Args:
        pred_pdb: path to predicted PDB
        native_path: path to native CIF/PDB
    
    Returns:
        dict with tm_score (normalized by reference length) and rmsd, or None
    """
    try:
        result = subprocess.run(
            [USALIGN, pred_pdb, native_path, "-ter", "2"],
            capture_output=True, text=True, timeout=120,
        )
        tm_scores = []
        rmsd = None
        for line in result.stdout.split("\n"):
            if line.startswith("TM-score="):
                match = re.search(r"TM-score=\s*([\d.]+)", line)
                if match:
                    tm_scores.append(float(match.group(1)))
            if "RMSD=" in line and "Aligned length=" in line:
                rmsd_match = re.search(r"RMSD=\s*([\d.]+)", line)
                if rmsd_match:
                    rmsd = float(rmsd_match.group(1))

        if len(tm_scores) >= 2:
            return {'tm_score': tm_scores[1], 'rmsd': rmsd}  # normalized by reference
        elif tm_scores:
            return {'tm_score': tm_scores[0], 'rmsd': rmsd}
    except Exception as e:
        pass
    return None


# ============================================================
# Batch analysis over a prediction directory
# ============================================================

def analyze_prediction_dir(prediction_dir, pdb_id, cif_path, 
                           run_dockq_flag=True, run_tm_flag=True,
                           recycle_0_only=True):
    """
    Run DockQ and/or TM-align on prediction PDBs for a given pdb_id.
    
    Args:
        prediction_dir: root output dir (e.g. /path/to/external-data/olig_af_batch_multimer/)
        pdb_id: PDB ID
        cif_path: path to reference CIF
        run_dockq_flag: whether to compute DockQ
        run_tm_flag: whether to compute TM-score
        recycle_0_only: if True, only process recycle 0 results; 
                        if False, process all recycles
    
    Returns:
        list of result dicts (one per PDB file)
    """
    pdb_dir = Path(prediction_dir) / pdb_id
    pattern = re.compile(rf"{pdb_id}_qid(\d+)_qb(True|False)(?:_mask(\d+))?")

    results = []

    for run_dir in pdb_dir.iterdir():
        if not run_dir.is_dir():
            continue

        match = pattern.match(run_dir.name)
        if not match:
            continue

        qid, qb, mask = match.groups()
        maskrate = int(mask) if mask else 0

        recycles_dir = run_dir / "pdb" / "recycles"
        if not recycles_dir.exists():
            continue

        # Include every seed (e.g. seed0–seed4); *_seed0.pdb would drop multi-seed runs.
        glob_pattern = "*_r0_seed*.pdb" if recycle_0_only else "*_seed*.pdb"
        for pdb_file in recycles_dir.glob(glob_pattern):
            fname_pattern = r"model_(\d+)_multimer_v3_r(\d+)_seed(\d+)\.pdb"
            fname_match = re.match(fname_pattern, pdb_file.name)
            if not fname_match:
                continue

            model_num, recycle, seed = map(int, fname_match.groups())
            pred_path = str(pdb_file)

            row = {
                'pdb_id': pdb_id,
                'qid': str(qid),
                'qb': str(qb),
                'maskrate': int(maskrate),
                'seed': int(seed),
                'model': f'model_{model_num}',
                'recycle': int(recycle),
            }

            if run_dockq_flag:
                dockq_result = run_dockq(pred_path, cif_path)
                if dockq_result:
                    row['DockQ'] = dockq_result['DockQ']
                    row['Fnat'] = dockq_result['Fnat']
                    row['iRMSD'] = dockq_result['iRMSD']
                    row['LRMSD'] = dockq_result['LRMSD']
                    row['F1'] = dockq_result['F1']
                else:
                    row['DockQ'] = np.nan
                    row['Fnat'] = np.nan
                    row['iRMSD'] = np.nan
                    row['LRMSD'] = np.nan
                    row['F1'] = np.nan

            if run_tm_flag:
                tm_result = run_tmalign(pred_path, cif_path)
                if tm_result:
                    row['tm_score'] = tm_result['tm_score']
                    row['tm_rmsd'] = tm_result['rmsd']
                else:
                    row['tm_score'] = np.nan
                    row['tm_rmsd'] = np.nan

            results.append(row)

    return results
