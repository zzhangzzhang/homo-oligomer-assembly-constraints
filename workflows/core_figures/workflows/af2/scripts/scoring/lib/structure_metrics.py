"""
Structure comparison - extract contact maps and compute metrics.

Usage:
    from structure_analysis import analyze_structures
    
    df = analyze_structures(
        prediction_dir='/path/to/external-data/olig_af_batch',
        pdb_id='3mkm',
        cif_path='/path/to/3mkm.cif'
    )
"""

import numpy as np
import pandas as pd
import gemmi
from scipy.spatial.distance import cdist
from pathlib import Path
import re


def extract_contacts_from_cif(cif_path, threshold=8.0):
    """Extract contact maps from experimental CIF."""
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()
    
    # Get sequence length
    poly_table = block.find_mmcif_category("_entity_poly.")
    seq_col = poly_table.find_column("pdbx_seq_one_letter_code_can")
    N = len("".join(seq_col[0].split()))
    
    # Get atoms
    atom_table = block.find_mmcif_category("_atom_site.")
    chains = np.array(atom_table.find_column("label_asym_id"))
    res_ids = np.array(atom_table.find_column("label_seq_id"))
    atom_names = np.array(atom_table.find_column("label_atom_id"))
    xs = np.array(atom_table.find_column("Cartn_x"), dtype=float)
    ys = np.array(atom_table.find_column("Cartn_y"), dtype=float)
    zs = np.array(atom_table.find_column("Cartn_z"), dtype=float)
    
    # Filter C-alpha
    ca_mask = (atom_names == "CA") & (res_ids != ".")
    ca_coords = np.stack([xs[ca_mask], ys[ca_mask], zs[ca_mask]], axis=1)
    ca_chains = chains[ca_mask]
    ca_seq_pos = res_ids[ca_mask].astype(int) - 1
    
    # Compute contacts
    dist_matrix = cdist(ca_coords, ca_coords, 'euclidean')
    contact_pairs = np.where(dist_matrix < threshold)
    
    intra = np.zeros((N, N), dtype=int)
    inter = np.zeros((N, N), dtype=int)
    
    row_idx, col_idx = contact_pairs
    pos_i = ca_seq_pos[row_idx]
    pos_j = ca_seq_pos[col_idx]
    same_chain = ca_chains[row_idx] == ca_chains[col_idx]
    
    intra[pos_i[same_chain], pos_j[same_chain]] = 1
    inter[pos_i[~same_chain], pos_j[~same_chain]] = 1
    
    return intra, inter


def extract_contacts_from_pdb(pdb_path, seq_length, threshold=8.0):
    """Extract contact maps from predicted PDB."""
    structure = gemmi.read_structure(pdb_path)
    
    ca_coords = []
    ca_chains = []
    ca_seq_pos = []
    
    for model in structure:
        for chain in model:
            for res in chain:
                res_idx = res.seqid.num - 1
                for atom in res:
                    if atom.name == "CA":
                        ca_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        ca_chains.append(chain.name)
                        ca_seq_pos.append(res_idx)
                        break
    
    if not ca_coords:
        return np.zeros((seq_length, seq_length), dtype=int), \
               np.zeros((seq_length, seq_length), dtype=int)
    
    coords_array = np.array(ca_coords)
    intra = np.zeros((seq_length, seq_length), dtype=int)
    inter = np.zeros((seq_length, seq_length), dtype=int)
    
    dist_matrix = cdist(coords_array, coords_array, 'euclidean')
    contacts = dist_matrix < threshold
    
    row_idx, col_idx = np.where(contacts)
    for i, j in zip(row_idx, col_idx):
        pos_i, pos_j = ca_seq_pos[i], ca_seq_pos[j]
        if pos_i >= seq_length or pos_j >= seq_length:
            continue
        if ca_chains[i] == ca_chains[j]:
            intra[pos_i, pos_j] = 1
        else:
            inter[pos_i, pos_j] = 1
    
    return intra, inter


def compute_metrics(pred, ref):
    """Compute F1, precision, recall."""
    # Upper triangle only, skip diagonal
    pred_subset = pred[np.triu_indices(pred.shape[0], k=6)]
    ref_subset = ref[np.triu_indices(ref.shape[0], k=6)]
    
    pred_bool = pred_subset.astype(bool)
    ref_bool = ref_subset.astype(bool)
    
    tp = np.sum(pred_bool & ref_bool)
    fp = np.sum(pred_bool & ~ref_bool)
    fn = np.sum(~pred_bool & ref_bool)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) 
          if (precision + recall) > 0 else 0.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def analyze_structures(prediction_dir, pdb_id, cif_path):
    """
    Loop over all structures and return DataFrame.
    
    Args:
        prediction_dir: Root directory (e.g., '/path/to/external-data/olig_af_batch')
        pdb_id: PDB ID (e.g., '3mkm')
        cif_path: Path to experimental CIF file
        
    Returns:
        DataFrame with structure comparison results
    """
    # Load reference
    ref_intra, ref_inter = extract_contacts_from_cif(cif_path)
    seq_length = ref_intra.shape[0]
    
    # Find all run directories
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
        
        # Find all PDB files
        pdb_dir_path = run_dir / "pdb" / "recycles"
        if not pdb_dir_path.exists():
            continue
        
        for pdb_file in pdb_dir_path.glob("*_seed*.pdb"):
            # Parse filename: model_X_ptm_rY_seed0.pdb
            #fname_pattern = r"model_(\d+)_ptm_r(\d+)_seed(\d+)\.pdb"
            fname_pattern = r"model_(\d+)_multimer_v3_r(\d+)_seed(\d+)\.pdb"
            fname_match = re.match(fname_pattern, pdb_file.name)
            if not fname_match:
                continue
            
            model, recycle, seed = map(int, fname_match.groups())
            
            # Extract contacts
            pred_intra, pred_inter = extract_contacts_from_pdb(
                str(pdb_file), seq_length
            )
            
            # Compute metrics
            intra_metrics = compute_metrics(pred_intra, ref_intra)
            inter_metrics = compute_metrics(pred_inter, ref_inter)
            
            # Store result with CONSISTENT DATATYPES
            results.append({
                'pdb_id': pdb_id,
                'qid': str(qid),           # String categorical (matches data_utils)
                'qb': str(qb),             # String categorical (matches data_utils)
                'maskrate': int(maskrate), # Integer (matches data_utils)
                'seed': int(seed),         # Integer (matches data_utils)
                'model': f'model_{model}', # String categorical (matches data_utils format)
                'recycle': int(recycle),   # Integer (matches data_utils)
                'intra_f1': intra_metrics['f1'],
                'intra_precision': intra_metrics['precision'],
                'intra_recall': intra_metrics['recall'],
                'inter_f1': inter_metrics['f1'],
                'inter_precision': inter_metrics['precision'],
                'inter_recall': inter_metrics['recall'],
            })
    
    return pd.DataFrame(results)
