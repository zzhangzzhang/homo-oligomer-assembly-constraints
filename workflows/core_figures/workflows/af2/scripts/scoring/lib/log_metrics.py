"""
Parse AlphaFold logs and return clean DataFrame.

Usage:
    from data_utils import analyze_results
    df = analyze_results('/path/to/external-data/olig_af_batch')
"""

import os
import re
import pandas as pd


def parse_af_log(file_path, pdb_id, qid, qb, maskrate):
    """Parse AlphaFold log file and extract metrics for every seed in the log."""
    pattern = re.compile(
        r"seed=(?P<seed>\d+)\s+model=(?P<model>[\w_]+)\s+recycle=(?P<recycle>\d+)\s+"
        r"plddt=(?P<plddt>[\d.]+)(?:\s+ptm=(?P<ptm>[\d.]+))?(?:\s+i_ptm=(?P<i_ptm>[\d.]+))?"
    )

    if not os.path.exists(file_path):
        return []

    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                seed_val = int(match.group("seed"))
                model_str = match.group('model')
                
                # Extract just the model number (e.g., 'model_1_multimer_v3' -> 1)
                model_num_match = re.search(r'model_(\d+)', model_str)
                if model_num_match:
                    model_num = int(model_num_match.group(1))
                    model = f'model_{model_num}'  # Consistent format: 'model_1', 'model_2', etc.
                else:
                    model = model_str  # Fallback to full name if pattern doesn't match
                
                data_list.append({
                    'pdb_id': pdb_id,
                    'qid': str(qid),           # String categorical
                    'qb': str(qb),             # String categorical  
                    'maskrate': int(maskrate), # Integer
                    'seed': seed_val,          # Integer (all seeds)
                    'model': model,            # String: 'model_1', 'model_2', etc.
                    'recycle': int(match.group('recycle')),  # Integer
                    'plddt': float(match.group('plddt')),    # Float
                    'ptm': float(match.group('ptm')) if match.group('ptm') else None,
                    'i_ptm': float(match.group('i_ptm')) if match.group('i_ptm') else None,
                })
    
    return data_list


def analyze_results(root_dir):
    """Walk through directory and gather all data into DataFrame."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Path {root_dir} does not exist.")
    
    all_results = []
    
    for pdb_dir in os.listdir(root_dir):
        pdb_path = os.path.join(root_dir, pdb_dir)
        if not os.path.isdir(pdb_path):
            continue
            
        for run_dir in os.listdir(pdb_path):
            # Match pattern: qidXX_qbTrue/False or qidXX_qbTrue/False_maskXX
            match = re.search(r"qid(\d+)_qb(True|False)(?:_mask(\d+))?", run_dir)
            if not match:
                continue
            
            qid, qb, mask = match.groups()
            maskrate = int(mask) if mask else 0
            
            log_file = os.path.join(pdb_path, run_dir, "log.txt")
            all_results.extend(parse_af_log(log_file, pdb_dir, qid, qb, maskrate))
    
    return pd.DataFrame(all_results)