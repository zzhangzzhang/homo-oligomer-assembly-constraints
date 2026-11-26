"""
Detect multimer run folders that have enough finished seeds at recycle 0.

Used to scope rebuild / DockQ to proteins where multi-seed inference is complete.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Union

_SEED_PAT = re.compile(r"model_1_multimer_v3_r0_seed(\d+)\.pdb$")


def seeds_at_r0_model1(recycles_dir: Path) -> set:
    """Distinct seed indices from model_1 recycle-0 PDBs under recycles_dir."""
    if not recycles_dir.is_dir():
        return set()
    seeds: set = set()
    for f in recycles_dir.glob("model_1_multimer_v3_r0_seed*.pdb"):
        m = _SEED_PAT.match(f.name)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def pdb_ids_with_min_seeds(
    data_path: Union[str, Path],
    pdb_ids: Sequence[str],
    *,
    min_seeds: int = 5,
    qids: Iterable[int] = (0, 30),
    qbs: Iterable[str] = ("True", "False"),
) -> List[str]:
    """
    Keep pdb_id only if every (qid, qb) run dir has at least ``min_seeds`` distinct seeds
    at recycle 0 (checked via model_1 ... r0_seed*.pdb).
    """
    root = Path(data_path)
    qbs = tuple(qbs)
    qids = tuple(qids)
    ready: List[str] = []
    for pid in pdb_ids:
        ok = True
        for qid in qids:
            for qb in qbs:
                rec = root / pid / f"{pid}_qid{qid}_qb{qb}" / "pdb" / "recycles"
                if len(seeds_at_r0_model1(rec)) < min_seeds:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            ready.append(pid)
    return ready
