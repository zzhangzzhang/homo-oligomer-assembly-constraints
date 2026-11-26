"""Deterministic helpers for the reverse-assembly-discordance audit.

The functions in this module are deliberately policy-light and testable.  The
orchestrator owns file discovery and stage transitions; this module owns DOI
normalization, evidence/construct decisions, crystallographic reconstruction,
condition/ligand geometry, PISA parsing, ranking, and final classification.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import gemmi
except ImportError:  # pragma: no cover - surfaced with a useful CLI error
    gemmi = None


UNKNOWN = {"", ".", "?", "unknown", "not available", "n/a", "none"}
SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"
OLIGOMER_COUNTS = {
    "dimer": 2,
    "dimeric": 2,
    "trimer": 3,
    "trimeric": 3,
    "tetramer": 4,
    "tetrameric": 4,
    "pentamer": 5,
    "pentameric": 5,
    "hexamer": 6,
    "hexameric": 6,
    "heptamer": 7,
    "heptameric": 7,
    "octamer": 8,
    "octameric": 8,
    "decamer": 10,
    "decameric": 10,
    "dodecamer": 12,
    "dodecameric": 12,
    "oligomer": 2,
    "oligomeric": 2,
}
POSITIVE_STATE_RE = re.compile(
    r"\b(?:homo)?(?:di|tri|tetra|penta|hexa|hepta|octa|deca|dodeca)mer(?:ic)?\b"
    r"|\b(?:[2-9]|[1-9]\d+)[- ]?mer(?:ic)?\b"
    r"|\boligomer(?:ic|ization|ise|ize|s)?\b|\bmultimer(?:ic|s)?\b",
    re.I,
)
NEGATED_STATE_RE = re.compile(
    r"\b(?:not|non|no|neither|without|fails?\s+to|did\s+not)\b"
    r"[^.;:]{0,45}\b(?:oligomer|dimer|trimer|tetramer|multimer)",
    re.I,
)
CRYSTAL_ONLY_RE = re.compile(
    r"\b(?:crystal packing|crystallographic (?:contact|artefact|artifact)|"
    r"asymmetric unit|in the crystal)\b",
    re.I,
)
SOLUTION_METHOD_RE = re.compile(
    r"\b(?:SEC|size[- ]exclusion|gel filtration|MALS|AUC|"
    r"ultracentrifug|mass spectrom|native[- ]MS|SAXS|SANS|"
    r"light scattering|analytical ultracentrifug|cryo[- ]?EM|"
    r"cross[- ]?link|FRET|EMSA|sedimentation)\b",
    re.I,
)
RANGE_RE = re.compile(
    r"(?:residues?|aa|amino acids?)\s*[A-Za-z]?(-?\d+)\s*"
    r"(?:-|–|—|to)\s*[A-Za-z]?(-?\d+)",
    re.I,
)
MUTATION_RE = re.compile(r"\b([A-Z])(\d+)([A-Z])\b")
PH_RE = re.compile(r"\bpH\s*(?:of\s*)?([0-9]+(?:\.[0-9]+)?)", re.I)


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text)
    return text.rstrip(" .;,")


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in UNKNOWN else text


def truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_cif_block(path: str | Path):
    if gemmi is None:
        raise RuntimeError("gemmi is required (use the sam conda environment)")
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as handle:
            return gemmi.cif.read_string(handle.read()).sole_block()
    return gemmi.cif.read(path).sole_block()


def cif_category(block, prefix: str) -> dict[str, list[str]]:
    category = block.get_mmcif_category(prefix)
    return {str(key): list(values) for key, values in category.items()}


def assembly_monomer_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require every deposited biological-assembly row to have exact count 1."""
    if not rows:
        return {
            "strict_monomer": False,
            "reason": "no_deposited_assembly_rows",
            "assembly_count": 0,
            "assembly_ids": [],
            "oligomeric_counts": [],
        }
    parsed: list[int | None] = []
    ids: list[str] = []
    for index, row in enumerate(rows):
        ids.append(clean(row.get("id")) or str(index + 1))
        raw = clean(row.get("oligomeric_count"))
        try:
            count = int(raw)
        except (TypeError, ValueError):
            count = None
        parsed.append(count)
    if any(value is None for value in parsed):
        reason = "assembly_oligomeric_count_missing_or_invalid"
    elif any(value != 1 for value in parsed):
        reason = "at_least_one_deposited_assembly_nonmonomeric"
    else:
        reason = "all_deposited_assemblies_count_one"
    return {
        "strict_monomer": reason == "all_deposited_assemblies_count_one",
        "reason": reason,
        "assembly_count": len(rows),
        "assembly_ids": ids,
        "oligomeric_counts": parsed,
    }


def assembly_rows_from_block(block) -> list[dict[str, str]]:
    category = cif_category(block, "_pdbx_struct_assembly.")
    n = max((len(values) for values in category.values()), default=0)
    return [
        {key: values[index] if index < len(values) else "" for key, values in category.items()}
        for index in range(n)
    ]


def positive_oligomer_state(
    *,
    text: str,
    reported_state: str = "",
    subunit_count: int | None = None,
    assembly_class: str = "",
    biological_context: str = "",
) -> tuple[bool, str, int | None]:
    """Accept an affirmative >1 state, rejecting negation-only/crystal-only text."""
    combined = " ".join(filter(None, [reported_state, text])).strip()
    count = subunit_count if subunit_count and subunit_count > 0 else None
    if count is None:
        lower = combined.lower()
        for token, value in sorted(OLIGOMER_COUNTS.items(), key=lambda item: -len(item[0])):
            if re.search(rf"\b{re.escape(token)}\b", lower):
                count = value
                break
    # The evidence quote itself must state the positive assembly.  A model-
    # inferred subunit count or state label cannot rescue a quote that merely
    # says an intervention had "no effect on oligomerization".
    positive_spans = list(POSITIVE_STATE_RE.finditer(text))
    negated = list(NEGATED_STATE_RE.finditer(text))
    unnegated_spans = [
        match
        for match in positive_spans
        if not any(
            neg.start() <= match.start() <= neg.end() + 5
            for neg in negated
        )
    ]
    if count is not None and count > 1:
        positive = bool(unnegated_spans)
    else:
        positive = bool(unnegated_spans)
    if not positive:
        if positive_spans and negated:
            return False, "oligomer_words_only_negated", count
        return False, "no_positive_oligomer_state", count
    context = f"{biological_context} {combined}"
    if CRYSTAL_ONLY_RE.search(context) and not SOLUTION_METHOD_RE.search(context):
        return False, "crystallographic_evidence_only", count
    if assembly_class and assembly_class not in {
        "discrete_free_homo_oligomer",
        "templated_by_nucleic_acid",
        "membrane_embedded_assembly",
        "higher_order_assembly",
    }:
        return False, f"unsupported_assembly_class:{assembly_class}", count
    return True, "positive_oligomer_state", count


def evidence_quotes_verified(claim: Mapping[str, Any]) -> bool:
    evidence = list(claim.get("result_evidence") or [])
    if not evidence:
        return False
    return all(bool((item.get("verification") or {}).get("verified")) for item in evidence)


def flatten_claim(
    doi: str,
    uniprot: str,
    claim: Mapping[str, Any],
    construct: Mapping[str, Any],
) -> dict[str, Any] | None:
    claim_type = "event" if "to_target_subunit_counts" in claim else "state"
    if claim_type == "state":
        counts = [claim.get("subunit_count")]
        state = clean(claim.get("state_label"))
    else:
        counts = list(claim.get("to_target_subunit_counts") or [])
        state = "; ".join(map(str, claim.get("to_states") or []))
    count = max((int(value) for value in counts if value is not None), default=None)
    quote = " ".join(
        clean(item.get("quote")) for item in claim.get("result_evidence") or [] if clean(item.get("quote"))
    )
    accepted, reason, count = positive_oligomer_state(
        text=quote,
        reported_state=state,
        subunit_count=count,
        assembly_class=clean(
            claim.get("assembly_class")
            or ";".join(claim.get("to_assembly_classes") or [])
        ),
        biological_context=clean(claim.get("biological_context")),
    )
    citation_status = clean((claim.get("citation_scope") or {}).get("status"))
    origin = clean(claim.get("claim_origin"))
    direct = origin == "current_paper_experiment" and citation_status == "current_paper"
    verified = evidence_quotes_verified(claim)
    if not accepted or not verified or not direct:
        return None
    return {
        "doi": normalize_doi(doi),
        "uniprot": uniprot.upper(),
        "claim_id": clean(claim.get("claim_id")),
        "claim_type": claim_type,
        "state": state,
        "subunit_count": count,
        "quote": quote,
        "quote_verified": verified,
        "claim_origin": origin,
        "citation_status": citation_status,
        "method": clean(claim.get("method")),
        "method_result_link_type": clean(claim.get("method_result_link_type")),
        "biological_context": clean(claim.get("biological_context")),
        "conditions": "; ".join(
            map(str, claim.get("conditions") or claim.get("compared_conditions") or [])
        ),
        "condition_linked": bool(claim.get("condition_linked")),
        "trigger_type": clean(claim.get("trigger_type")),
        "trigger": clean(claim.get("trigger_name")),
        "construct_id": clean(claim.get("construct_id")),
        "construct_text": clean(construct.get("construct_text")),
        "construct_relation": clean(construct.get("relation_to_parent")),
        "construct_resolution": clean(construct.get("resolution_status")),
        "construct_modifications": "; ".join(map(str, construct.get("modifications") or [])),
        "evidence_reason": reason,
        "tier": clean((claim.get("deterministic_tiering") or {}).get("initial_tier")),
    }


def _normalized_species(values: Iterable[str]) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        for value in values
        if clean(value)
    }


def construct_compatibility(
    evidence: Mapping[str, Any],
    entity: Mapping[str, Any] | None,
    observed_range: tuple[int, int] | None,
) -> tuple[str, str]:
    """Conservative compatibility call; exact UniProt is assumed upstream."""
    if not entity:
        return "unresolved", "target_entity_metadata_missing"
    text = clean(evidence.get("construct_text"))
    resolution = clean(evidence.get("construct_resolution")).lower()
    relation = clean(evidence.get("construct_relation")).lower()
    mods = clean(evidence.get("construct_modifications"))
    pdb_mutation_count = int(entity.get("mutation_count") or 0)
    pdb_mutation_text = clean(entity.get("mutation_description"))
    if resolution and resolution != "resolved":
        return "unresolved", f"terra_construct_{resolution}"

    terra_mutations = {f"{a}{n}{b}" for a, n, b in MUTATION_RE.findall(f"{text} {mods}")}
    pdb_mutations = {f"{a}{n}{b}" for a, n, b in MUTATION_RE.findall(pdb_mutation_text)}
    variant_words = re.search(r"\b(?:mutant|variant|substitution|deletion)\b", f"{text} {mods}", re.I)
    if terra_mutations:
        if not terra_mutations <= pdb_mutations:
            return "incompatible", "terra_variant_not_present_in_pdb_entity"
    elif variant_words:
        return "unresolved", "variant_description_not_position_resolved"
    elif pdb_mutation_count > 0 and re.search(r"\b(?:wild type|wild-type|native)\b", text, re.I):
        return "incompatible", "terra_wild_type_but_pdb_entity_mutated"

    match = RANGE_RE.search(text)
    if match:
        expected = tuple(sorted((int(match.group(1)), int(match.group(2)))))
        if observed_range is None:
            return "unresolved", "sifts_observed_range_missing"
        overlap = max(0, min(expected[1], observed_range[1]) - max(expected[0], observed_range[0]) + 1)
        expected_len = expected[1] - expected[0] + 1
        if overlap / expected_len < 0.8:
            return "incompatible", f"construct_range_mismatch:{expected}:{observed_range}"
        if abs(expected[0] - observed_range[0]) <= 15 and abs(expected[1] - observed_range[1]) <= 15:
            return "compatible", f"construct_range_matches:{expected}:{observed_range}"
        return "unresolved", f"construct_range_partial_match:{expected}:{observed_range}"

    source_species = _normalized_species(entity.get("source_organisms") or [])
    species_mismatch = False
    if source_species:
        species_tokens = [species for species in source_species if len(species.split()) >= 2]
        explicit_other = re.search(r"\bfrom\s+([A-Z][a-z]+(?:\s+[a-z][a-z-]+))", text)
        if explicit_other:
            named = explicit_other.group(1).lower()
            species_mismatch = not any(
                named in species or species in named for species in species_tokens
            )
    if species_mismatch:
        return "incompatible", "terra_species_differs_from_pdb_source"

    full_length = relation == "full_length" or bool(
        re.search(r"\b(?:full[- ]length|wild type|wild-type|native protein)\b", text, re.I)
    )
    domain_like = bool(
        re.search(r"\b(?:domain|fragment|peptide|chain|subunit|terminus|terminal)\b", text, re.I)
    )
    if full_length and not domain_like and pdb_mutation_count == 0:
        return "compatible", "resolved_full_length_unmutated_construct"
    if not text:
        return "unresolved", "terra_construct_text_empty"
    return "unresolved", "construct_not_precisely_resolved_against_pdb"


def parse_sifts(path: str | Path, target_uniprot: str) -> dict[str, dict[int, int]]:
    with gzip.open(path, "rb") as handle:
        root = ET.parse(handle).getroot()
    mapping: dict[str, dict[int, int]] = {}
    for entity in root.findall(f"{SIFTS_NS}entity"):
        if entity.get("type") != "protein":
            continue
        for residue in entity.findall(f".//{SIFTS_NS}residue"):
            try:
                label_seq = int(residue.get("dbResNum"))
            except (TypeError, ValueError):
                continue
            pdb_xref = None
            up_xref = None
            for xref in residue.findall(f"{SIFTS_NS}crossRefDb"):
                if xref.get("dbSource") == "PDB":
                    pdb_xref = xref
                elif (
                    xref.get("dbSource") == "UniProt"
                    and xref.get("dbAccessionId", "").upper() == target_uniprot.upper()
                ):
                    up_xref = xref
            if pdb_xref is None or up_xref is None:
                continue
            try:
                up_pos = int(up_xref.get("dbResNum"))
            except (TypeError, ValueError):
                continue
            chain = pdb_xref.get("dbChainId")
            if chain:
                mapping.setdefault(chain, {})[label_seq] = up_pos
    return mapping


def sifts_observed_range(mapping: Mapping[str, Mapping[int, int]]) -> tuple[int, int] | None:
    values = [position for chain in mapping.values() for position in chain.values()]
    return (min(values), max(values)) if values else None


def _value_list(block, tag: str) -> list[str]:
    return [clean(value) for value in block.find_values(tag)]


def extract_conditions(block) -> dict[str, Any]:
    fields = {
        "method": "_exptl_crystal_grow.method",
        "pH": "_exptl_crystal_grow.pH",
        "pH_range": "_exptl_crystal_grow.pdbx_pH_range",
        "temperature_K": "_exptl_crystal_grow.temp",
        "temperature_details": "_exptl_crystal_grow.temp_details",
        "details": "_exptl_crystal_grow.pdbx_details",
    }
    values = {name: _value_list(block, tag) for name, tag in fields.items()}
    known = any(any(items) for items in values.values())
    return {
        **{name: "; ".join(item for item in items if item) for name, items in values.items()},
        "condition_metadata_known": known,
    }


def nonpolymer_inventory(structure) -> list[dict[str, Any]]:
    if len(structure) == 0:
        return []
    output = []
    for chain in structure[0]:
        for residue in chain:
            if residue.entity_type == gemmi.EntityType.Polymer:
                continue
            atoms = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for atom in residue
                if atom.element.name != "H"
            ]
            if not atoms:
                continue
            elements = sorted({atom.element.name for atom in residue if atom.element.name != "H"})
            output.append(
                {
                    "chain": chain.name,
                    "seqid": str(residue.seqid),
                    "comp_id": residue.name,
                    "elements": elements,
                    "is_water": residue.name.upper() in {"HOH", "DOD", "WAT"},
                    "is_metal": len(atoms) == 1 and elements[0] not in {"C", "N", "O", "S", "P"},
                    "coords": np.asarray(atoms, dtype=float),
                }
            )
    return output


def ligand_bridge_geometry(
    instances: Sequence[Mapping[str, Any]],
    chain_a_atoms: np.ndarray,
    chain_b_atoms: np.ndarray,
    cutoff: float = 4.0,
) -> list[dict[str, Any]]:
    """Return non-water instances with an atom within cutoff of both chains."""
    output = []
    cutoff2 = cutoff * cutoff
    for instance in instances:
        if instance.get("is_water"):
            continue
        coords = np.asarray(instance["coords"], dtype=float)
        if not len(coords) or not len(chain_a_atoms) or not len(chain_b_atoms):
            continue
        d2a = ((coords[:, None, :] - chain_a_atoms[None, :, :]) ** 2).sum(axis=2)
        d2b = ((coords[:, None, :] - chain_b_atoms[None, :, :]) ** 2).sum(axis=2)
        min_a = float(np.sqrt(d2a.min()))
        min_b = float(np.sqrt(d2b.min()))
        if min_a <= cutoff and min_b <= cutoff:
            output.append(
                {
                    "chain": instance.get("chain"),
                    "seqid": instance.get("seqid"),
                    "comp_id": instance.get("comp_id"),
                    "is_metal": bool(instance.get("is_metal")),
                    "elements": ";".join(instance.get("elements") or []),
                    "instance_transform": instance.get("instance_transform"),
                    "symmetry_op": instance.get("symmetry_op"),
                    "lattice": stable_json(instance.get("lattice")),
                    "min_distance_chain_A_A": min_a,
                    "min_distance_chain_B_A": min_b,
                    "cutoff_A": cutoff,
                }
            )
    return output


def transform_points(
    cartesian: np.ndarray,
    frac_matrix: np.ndarray,
    frac_translation: np.ndarray,
    cell_frac: np.ndarray,
    cell_orth: np.ndarray,
    lattice: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    frac = cartesian @ cell_frac.T
    transformed = frac @ frac_matrix.T + frac_translation + np.asarray(lattice, dtype=float)
    return transformed @ cell_orth.T


def contact_pairs(
    coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float = 8.0
) -> tuple[np.ndarray, float | None]:
    if not len(coords_a) or not len(coords_b):
        return np.empty((0, 2), dtype=int), None
    d2 = ((coords_a[:, None, :] - coords_b[None, :, :]) ** 2).sum(axis=2)
    pairs = np.argwhere(d2 < cutoff * cutoff)
    return pairs, float(np.sqrt(d2.min()))


def physical_pair_signature(coords_a: np.ndarray, coords_b: np.ndarray) -> str:
    def fingerprint(coords: np.ndarray) -> str:
        rounded = np.round(coords, 3)
        order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
        return hashlib.sha1(rounded[order].tobytes()).hexdigest()

    return "|".join(sorted((fingerprint(coords_a), fingerprint(coords_b))))


@dataclass
class CrystalInterface:
    source_chain: str
    partner_chain: str
    op_index: int
    op_triplet: str
    frac_matrix: np.ndarray
    frac_translation: np.ndarray
    lattice: tuple[int, int, int]
    ca_contact_count: int
    min_ca_distance: float
    contact_pairs: list[tuple[int, int]]
    physical_signature: str
    partner_cartesian: np.ndarray

    @property
    def transform_id(self) -> str:
        la, lb, lc = self.lattice
        return f"{self.source_chain}-{self.partner_chain}@{self.op_triplet}@({la},{lb},{lc})"


def _chain_ca(chain, mapping: Mapping[int, int]) -> tuple[np.ndarray, list[int]]:
    coords = []
    positions = []
    for residue in chain.get_polymer():
        if not residue.label_seq or int(residue.label_seq) not in mapping:
            continue
        ca = next((atom for atom in residue if atom.name == "CA"), None)
        if ca is None:
            continue
        coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        positions.append(mapping[int(residue.label_seq)])
    return np.asarray(coords, dtype=float), positions


def enumerate_crystal_interfaces(
    structure,
    sifts_mapping: Mapping[str, Mapping[int, int]],
    cutoff: float = 8.0,
    max_interfaces: int = 8,
) -> tuple[list[CrystalInterface], dict[str, Any]]:
    """Regenerate target-chain mates and group by physical chain-pair transform."""
    if len(structure) == 0:
        return [], {"reason": "no_coordinate_model"}
    sg = structure.find_spacegroup()
    if sg is None or not structure.cell.is_crystal():
        return [], {"reason": "not_a_crystal"}
    model = structure[0]
    chains = {chain.name: chain for chain in model if chain.name in sifts_mapping}
    tables = {
        name: _chain_ca(chain, sifts_mapping[name])
        for name, chain in chains.items()
    }
    tables = {name: value for name, value in tables.items() if len(value[0])}
    if not tables:
        return [], {"reason": "no_sifts_mapped_ca_atoms"}

    frac = np.asarray(structure.cell.frac.mat, dtype=float)
    orth = np.asarray(structure.cell.orth.mat, dtype=float)
    all_coords = np.concatenate([item[0] for item in tables.values()])
    span = all_coords.max(axis=0) - all_coords.min(axis=0)
    cell_lengths = [structure.cell.a, structure.cell.b, structure.cell.c]
    nmax = max(1, int(math.ceil((float(span.max()) + cutoff) / min(cell_lengths))) + 1)
    ops = list(sg.operations())
    grouped: dict[str, CrystalInterface] = {}
    evaluated = 0
    for source_name in sorted(tables):
        source_coords, source_pos = tables[source_name]
        source_min, source_max = source_coords.min(axis=0), source_coords.max(axis=0)
        for partner_name in sorted(tables):
            partner_coords, partner_pos = tables[partner_name]
            for op_index, op in enumerate(ops):
                matrix = np.asarray(op.rot, dtype=float) / op.DEN
                translation = np.asarray(op.tran, dtype=float) / op.DEN
                base = transform_points(partner_coords, matrix, translation, frac, orth)
                for la in range(-nmax, nmax + 1):
                    for lb in range(-nmax, nmax + 1):
                        for lc in range(-nmax, nmax + 1):
                            lattice = (la, lb, lc)
                            if (
                                op_index == 0
                                and lattice == (0, 0, 0)
                                and source_name == partner_name
                            ):
                                continue
                            partner = base + np.asarray(lattice) @ orth.T
                            pmin, pmax = partner.min(axis=0), partner.max(axis=0)
                            if np.any(pmax + cutoff < source_min) or np.any(
                                pmin - cutoff > source_max
                            ):
                                continue
                            evaluated += 1
                            pairs, minimum = contact_pairs(source_coords, partner, cutoff)
                            if not len(pairs):
                                continue
                            residue_pairs = sorted(
                                {
                                    (int(source_pos[i]), int(partner_pos[j]))
                                    for i, j in pairs
                                }
                            )
                            signature = physical_pair_signature(source_coords, partner)
                            candidate = CrystalInterface(
                                source_name,
                                partner_name,
                                op_index,
                                op.triplet(),
                                matrix,
                                translation,
                                lattice,
                                len(pairs),
                                float(minimum),
                                residue_pairs,
                                signature,
                                partner,
                            )
                            previous = grouped.get(signature)
                            if previous is None or (
                                -candidate.ca_contact_count,
                                candidate.min_ca_distance,
                                candidate.transform_id,
                            ) < (
                                -previous.ca_contact_count,
                                previous.min_ca_distance,
                                previous.transform_id,
                            ):
                                grouped[signature] = candidate
    ranked = sorted(
        grouped.values(),
        key=lambda item: (-item.ca_contact_count, item.min_ca_distance, item.transform_id),
    )
    return ranked[:max_interfaces], {
        "reason": "ok",
        "transforms_evaluated": evaluated,
        "physical_interfaces_found": len(ranked),
        "interfaces_retained": min(len(ranked), max_interfaces),
        "interfaces_truncated": max(0, len(ranked) - max_interfaces),
    }


def _transform_chain(chain, matrix, translation, lattice, structure):
    cloned = chain.clone()
    frac = np.asarray(structure.cell.frac.mat, dtype=float)
    orth = np.asarray(structure.cell.orth.mat, dtype=float)
    for residue in cloned:
        for atom in residue:
            point = np.asarray([[atom.pos.x, atom.pos.y, atom.pos.z]], dtype=float)
            result = transform_points(point, matrix, translation, frac, orth, lattice)[0]
            atom.pos = gemmi.Position(*map(float, result))
    return cloned


def write_two_chain_interface(
    structure,
    interface: CrystalInterface,
    cif_path: str | Path,
    pdb_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    model = structure[0]
    source_original = next(
        chain for chain in model if chain.name == interface.source_chain
    )
    partner_original = next(
        chain for chain in model if chain.name == interface.partner_chain
    )
    source = gemmi.Chain("A")
    partner_polymer = gemmi.Chain("B")
    for residue in source_original.get_polymer():
        source.add_residue(residue.clone())
    for residue in partner_original.get_polymer():
        partner_polymer.add_residue(residue.clone())
    partner = _transform_chain(
        partner_polymer,
        interface.frac_matrix,
        interface.frac_translation,
        interface.lattice,
        structure,
    )
    partner.name = "B"
    output = gemmi.Structure()
    output.name = structure.name
    output.cell = structure.cell
    output.spacegroup_hm = "P 1"
    out_model = gemmi.Model("1")
    out_model.add_chain(source)
    out_model.add_chain(partner)
    output.add_model(out_model)
    output.make_mmcif_document().write_file(str(cif_path))
    output.write_pdb(str(pdb_path))

    def heavy_atoms(chain):
        return np.asarray(
            [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for residue in chain.get_polymer()
                for atom in residue
                if atom.element.name != "H"
            ],
            dtype=float,
        )

    return heavy_atoms(source), heavy_atoms(partner)


def transformed_nonpolymer_instances(
    inventory: Sequence[Mapping[str, Any]],
    structure,
    chain_a_atoms: np.ndarray,
    chain_b_atoms: np.ndarray,
    cutoff: float = 4.0,
) -> list[dict[str, Any]]:
    """Generate all nearby crystallographic copies of each nonpolymer instance."""
    # Waters can number in the thousands and are not ligand/metal bridges for
    # this audit.  Excluding them before symmetry expansion is both the intended
    # policy and a large deterministic performance improvement.
    inventory = [item for item in inventory if not item.get("is_water")]
    if not inventory:
        return []
    sg = structure.find_spacegroup()
    if sg is None:
        return list(inventory)
    frac = np.asarray(structure.cell.frac.mat, dtype=float)
    orth = np.asarray(structure.cell.orth.mat, dtype=float)
    combined = np.concatenate((chain_a_atoms, chain_b_atoms))
    target_min, target_max = combined.min(axis=0), combined.max(axis=0)
    cell_lengths = [structure.cell.a, structure.cell.b, structure.cell.c]
    span = target_max - target_min
    nmax = max(1, int(math.ceil((float(span.max()) + cutoff) / min(cell_lengths))) + 1)
    output = []
    for op_index, op in enumerate(sg.operations()):
        matrix = np.asarray(op.rot, dtype=float) / op.DEN
        translation = np.asarray(op.tran, dtype=float) / op.DEN
        for item in inventory:
            base = transform_points(item["coords"], matrix, translation, frac, orth)
            for la in range(-nmax, nmax + 1):
                for lb in range(-nmax, nmax + 1):
                    for lc in range(-nmax, nmax + 1):
                        lattice = (la, lb, lc)
                        coords = base + np.asarray(lattice) @ orth.T
                        if np.any(coords.max(axis=0) + cutoff < target_min) or np.any(
                            coords.min(axis=0) - cutoff > target_max
                        ):
                            continue
                        copy = dict(item)
                        copy["coords"] = coords
                        copy["symmetry_op"] = op.triplet()
                        copy["op_index"] = op_index
                        copy["lattice"] = lattice
                        copy["instance_transform"] = (
                            f"{item.get('chain')}:{item.get('seqid')}@"
                            f"{op.triplet()}@{lattice}"
                        )
                        output.append(copy)
    return output


def number(value: str | None, kind=float):
    value = clean(value)
    try:
        return kind(value) if value else None
    except ValueError:
        return None


def pisa_support_decision(
    interface_area_A2: float | None,
    interaction_energy_kcal_mol: float | None,
    dissociation_energy_kcal_mol: float | None,
) -> bool:
    """Conservative local-PISA support rule used by the audit."""
    return bool(
        interface_area_A2 is not None
        and interface_area_A2 >= 400.0
        and interaction_energy_kcal_mol is not None
        and interaction_energy_kcal_mol < 0
        and dissociation_energy_kcal_mol is not None
        and dissociation_energy_kcal_mol > 0
    )


def parse_pisa_xml(interface_path: str | Path, assembly_path: str | Path) -> dict[str, Any]:
    interface_root = ET.parse(interface_path).getroot()
    protein_interfaces = []
    for interface in interface_root.findall(".//interface"):
        molecules = interface.findall("molecule")
        if len(molecules) != 2:
            continue
        if any(clean(molecule.findtext("class")) != "Protein" for molecule in molecules):
            continue
        protein_interfaces.append(
            {
                "id": number(interface.findtext("id"), int),
                "interface_area_A2": number(interface.findtext("int_area"), float),
                "buried_surface_area_A2": sum(
                    number(molecule.findtext("int_area"), float) or 0.0
                    for molecule in molecules
                ),
                "solvation_energy_kcal_mol": number(
                    interface.findtext("int_solv_en"), float
                ),
                "stabilization_energy_kcal_mol": number(
                    interface.findtext("stab_en"), float
                ),
                "p_value": number(interface.findtext("pvalue"), float),
            }
        )
    assembly_root = ET.parse(assembly_path).getroot()
    assembly = assembly_root.find(".//asu_complex/assembly")
    metrics = {
        "assembly_bsa_A2": None,
        "interaction_energy_kcal_mol": None,
        "dissociation_energy_kcal_mol": None,
    }
    if assembly is not None:
        metrics = {
            "assembly_bsa_A2": number(assembly.findtext("bsa"), float),
            "interaction_energy_kcal_mol": number(assembly.findtext("int_energy"), float),
            "dissociation_energy_kcal_mol": number(
                assembly.findtext("diss_energy"), float
            ),
        }
    best = max(
        protein_interfaces,
        key=lambda row: row.get("interface_area_A2") or 0.0,
        default={},
    )
    area = best.get("interface_area_A2")
    dissociation = metrics.get("dissociation_energy_kcal_mol")
    supported = pisa_support_decision(
        area,
        metrics.get("interaction_energy_kcal_mol"),
        dissociation,
    )
    return {
        **metrics,
        "n_protein_interfaces": len(protein_interfaces),
        "protein_interface_area_A2": area,
        "protein_interface_bsa_A2": best.get("buried_surface_area_A2"),
        "solvation_energy_kcal_mol": best.get("solvation_energy_kcal_mol"),
        "stabilization_energy_kcal_mol": best.get(
            "stabilization_energy_kcal_mol"
        ),
        "pisa_supported": supported,
        "pisa_support_rule": (
            "area>=400A2 and interaction_energy<0 and dissociation_energy>0"
        ),
    }


def ranking_features(row: Mapping[str, Any]) -> dict[str, Any]:
    method = clean(row.get("method"))
    quote = clean(row.get("quote"))
    trigger_type = clean(row.get("trigger_type")).lower()
    direct = int(
        clean(row.get("claim_origin")) == "current_paper_experiment"
        and clean(row.get("citation_status")) == "current_paper"
    )
    exact_quote = int(truth(row.get("quote_verified")) and bool(quote))
    construct = {"compatible": 2, "unresolved": 1, "incompatible": 0}.get(
        clean(row.get("construct_compatibility")).lower(), 0
    )
    clear_method = int(bool(SOLUTION_METHOD_RE.search(method)))
    stoichiometry = int(int(row.get("subunit_count") or 0) > 1)
    trigger_stratum = {
        "ligand": 0,
        "metal": 1,
        "ph": 2,
        "temperature": 3,
        "nucleic_acid": 4,
        "concentration": 5,
        "mutation": 6,
        "": 7,
    }.get(trigger_type, 8)
    return {
        "rank_direct_current_experiment": direct,
        "rank_exact_verified_quote": exact_quote,
        "rank_construct_resolution": construct,
        "rank_clear_solution_method": clear_method,
        "rank_explicit_stoichiometry": stoichiometry,
        "rank_trigger_stratum": trigger_stratum,
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    f = ranking_features(row)
    return (
        -f["rank_direct_current_experiment"],
        -f["rank_exact_verified_quote"],
        -f["rank_construct_resolution"],
        -f["rank_clear_solution_method"],
        -f["rank_explicit_stoichiometry"],
        f["rank_trigger_stratum"],
        -int(row.get("n_missed_bio_unique") or 0),
        clean(row.get("pdb")),
        clean(row.get("claim_id")),
    )


def explicit_trigger_absence(
    trigger_type: str,
    trigger: str,
    conditions: Mapping[str, Any],
    inventory_component_ids: Iterable[str],
) -> tuple[bool, str]:
    kind = clean(trigger_type).lower()
    trigger_text = clean(trigger)
    details = " ".join(
        clean(conditions.get(field))
        for field in ("method", "pH", "pH_range", "temperature_K", "details")
    )
    if kind in {"ligand", "metal"} and trigger_text:
        tokens = trigger_component_ids(kind, trigger_text)
        inventory = {value.upper() for value in inventory_component_ids}
        if tokens and not tokens & inventory:
            return True, f"trigger_components_absent_from_complete_inventory:{sorted(tokens)}"
    if kind == "ph":
        trigger_match = PH_RE.search(trigger_text)
        crystal_match = PH_RE.search(f"pH {clean(conditions.get('pH'))} {details}")
        if trigger_match and crystal_match:
            target = float(trigger_match.group(1))
            crystal = float(crystal_match.group(1))
            if abs(target - crystal) >= 1.0:
                return True, f"crystal_pH_{crystal:g}_differs_from_trigger_pH_{target:g}"
    return False, "trigger_absence_not_explicit"


def trigger_component_ids(trigger_type: str, trigger: str) -> set[str]:
    """Extract only explicit CCD-like IDs or unambiguous metal names."""
    kind = clean(trigger_type).lower()
    if kind not in {"ligand", "metal"}:
        return set()
    trigger_text = clean(trigger)
    ignored = {"DNA", "RNA", "PROTEIN", "WT", "SALT", "LOW", "HIGH"}
    components = {
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9]{1,4}\b", trigger_text)
        if token not in ignored
    }
    metal_names = {
        "zinc": "ZN",
        "magnesium": "MG",
        "calcium": "CA",
        "manganese": "MN",
        "iron": "FE",
        "copper": "CU",
        "cobalt": "CO",
        "nickel": "NI",
    }
    lower_trigger = trigger_text.lower()
    components.update(
        code for name, code in metal_names.items() if name in lower_trigger
    )
    return components


def bridge_trigger_match(
    trigger_type: str,
    trigger: str,
    bridge_component_ids: Iterable[str],
) -> dict[str, Any]:
    """Conservative exact-component bridge/trigger match with provenance."""
    expected = trigger_component_ids(trigger_type, trigger)
    observed = {
        clean(component).upper()
        for component in bridge_component_ids
        if clean(component)
    }
    matching = expected & observed
    if matching:
        reason = "exact_explicit_trigger_component_id_matches_bridge_component"
    elif not observed:
        reason = "no_geometric_bridge_components"
    elif not expected:
        reason = "literature_trigger_has_no_unambiguous_component_id"
    else:
        reason = "bridge_components_do_not_match_explicit_trigger_components"
    return {
        "bridge_matches_trigger": bool(matching),
        "bridge_matching_components": sorted(matching),
        "bridge_expected_trigger_components": sorted(expected),
        "bridge_observed_components": sorted(observed),
        "bridge_match_reason": reason,
        "bridge_match_rule": (
            "exact component ID explicitly named in trigger; metal names map "
            "to their element component ID; no chemical-similarity inference"
        ),
    }


def classify_record(row: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the approved mechanistic precedence deterministically."""
    compatibility = clean(row.get("construct_compatibility")).lower()
    if compatibility == "incompatible":
        return "construct_or_context_mismatch", clean(row.get("construct_reason"))
    if compatibility != "compatible":
        return "unresolved", "construct_compatibility_not_resolved"
    if truth(row.get("any_pisa_supported")):
        return "missed_lattice_assembly", "at_least_one_reconstructed_interface_PISA_supported"
    if truth(row.get("any_ligand_bridge")):
        if truth(row.get("bridge_matches_trigger")):
            reason = "trigger_matching_geometric_nonpolymer_bridge_detected"
        else:
            reason = "nonmatching_geometric_nonpolymer_bridge_detected"
        return "weak_or_ligand_bridged_lattice", reason
    if truth(row.get("explicit_trigger_absence")):
        return "condition_gated_alternate_state", clean(row.get("trigger_absence_reason"))
    if int(row.get("pisa_success_count") or 0) > 0:
        return "weak_or_ligand_bridged_lattice", "reconstructed_interfaces_not_PISA_supported"
    return "unresolved", "no_successful_local_PISA_result"


def classification_invariant_errors(row: Mapping[str, Any]) -> list[str]:
    """Return violations between evidence booleans and assigned class."""
    compatibility = clean(row.get("construct_compatibility")).lower()
    classification = clean(row.get("classification"))
    expected, _ = classify_record(row)
    errors = []
    if classification != expected:
        errors.append(f"classification_expected_{expected}_observed_{classification}")
    pisa = truth(row.get("any_pisa_supported"))
    bridge = truth(row.get("any_ligand_bridge"))
    absence = truth(row.get("explicit_trigger_absence"))
    successes = int(row.get("pisa_success_count") or 0)
    if classification == "missed_lattice_assembly" and not pisa:
        errors.append("missed_lattice_requires_pisa_support")
    if classification == "condition_gated_alternate_state" and (
        not absence or pisa or bridge
    ):
        errors.append("condition_gated_requires_absence_without_pisa_or_bridge")
    if (
        classification == "weak_or_ligand_bridged_lattice"
        and compatibility == "compatible"
        and (pisa or not (bridge or successes > 0))
    ):
        errors.append("weak_class_requires_no_pisa_and_bridge_or_success")
    if classification == "construct_or_context_mismatch" and compatibility != "incompatible":
        errors.append("construct_mismatch_class_requires_incompatible_construct")
    if classification == "unresolved" and compatibility == "compatible" and successes > 0:
        errors.append("compatible_successful_interfaces_cannot_be_unresolved")
    return errors

