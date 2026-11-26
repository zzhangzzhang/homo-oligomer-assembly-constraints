"""Conservative DOI-level protein identity grounding.

The graph built here is deliberately local and non-transitive.  A document
alias or construct is usable only when one exact source passage contains an
explicit identity anchor.  Manifest metadata is allow-listed so labels or
summaries produced by an earlier model cannot become identity evidence.
"""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from .aliases import (
    MAX_GENE_ALIASES_PER_ENTITY,
    MAX_IDENTITY_LABELS_PER_TARGET,
    MAX_MATCH_LABEL_CHARS,
    bounded_match_labels,
    normalize_gene_aliases,
)
from .doi import normalize_doi


CLAIM_STATUSES = (
    "exact",
    "construct_of_exact",
    "ambiguous",
    "mismatch",
    "unlinked",
)
IDENTITY_POLICY = {
    "explicit_anchor_required": True,
    "transitive_alias_propagation": False,
    "bare_cooccurrence_is_identity": False,
    "prefix_inference": False,
    "slash_group_inference": False,
    "prior_model_output_fields_used": False,
    "claim_evidence_must_be_exact": True,
    "explicit_competitor_overrides_focus_alias": True,
    "constructs_never_project_to_full_length": True,
    "manifest_aliases_require_unique_doi_target": True,
    "entry_titles_require_single_exact_accession_entity": True,
    "sanctioned_alias_construct_path_depth": 2,
    "max_gene_aliases_per_entity": MAX_GENE_ALIASES_PER_ENTITY,
    "max_identity_labels_per_target": MAX_IDENTITY_LABELS_PER_TARGET,
    "max_identity_label_chars": MAX_MATCH_LABEL_CHARS,
}
NODE_TYPES = (
    "canonical_protein",
    "canonical_accession",
    "canonical_gene",
    "canonical_species",
    "paper_alias",
    "exact_construct",
    "exact_domain",
    "exact_variant",
    "exact_processed_product",
    "competing_protein",
    "pdb_entity",
    "canonical_name_variant",
    "entry_title_alias",
)
CONSTRUCT_RELATIONS = {
    "construct": ("exact_construct", "construct_of"),
    "fragment": ("exact_construct", "fragment_of"),
    "truncation": ("exact_construct", "truncation_of"),
    "domain": ("exact_domain", "domain_of"),
    "variant": ("exact_variant", "variant_of"),
    "mutant": ("exact_variant", "variant_of"),
    "processed product": ("exact_processed_product", "processed_product_of"),
    "proteolytic product": ("exact_processed_product", "processed_product_of"),
    "cleavage product": ("exact_processed_product", "processed_product_of"),
}
_CONSTRUCT_NODE_TYPES = {
    "exact_construct",
    "exact_domain",
    "exact_variant",
    "exact_processed_product",
}
_MANIFEST_ENTITY_FIELDS = (
    "pdb_id",
    "entity_id",
    "chains",
    "description",
    "polymer_type",
    "mutation_description",
    "sequence_length",
    "mutation_count",
    "source_organisms",
    "source_genes",
    "source_gene_alias_audit",
    "entry_title",
    "entry_polymer_entity_count",
    "entry_distinct_uniprot_accessions",
)
_PROTEIN_NODE_TYPES = {
    "canonical_protein",
    "canonical_accession",
    "canonical_gene",
    "competing_protein",
    "paper_alias",
    "canonical_name_variant",
    "entry_title_alias",
} | _CONSTRUCT_NODE_TYPES

_GENERIC_NAMES = {"protein", "enzyme", "protease"}
_GENERIC_LEADING = re.compile(
    r"^(?:bifunctional\s+protein|protein|lipoprotein)\s+",
    re.I,
)
_CONSTRUCT_RELATION_TYPES = {
    "construct_of",
    "fragment_of",
    "truncation_of",
    "domain_of",
    "variant_of",
    "processed_product_of",
}
_COMMON_SPECIES_FORMS = {
    "homo sapiens": {"human", "humans"},
    "mus musculus": {"mouse", "mice", "murine"},
    "rattus norvegicus": {"rat", "rats"},
    "escherichia coli": {"e. coli", "e coli"},
    "saccharomyces cerevisiae": {"yeast", "budding yeast", "s. cerevisiae"},
    "drosophila melanogaster": {"drosophila", "fruit fly"},
}


def _normalized(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:")


def _identity_key(value):
    """Comparison key for conservative protein-name punctuation variants."""
    text = _normalized(value)
    text = re.sub(r"\balpha(?=[a-z0-9]|\b)", "a", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _informative_name(value):
    words = re.findall(r"[a-z0-9]+", _normalized(value))
    return (
        bool(words)
        and len(str(value)) <= MAX_MATCH_LABEL_CHARS
        and any(len(word) >= 2 for word in words)
        and " ".join(words) not in _GENERIC_NAMES
    )


def _canonical_name_variants(value):
    """Return display variants; uniqueness is enforced across DOI targets."""
    original = str(value or "").strip()
    if not original:
        return set()
    variants = set()
    punctuation = re.sub(r"[_/,:;()]+", " ", original)
    punctuation = re.sub(r"[\u2010-\u2015-]+", " ", punctuation)
    punctuation = re.sub(r"\s+", " ", punctuation).strip()
    if punctuation and _normalized(punctuation) != _normalized(original):
        variants.add(punctuation)
    stripped = _GENERIC_LEADING.sub("", punctuation or original).strip()
    if stripped and _identity_key(stripped) != _identity_key(original):
        variants.add(stripped)

    # PDB descriptions commonly use "NAME, ADJECTIVE"; retain both words but
    # put the adjective in the form used by prose.
    adjective = re.fullmatch(r"\s*([^,]+),\s*([A-Za-z][A-Za-z -]+)\s*", original)
    if adjective:
        variants.add("%s %s" % (adjective.group(2), adjective.group(1)))

    for base in tuple(variants | {punctuation or original}):
        if re.search(r"\balpha\b|α", base, re.I):
            variants.add(re.sub(r"\balpha\b|α", "a", base, flags=re.I))
        mass_after = re.fullmatch(
            r"\s*(?P<name>[A-Za-z][A-Za-z0-9 -]*?)\s+"
            r"(?P<mass>\d+(?:\.\d+)?\s*kDa)\s*",
            base,
            re.I,
        )
        if mass_after:
            variants.add("%s %s" % (mass_after.group("mass"), mass_after.group("name")))
        mass_before = re.fullmatch(
            r"\s*(?P<mass>\d+(?:\.\d+)?\s*kDa)\s+"
            r"(?P<name>[A-Za-z][A-Za-z0-9 -]*)\s*",
            base,
            re.I,
        )
        if mass_before:
            variants.add("%s %s" % (mass_before.group("name"), mass_before.group("mass")))
        numbered_name = re.fullmatch(
            r"(?P<name>[A-Za-z][A-Za-z0-9 -]{5,}?)\s+(?:I|II|III|IV|V|VI)",
            base,
            re.I,
        )
        if numbered_name:
            variants.add(numbered_name.group("name").strip())
    return {
        re.sub(r"\s+", " ", item).strip()
        for item in variants
        if _informative_name(item) and _normalized(item) != _normalized(original)
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(kind, value):
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return "%s:%s" % (kind, digest)


def identity_graph_hashes(identity_graph):
    """Return stable hashes for the graph and its declared grounding policy."""
    policy = identity_graph.get("policy") or {}
    return {
        "identity_graph_sha256": hashlib.sha256(
            _canonical_json(identity_graph).encode("utf-8")
        ).hexdigest(),
        "identity_policy_sha256": hashlib.sha256(
            _canonical_json(policy).encode("utf-8")
        ).hexdigest(),
    }


def _strings(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _entity_key(entity):
    return (
        str(entity.get("pdb_id") or "").upper(),
        str(entity.get("entity_id") or ""),
        _canonical_json(entity),
    )


def _clean_entity(raw):
    entity = {}
    for field in _MANIFEST_ENTITY_FIELDS:
        if field not in raw:
            continue
        value = raw[field]
        if field in {
            "chains",
            "source_organisms",
            "entry_distinct_uniprot_accessions",
        }:
            value = _strings(value)
        elif field == "source_genes":
            value, audit = normalize_gene_aliases(
                value,
                prior_audit=raw.get("source_gene_alias_audit"),
            )
            entity["source_gene_alias_audit"] = audit
        elif field == "source_gene_alias_audit":
            if "source_gene_alias_audit" in entity:
                continue
            value = dict(value) if isinstance(value, dict) else {}
        elif field == "pdb_id":
            value = str(value or "").upper()
        entity[field] = value
    return entity


def _safe_entry_title(entity, accession):
    accessions = sorted(
        {
            str(value).strip().upper()
            for value in entity.get("entry_distinct_uniprot_accessions") or []
            if str(value).strip()
        }
    )
    title = entity.get("entry_title")
    if (
        entity.get("entry_polymer_entity_count") == 1
        and accessions == [str(accession or "").upper()]
        and isinstance(title, str)
        and title.strip()
    ):
        return title.strip()
    return ""


def _entry_title_extractions(title):
    """Extract only the subject of a conservative ``structure of X from Y``."""
    match = re.search(
        r"\b(?:the\s+)?structure\s+of\s+(?P<name>.+?)\s+from\s+"
        r"[A-Za-z0-9]",
        str(title or ""),
        re.I,
    )
    if not match:
        return []
    name = match.group("name").strip(" \t\r\n\"'.,;:")
    if (
        not _informative_name(name)
        or len(name) > 80
        or re.search(
            r"\b(?:complex\s+with|bound\s+to|and|or)\b|[&+]",
            name,
            re.I,
        )
    ):
        return []
    return [(name, "structure_of_name_from_organism")]


def _entry_title_construct_extractions(title, target_names=None):
    """Type construct names from a uniquely accession-bound RCSB title."""
    raw = str(title or "").strip()
    normalized = _normalized(raw)
    if not raw or re.search(r"\b(?:complex\s+with|bound\s+to)\b|[&+]", raw, re.I):
        return []

    relation = None
    node_type = None
    if re.search(r"\b(?:self[- ]cleav|proteoly|processed|mature product)\w*\b", normalized):
        node_type, relation = "exact_processed_product", "processed_product_of"
    elif re.search(r"\b(?:n|c)[- ]terminal\s+domain\b|\b[a-z0-9-]+\s+domain\b", normalized):
        node_type, relation = "exact_domain", "domain_of"
    elif re.search(r"\bfragment\b", normalized):
        node_type, relation = "exact_construct", "fragment_of"

    values = []
    if relation:
        subject = re.sub(
            r"^(?:(?:refined|solution|nmr|crystal)\s+)?structure\s+of\s+",
            "",
            raw,
            flags=re.I,
        )
        subject = re.split(r"\s+(?:in\s+complex\s+with|bound\s+to)\s+", subject, 1, flags=re.I)[0]
        subject = re.split(r"\s*:\s*(?:mutant|high|low|intermediate|neutral)\b", subject, 1, flags=re.I)[0]
        subject = subject.strip(" \t\r\n\"'.,;:")
        if _informative_name(subject) and len(subject) <= 120:
            values.append((subject, node_type, relation, "typed_title_subject"))
        descriptor = re.search(
            r"\b(?P<name>(?:(?:N|C)[- ]terminal|[A-Za-z0-9-]{1,20})\s+domain)\b",
            raw,
            re.I,
        )
        if descriptor:
            values.append(
                (
                    descriptor.group("name"),
                    node_type,
                    relation,
                    "typed_title_descriptor",
                )
            )
        fragment = re.search(
            r"\b(?P<name>(?:(?:N|C)[- ]terminal\s+)?fragment)\b",
            raw,
            re.I,
        )
        if fragment:
            values.append(
                (
                    fragment.group("name"),
                    node_type,
                    relation,
                    "typed_title_descriptor",
                )
            )
        for parenthetical in re.findall(r"\(([^()\n]{1,80})\)", raw):
            alias = _clean_alias(parenthetical)
            if alias:
                values.append((alias, node_type, relation, "typed_title_parenthetical"))

    canonical_is_precursor = any(
        re.search(r"\b(?:polyprotein|precursor)\b", str(name or ""), re.I)
        for name in target_names or []
    )
    if canonical_is_precursor:
        product = re.search(
            r"\b(?P<name>(?:(?:[A-Za-z0-9.-]+\s+){0,5}"
            r"(?:capsid|matrix|nucleocapsid)\s+protein))\b",
            raw,
            re.I,
        )
        if product:
            values.append(
                (
                    re.search(
                        r"(?:capsid|matrix|nucleocapsid)\s+protein",
                        product.group("name"),
                        re.I,
                    ).group(0),
                    "exact_processed_product",
                    "processed_product_of",
                    "typed_title_polyprotein_product_descriptor",
                )
            )

    unique = {}
    for label, extracted_type, extracted_relation, rule in values:
        cleaned = str(label or "").strip(" \t\r\n\"'.,;:")
        if cleaned and len(cleaned) <= 120:
            unique[
                (_identity_key(cleaned), extracted_type, extracted_relation)
            ] = (cleaned, extracted_type, extracted_relation, rule)
    return sorted(unique.values(), key=lambda item: (_normalized(item[0]), item[1:]))


def collect_doi_targets(manifest_rows, doi=None):
    """Return every canonical manifest target for one normalized DOI.

    Duplicate rows for the same DOI/accession are merged.  Only primary
    manifest identity fields are consumed; unknown fields are ignored.
    """
    rows = list(manifest_rows or [])
    normalized_dois = sorted(
        {
            normalize_doi(row.get("doi"))
            for row in rows
            if isinstance(row, dict) and row.get("doi")
        }
    )
    if doi is None:
        if len(normalized_dois) != 1:
            raise ValueError("doi is required when manifest rows span zero or multiple DOIs")
        doi = normalized_dois[0]
    else:
        doi = normalize_doi(doi)

    merged = {}
    for raw in rows:
        if not isinstance(raw, dict) or not raw.get("doi"):
            continue
        if normalize_doi(raw["doi"]) != doi:
            continue
        accession = str(raw.get("uniprot_accession") or "").strip().upper()
        group_id = str(raw.get("group_id") or "").strip()
        identity_key = accession or ("group:" + group_id)
        if not identity_key or identity_key == "group:":
            continue
        item = merged.setdefault(
            identity_key,
            {
                "doi": doi,
                "uniprot_accession": accession,
                "group_ids": set(),
                "target_names": set(),
                "pdb_ids": set(),
                "genes": set(),
                "species": set(),
                "entry_titles": set(),
                "entities": {},
            },
        )
        if group_id:
            item["group_ids"].add(group_id)
        bounded_names, _ = bounded_match_labels(raw.get("target_names") or [], 32)
        item["target_names"].update(bounded_names)
        item["pdb_ids"].update(
            value.upper() for value in _strings(raw.get("pdb_ids"))
        )
        for raw_entity in raw.get("entities") or []:
            if not isinstance(raw_entity, dict):
                continue
            entity = _clean_entity(raw_entity)
            genes, gene_audit = normalize_gene_aliases(
                entity.get("source_genes"),
                prior_audit=entity.get("source_gene_alias_audit"),
            )
            entity["source_genes"] = genes
            entity["source_gene_alias_audit"] = gene_audit
            item["genes"].update(genes)
            item["species"].update(_strings(entity.get("source_organisms")))
            if entity.get("pdb_id"):
                item["pdb_ids"].add(entity["pdb_id"])
            safe_title = _safe_entry_title(entity, accession)
            if safe_title:
                item["entry_titles"].add(safe_title)
            item["entities"][_entity_key(entity)] = entity

    targets = []
    for identity_key, raw in sorted(merged.items()):
        names, name_audit = bounded_match_labels(raw["target_names"], 32)
        genes, gene_audit = bounded_match_labels(
            raw["genes"], MAX_GENE_ALIASES_PER_ENTITY, max_chars=64
        )
        species, species_audit = bounded_match_labels(raw["species"], 32)
        groups = sorted(raw["group_ids"])
        target_identity = {
            "doi": doi,
            "group_id": groups[0] if groups else "",
            "group_ids": groups,
            "uniprot_accession": raw["uniprot_accession"],
            "target_names": names,
            "genes": genes,
            "species": species,
            "pdb_ids": sorted(raw["pdb_ids"]),
            "entry_titles": sorted(
                raw["entry_titles"], key=lambda value: (_normalized(value), value)
            ),
            "alias_audit": {
                "target_names": name_audit,
                "genes": gene_audit,
                "species": species_audit,
            },
        }
        target_identity["target_id"] = _stable_id(
            "target",
            [doi, raw["uniprot_accession"] or identity_key],
        )
        target_identity["entities"] = sorted(raw["entities"].values(), key=_entity_key)
        targets.append(target_identity)
    targets = sorted(
        targets,
        key=lambda item: (
            item["uniprot_accession"],
            item["group_id"],
            item["target_id"],
        ),
    )
    ownership = defaultdict(set)
    candidate_values = defaultdict(lambda: defaultdict(set))
    for target in targets:
        for label in (
            list(target["target_names"])
            + list(target["genes"])
            + _strings(target["uniprot_accession"])
        ):
            ownership[_identity_key(label)].add(target["target_id"])
    for target in targets:
        for name in target["target_names"]:
            for variant in _canonical_name_variants(name):
                ownership[_identity_key(variant)].add(target["target_id"])
                candidate_values[target["target_id"]]["canonical_name_variants"].add(
                    variant
                )
        for title in target["entry_titles"]:
            typed_title = _entry_title_construct_extractions(
                title, target.get("target_names")
            )
            if _informative_name(title) and not typed_title:
                ownership[_identity_key(title)].add(target["target_id"])
                candidate_values[target["target_id"]]["entry_title_aliases"].add(title)
            for extracted, _ in _entry_title_extractions(title) if not typed_title else []:
                ownership[_identity_key(extracted)].add(target["target_id"])
                candidate_values[target["target_id"]]["entry_title_aliases"].add(
                    extracted
                )
            for extracted, node_type, relation_type, rule in typed_title:
                ownership[_identity_key(extracted)].add(target["target_id"])
                candidate_values[target["target_id"]]["entry_title_constructs"].add(
                    (extracted, node_type, relation_type, title, rule)
                )
    for target in targets:
        target_id = target["target_id"]
        for field in ("canonical_name_variants", "entry_title_aliases"):
            target[field] = sorted(
                {
                    value
                    for value in candidate_values[target_id][field]
                    if ownership[_identity_key(value)] == {target_id}
                    and _informative_name(value)
                },
                key=lambda value: (_normalized(value), value),
            )
        target["entry_title_constructs"] = [
            {
                "label": value,
                "node_type": node_type,
                "relation_type": relation_type,
                "full_entry_title": title,
                "extraction_rule": rule,
            }
            for value, node_type, relation_type, title, rule in sorted(
                candidate_values[target_id]["entry_title_constructs"],
                key=lambda item: (_normalized(item[0]), item[1:]),
            )
            if ownership[_identity_key(value)] == {target_id}
        ]
    return targets


def _species_forms(species):
    """Derive comparison forms from supplied taxa, never from a fixed map."""
    normalized = _normalized(species)
    forms = {normalized} if normalized else set()
    words = re.findall(r"[a-z0-9-]+", normalized)
    if len(words) >= 2:
        forms.add("%s. %s" % (words[0][0], words[1]))
        forms.add("%s %s" % (words[0][0], words[1]))
    forms.update(_COMMON_SPECIES_FORMS.get(normalized, set()))
    return forms


def compare_species(canonical_species, observed_species):
    """Compare arbitrary supplied organism names using manifest-derived forms."""
    canonical = _strings(canonical_species)
    observed = _strings(observed_species)
    if not canonical or not observed:
        return {
            "status": "unlinked",
            "canonical_species": canonical,
            "observed_species": observed,
            "matched_species": [],
        }
    matches = []
    for expected in canonical:
        expected_forms = _species_forms(expected)
        if any(_normalized(value) in expected_forms for value in observed):
            matches.append(expected)
    status = "exact" if len(matches) == 1 else "ambiguous" if len(matches) > 1 else "mismatch"
    return {
        "status": status,
        "canonical_species": canonical,
        "observed_species": observed,
        "matched_species": sorted(matches),
    }


compare_organisms = compare_species


def _target_labels(target):
    categories = (
        _strings(target.get("uniprot_accession")),
        target.get("target_names") or [],
        target.get("genes") or [],
        target.get("canonical_name_variants") or [],
        target.get("entry_title_aliases") or [],
        [
            item.get("label", "")
            for item in target.get("entry_title_constructs") or []
            if isinstance(item, dict)
        ],
        target.get("_paper_aliases") or [],
    )
    labels = []
    seen = set()
    for values in categories:
        bounded, _ = bounded_match_labels(
            values, MAX_IDENTITY_LABELS_PER_TARGET
        )
        for value in bounded:
            key = _normalized(value)
            if not key or key in seen:
                continue
            seen.add(key)
            if len(labels) < MAX_IDENTITY_LABELS_PER_TARGET:
                labels.append(value)
    return labels


def _term_pattern(term):
    if not (2 <= len(str(term or "").strip()) <= MAX_MATCH_LABEL_CHARS):
        return r"(?!x)x"
    pieces = re.findall(r"[A-Za-zΑ-Ωα-ω0-9]+", str(term or ""))
    if not pieces:
        return r"(?!x)x"
    patterns = []
    for piece in pieces:
        if _normalized(piece) == "alpha":
            patterns.append(r"(?:alpha|a|α)")
        else:
            patterns.append(re.escape(piece))
    return r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % (
        r"[\W_]*".join(patterns)
    )


@lru_cache(maxsize=4096)
def _compiled_term_pattern(term):
    return re.compile(_term_pattern(term), re.I)


@lru_cache(maxsize=4096)
def _compiled_ending_term_pattern(term):
    return re.compile(_term_pattern(term) + r"\s*$", re.I)


def _contains_term(text, term):
    return _contains_term_cached(str(text or ""), str(term or "").strip())


@lru_cache(maxsize=32768)
def _contains_term_cached(text, term):
    return bool(_compiled_term_pattern(term).search(text))


def _sentences(text):
    raw = str(text or "")
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", raw):
        anchor = match.group(0).strip()
        if anchor:
            yield anchor


def _passage_records(passages, doi):
    records = []
    for index, passage in enumerate(passages or []):
        if not isinstance(passage, dict):
            continue
        if passage.get("doi") and normalize_doi(passage["doi"]) != doi:
            continue
        source_id = str(passage.get("source_id") or "").strip()
        text = str(passage.get("text") or "")
        if not source_id or not text:
            continue
        records.append(
            {
                "source_id": source_id,
                "text": text,
                "index": index,
                "species": _strings(
                    passage.get("species") or passage.get("organism")
                ),
            }
        )
    return sorted(records, key=lambda item: (item["source_id"], item["index"], item["text"]))


def _target_candidates(anchor, targets):
    candidates = []
    for target in targets:
        if any(_contains_term(anchor, label) for label in _target_labels(target)):
            candidates.append(target)
    if len(candidates) == 1:
        return [] if _anchor_species_mismatch(anchor, candidates[0]) else candidates
    if not candidates:
        return candidates
    species_tied = [
        target
        for target in candidates
        if any(
            any(_contains_term(anchor, form) for form in _species_forms(species))
            for species in target["species"]
        )
    ]
    return species_tied or candidates


def _anchor_species_mentions(anchor, target):
    observed = set()
    for species in target.get("species") or []:
        words = re.findall(r"[A-Za-z][A-Za-z-]*", str(species))
        if len(words) < 2:
            continue
        genus = words[0]
        pattern = (
            r"(?<![A-Za-z])(?P<species>(?:(?i:%s)|%s\.)\s+[a-z][a-z-]+)"
            % (re.escape(genus), re.escape(genus[0].upper()))
        )
        observed.update(
            match.group("species")
            for match in re.finditer(pattern, str(anchor or ""))
        )
    return sorted(observed)


def _anchor_species_mismatch(anchor, target):
    observed = _anchor_species_mentions(anchor, target)
    return bool(
        observed
        and compare_species(target.get("species") or [], observed)["status"]
        == "mismatch"
    )


def _targets_ending(text, targets):
    stripped = str(text or "").rstrip()
    candidates = []
    for target in targets:
        for label in _target_labels(target):
            if _compiled_ending_term_pattern(label).search(stripped):
                candidates.append(target)
                break
    if len(candidates) == 1 and _anchor_species_mismatch(stripped, candidates[0]):
        return []
    if len(candidates) > 1:
        species_tied = [
            target
            for target in candidates
            if any(
                any(_contains_term(stripped, form) for form in _species_forms(species))
                for species in target["species"]
            )
        ]
        if species_tied:
            candidates = species_tied
    return candidates


def _clean_alias(value):
    alias = str(value or "").strip(" \t\r\n\"'.,;:")
    if (
        not alias
        or len(alias) > 80
        or "/" in alias
        or re.search(r"\b(?:and|or)\b", alias, re.I)
        or re.search(
            r"\b(?:domain|construct|fragment|truncation|variant|mutant|product)\s+of\b",
            alias,
            re.I,
        )
    ):
        return ""
    if not re.search(r"[A-Za-zΑ-Ωα-ω0-9]", alias):
        return ""
    return alias


def _parenthetical_relations(anchor, targets):
    relations = []
    for match in re.finditer(r"\(([^()\n]{1,100})\)", anchor):
        inside = _clean_alias(match.group(1))
        if not inside:
            continue
        left_targets = _targets_ending(anchor[: match.start()], targets)
        if len(left_targets) == 1:
            target = left_targets[0]
            if any(
                _identity_key(inside) == _identity_key(label)
                for label in _target_labels(target)
            ):
                continue
            left = anchor[: match.start()].rstrip()
            if re.search(
                r"\b(?:Gag(?:-Pro(?:-Pol)?)?|precursors?|partners?)\s*$",
                left,
                re.I,
            ):
                continue
            typed_title_relations = {
                (item["node_type"], item["relation_type"])
                for item in target.get("entry_title_constructs") or []
                if _compiled_ending_term_pattern(item.get("label", "")).search(left)
            }
            if len(typed_title_relations) == 1:
                node_type, relation_type = next(iter(typed_title_relations))
                relations.append((inside, node_type, relation_type, target))
            elif re.search(
                r"\b(?:N|C)[- ]terminal\s+fragment\s+of\s+",
                left,
                re.I,
            ):
                relations.append(
                    (inside, "exact_construct", "fragment_of", target)
                )
            elif re.match(r"wt(?:\s|[-_])", inside, re.I) and re.search(
                r"\b\d+(?:\.\d+)?\s*kDa\b", left, re.I
            ):
                relations.append(
                    (inside, "exact_variant", "variant_of", target)
                )
            else:
                relations.append((inside, "paper_alias", "alias_of", target))
            continue
    return relations


def _pdb_parenthetical_relations(anchor, targets):
    """Bind a paper name only when its parenthetical cites one unique target PDB."""
    relations = []
    for match in re.finditer(
        r"\((?P<alias>[A-Za-zΑ-Ωα-ω0-9][A-Za-zΑ-Ωα-ω0-9_.-]{1,39})"
        r"\s*,\s*PDB\s*:\s*(?P<pdb>[0-9][A-Za-z0-9]{3})\)",
        str(anchor or ""),
        re.I,
    ):
        pdb_id = match.group("pdb").upper()
        matches = [
            target
            for target in targets
            if pdb_id in {str(value).upper() for value in target.get("pdb_ids") or []}
        ]
        if len(matches) == 1 and not _anchor_species_mismatch(anchor, matches[0]):
            relations.append(
                (match.group("alias"), "paper_alias", "alias_of", matches[0])
            )
    return relations


def _marked_alias_relations(anchor, targets, candidates=None):
    relations = []
    marker = (
        r"(?:also\s+known\s+as|also\s+called|hereafter(?:\s+called)?|"
        r"abbreviated\s+as|referred\s+to\s+as|designated)"
    )
    candidates = (
        _target_candidates(anchor, targets)
        if candidates is None
        else candidates
    )
    if len(candidates) != 1:
        return relations
    for target in candidates:
        for label in _target_labels(target):
            match = re.search(
                _term_pattern(label)
                + r"\s*(?:,|is|was)?\s*"
                + marker
                + r"\s+([A-Za-zΑ-Ωα-ω0-9][A-Za-zΑ-Ωα-ω0-9_.-]{1,39})",
                anchor,
                re.I,
            )
            if match:
                alias = _clean_alias(match.group(1))
                if alias:
                    relations.append((alias, "paper_alias", "alias_of", target))
    return relations


def _descriptor_relations(anchor, targets, candidates=None):
    relations = []
    descriptor_pattern = (
        r"(?P<descriptor>(?:(?:N|C)[- ]terminal\s+)?(?:domain|construct|fragment|"
        r"truncation)|variant|mutant|processed\s+product|proteolytic\s+product|"
        r"cleavage\s+product)"
    )
    alias_pattern = (
        r"(?P<alias>[A-Za-zΑ-Ωα-ω0-9]"
        r"[A-Za-zΑ-Ωα-ω0-9_.-]{1,39})"
    )
    candidates = (
        _target_candidates(anchor, targets)
        if candidates is None
        else candidates
    )
    if len(candidates) != 1:
        return relations
    for target in candidates:
        for label in _target_labels(target):
            target_pattern = _term_pattern(label)
            patterns = (
                alias_pattern
                + r"\s+(?:is|was|represents|denotes)\s+(?:(?:the|a|an)\s+)?"
                + descriptor_pattern
                + r"\s+(?:of|from)\s+(?:the\s+)?"
                + target_pattern,
                alias_pattern
                + r"\s*,\s*(?:the\s+)?"
                + descriptor_pattern
                + r"\s+(?:of|from)\s+(?:the\s+)?"
                + target_pattern,
                descriptor_pattern
                + r"\s+(?:of|from)\s+(?:the\s+)?"
                + target_pattern
                + r"\s*(?:,|\(|called|designated)\s*"
                + alias_pattern,
                target_pattern
                + r"(?:'s)?\s+"
                + descriptor_pattern
                + r"\s*(?:,|\(|called|designated)\s*"
                + alias_pattern,
            )
            for pattern in patterns:
                match = re.search(pattern, anchor, re.I)
                if not match:
                    continue
                alias = _clean_alias(match.group("alias"))
                descriptor = _normalized(match.group("descriptor"))
                descriptor = re.sub(r"^(?:n|c)[- ]terminal ", "", descriptor)
                if alias and descriptor in CONSTRUCT_RELATIONS:
                    node_type, relation_type = CONSTRUCT_RELATIONS[descriptor]
                    relations.append((alias, node_type, relation_type, target))
    for target in candidates:
        for label in _target_labels(target):
            match = re.search(
                r"(?P<alias>[A-Za-zΑ-Ωα-ω0-9][A-Za-zΑ-Ωα-ω0-9_.-]{1,39})"
                r"\s+(?:is|was)\s+(?:proteolytically\s+)?cleaved\s+from\s+"
                + _term_pattern(label),
                anchor,
                re.I,
            )
            if match:
                relations.append(
                    (
                        match.group("alias"),
                        "exact_processed_product",
                        "processed_product_of",
                        target,
                    )
                )
    return relations


def _target_identity(target):
    return {
        key: target[key]
        for key in (
            "target_id",
            "doi",
            "group_id",
            "group_ids",
            "uniprot_accession",
            "target_names",
            "genes",
            "species",
            "pdb_ids",
            "entry_titles",
            "canonical_name_variants",
            "entry_title_aliases",
            "alias_audit",
        )
    }


def build_identity_graph(
    manifest_rows,
    passages,
    doi=None,
    target_group_id=None,
):
    """Build a deterministic, one-hop identity graph for a DOI.

    When ``target_group_id`` is supplied, canonical siblings in the same paper
    are typed as competing proteins.  No edge is inferred from co-occurrence,
    prefixes, slash groups, or another edge.
    """
    targets = collect_doi_targets(manifest_rows, doi=doi)
    if not targets:
        raise ValueError("no canonical targets found for DOI")
    doi = targets[0]["doi"]
    focus = None
    if target_group_id:
        matches = [
            target
            for target in targets
            if target_group_id in target["group_ids"]
            or target_group_id == target["target_id"]
            or str(target_group_id).upper() == target["uniprot_accession"]
        ]
        if len(matches) != 1:
            raise ValueError("target_group_id does not identify exactly one DOI target")
        focus = matches[0]
    elif len(targets) == 1:
        focus = targets[0]

    nodes = {}
    edges = {}
    protein_node_ids = {}

    def add_node(node_type, label, extra=None, identity_key=None):
        key = identity_key if identity_key is not None else [doi, node_type, _normalized(label)]
        node_id = _stable_id("node", key)
        node = {
            "node_id": node_id,
            "node_type": node_type,
            "label": str(label),
            "normalized_label": _normalized(label),
        }
        if extra:
            node.update(extra)
        nodes.setdefault(node_id, node)
        return node_id

    def add_edge(
        source_node_id,
        target,
        relation_type,
        source_id,
        anchor_quote,
        target_node_id=None,
        provenance=None,
    ):
        target_node_id = target_node_id or protein_node_ids[target["target_id"]]
        edge = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "relation_type": relation_type,
            "source_id": str(source_id),
            "anchor_quote": str(anchor_quote),
            "target_identity": _target_identity(target),
            "ambiguous": False,
            "candidate_target_ids": [target["target_id"]],
        }
        if provenance:
            edge["provenance"] = dict(provenance)
        edge["edge_id"] = _stable_id(
            "edge",
            [
                source_node_id,
                target_node_id,
                relation_type,
                source_id,
                anchor_quote,
                target_node_id,
                provenance or {},
            ],
        )
        edges.setdefault(edge["edge_id"], edge)

    for target in targets:
        identity_labels = _target_labels(target)
        all_identity_candidates = (
            _strings(target.get("uniprot_accession"))
            + list(target.get("target_names") or [])
            + list(target.get("genes") or [])
            + list(target.get("canonical_name_variants") or [])
            + list(target.get("entry_title_aliases") or [])
            + list(target.get("_paper_aliases") or [])
        )
        _, identity_label_audit = bounded_match_labels(
            all_identity_candidates, MAX_IDENTITY_LABELS_PER_TARGET
        )
        identity_label_audit["returned_label_count"] = len(identity_labels)
        identity_label_audit["complete"] = not identity_label_audit["truncated"]
        target.setdefault("alias_audit", {})[
            "identity_labels"
        ] = identity_label_audit
        protein_type = (
            "competing_protein"
            if focus is not None and target["target_id"] != focus["target_id"]
            else "canonical_protein"
        )
        protein_label = (
            target["target_names"][0]
            if target["target_names"]
            else target["uniprot_accession"]
        )
        protein_node_id = add_node(
            protein_type,
            protein_label,
            {
                "target_identity": _target_identity(target),
                "identity_labels": identity_labels,
            },
            identity_key=[doi, "protein", target["target_id"], protein_type],
        )
        protein_node_ids[target["target_id"]] = protein_node_id
        manifest_source = "manifest:%s" % (
            target["group_id"] or target["target_id"]
        )
        for accession in _strings(target["uniprot_accession"]):
            child = add_node(
                "canonical_accession",
                accession,
                {"target_id": target["target_id"]},
                [doi, "accession", target["target_id"], _normalized(accession)],
            )
            add_edge(child, target, "accession_of", manifest_source, accession)
        for gene in target["genes"]:
            child = add_node(
                "canonical_gene",
                gene,
                {"target_id": target["target_id"]},
                [doi, "gene", target["target_id"], _normalized(gene)],
            )
            add_edge(child, target, "gene_of", manifest_source, gene)
        for species in target["species"]:
            child = add_node(
                "canonical_species",
                species,
                {"target_id": target["target_id"]},
                [doi, "species", target["target_id"], _normalized(species)],
            )
            add_edge(child, target, "species_of", manifest_source, species)
        for variant in target.get("canonical_name_variants") or []:
            source_name = next(
                (
                    name
                    for name in target["target_names"]
                    if variant in _canonical_name_variants(name)
                ),
                protein_label,
            )
            child = add_node(
                "canonical_name_variant",
                variant,
                {"target_id": target["target_id"]},
                [doi, "name_variant", target["target_id"], _normalized(variant)],
            )
            add_edge(
                child,
                target,
                "canonical_name_variant_of",
                manifest_source,
                source_name,
                provenance={
                    "kind": "manifest_canonical_name_variant",
                    "source_name": source_name,
                    "derivation_rule": "conservative_canonical_name_variant",
                },
            )
        for title in target.get("entry_titles") or []:
            if title in (target.get("entry_title_aliases") or []):
                child = add_node(
                    "entry_title_alias",
                    title,
                    {"target_id": target["target_id"], "full_entry_title": title},
                    [doi, "entry_title", target["target_id"], _normalized(title)],
                )
                add_edge(
                    child,
                    target,
                    "entry_title_of",
                    manifest_source,
                    title,
                    provenance={
                        "kind": "safe_rcsb_entry_title",
                        "full_entry_title": title,
                        "extraction_rule": "full_safe_entry_title",
                    },
                )
            for extracted, rule in _entry_title_extractions(title):
                if extracted not in (target.get("entry_title_aliases") or []):
                    continue
                child = add_node(
                    "entry_title_alias",
                    extracted,
                    {"target_id": target["target_id"], "full_entry_title": title},
                    [doi, "entry_title_name", target["target_id"], _normalized(extracted)],
                )
                add_edge(
                    child,
                    target,
                    "entry_title_name_of",
                    manifest_source,
                    title,
                    provenance={
                        "kind": "safe_rcsb_entry_title_extraction",
                        "full_entry_title": title,
                        "extraction_rule": rule,
                    },
                )
        for typed in target.get("entry_title_constructs") or []:
            label = typed["label"]
            child = add_node(
                typed["node_type"],
                label,
                {
                    "target_id": target["target_id"],
                    "full_entry_title": typed["full_entry_title"],
                },
                [
                    doi,
                    "entry_title_construct",
                    target["target_id"],
                    typed["relation_type"],
                    _normalized(label),
                ],
            )
            add_edge(
                child,
                target,
                typed["relation_type"],
                manifest_source,
                typed["full_entry_title"],
                provenance={
                    "kind": "safe_rcsb_entry_title_construct",
                    "full_entry_title": typed["full_entry_title"],
                    "extraction_rule": typed["extraction_rule"],
                },
            )
        for entity in target["entities"]:
            pdb_id = str(entity.get("pdb_id") or "")
            entity_id = str(entity.get("entity_id") or "")
            label = "%s_%s" % (pdb_id, entity_id)
            child = add_node(
                "pdb_entity",
                label,
                {"target_id": target["target_id"], "entity": entity},
                [doi, "pdb_entity", target["target_id"], pdb_id, entity_id],
            )
            add_edge(child, target, "pdb_entity_of", manifest_source, label)

    records = _passage_records(passages, doi)
    for record in records:
        for anchor in _sentences(record["text"]):
            candidates = _target_candidates(anchor, targets)
            relations = (
                _parenthetical_relations(anchor, targets)
                + _pdb_parenthetical_relations(anchor, targets)
                + _marked_alias_relations(anchor, targets, candidates)
                + _descriptor_relations(anchor, targets, candidates)
            )
            for alias, node_type, relation_type, target in relations:
                alias = _clean_alias(alias)
                if not alias:
                    continue
                source_node_id = add_node(node_type, alias)
                add_edge(
                    source_node_id,
                    target,
                    relation_type,
                    record["source_id"],
                    anchor,
                )

    # A paper protein alias may anchor one additional, explicitly typed hop to
    # a construct/domain/variant.  No alias-to-alias closure is performed.
    node_by_id = {node_id: node for node_id, node in nodes.items()}
    alias_bindings = defaultdict(list)
    for edge in edges.values():
        if edge["relation_type"] != "alias_of":
            continue
        node = node_by_id.get(edge["source_node_id"]) or {}
        alias_bindings[_identity_key(node.get("label"))].append((node, edge))
    augmented_targets = []
    unique_alias_edges = {}
    for target in targets:
        augmented = dict(target)
        aliases = []
        for bindings in alias_bindings.values():
            target_ids = {
                binding[1]["target_identity"]["target_id"] for binding in bindings
            }
            if target_ids != {target["target_id"]}:
                continue
            for node, edge in bindings:
                aliases.append(node["label"])
                unique_alias_edges[
                    (target["target_id"], _identity_key(node["label"]))
                ] = (node, edge)
        augmented["_paper_aliases"] = aliases
        augmented_targets.append(augmented)
    for record in records:
        for anchor in _sentences(record["text"]):
            candidates = _target_candidates(anchor, augmented_targets)
            relations = (
                _parenthetical_relations(anchor, augmented_targets)
                + _descriptor_relations(
                    anchor, augmented_targets, candidates
                )
            )
            for alias, node_type, relation_type, target in relations:
                if relation_type not in _CONSTRUCT_RELATION_TYPES:
                    continue
                paper_support = [
                    unique_alias_edges[(target["target_id"], _identity_key(label))]
                    for label in target.get("_paper_aliases") or []
                    if _contains_term(anchor, label)
                    and (target["target_id"], _identity_key(label))
                    in unique_alias_edges
                ]
                if not paper_support:
                    continue
                support_node, support_edge = paper_support[0]
                source_node_id = add_node(node_type, alias)
                add_edge(
                    source_node_id,
                    target,
                    relation_type,
                    record["source_id"],
                    anchor,
                    target_node_id=support_node["node_id"],
                    provenance={
                        "kind": "sanctioned_paper_alias_construct_path",
                        "path_depth": 2,
                        "supporting_alias_edge_id": support_edge["edge_id"],
                    },
                )

    # Sanction one exact two-hop proteolytic mass chain.  This requires all of:
    # a canonical mass-labelled wt construct, a direct self-processing sentence,
    # and a separate exact parenthetical naming of the resulting mass construct.
    node_by_id = {node_id: node for node_id, node in nodes.items()}
    mass_support = []
    for edge in list(edges.values()):
        if edge.get("relation_type") not in _CONSTRUCT_RELATION_TYPES:
            continue
        node = node_by_id.get(edge.get("source_node_id")) or {}
        match = re.fullmatch(
            r"wt[\s_-]+(?P<mass>\d+(?:\.\d+)?)\s*PR",
            str(node.get("label") or ""),
            re.I,
        )
        if match and edge.get("target_node_id") == protein_node_ids.get(
            (edge.get("target_identity") or {}).get("target_id")
        ):
            mass_support.append((float(match.group("mass")), node, edge))
    for parent_mass, support_node, support_edge in mass_support:
        for process_record in records:
            for process_anchor in _sentences(process_record["text"]):
                process = re.search(
                    _term_pattern(support_node["label"])
                    + r".{0,180}\b(?:self[- ]process\w*|cleav\w*\s+off)\b"
                    + r".{0,180}\b(?:yield\w*|result\w*)\s+in\b"
                    + r".{0,80}\b(?P<mass>\d+(?:\.\d+)?)\s*kDa\b"
                    + r".{0,50}\b(?:form\s+)?protease\b",
                    process_anchor,
                    re.I,
                )
                if not process:
                    continue
                product_mass = float(process.group("mass"))
                if product_mass >= parent_mass:
                    continue
                product_label = "wt %s PR" % process.group("mass")
                alias_support = []
                for alias_record in records:
                    for alias_anchor in _sentences(alias_record["text"]):
                        if not re.search(
                            r"\b"
                            + re.escape(process.group("mass"))
                            + r"\s*kDa\b.{0,60}\bprotease\b.{0,20}\("
                            + re.escape(product_label)
                            + r"\)",
                            alias_anchor,
                            re.I,
                        ):
                            continue
                        alias_support.append((alias_record, alias_anchor))
                if len(alias_support) != 1:
                    continue
                alias_record, alias_anchor = alias_support[0]
                target_id = support_edge["target_identity"]["target_id"]
                target = next(
                    candidate
                    for candidate in targets
                    if candidate["target_id"] == target_id
                )
                source_node_id = add_node(
                    "exact_processed_product",
                    product_label,
                )
                add_edge(
                    source_node_id,
                    target,
                    "processed_product_of",
                    alias_record["source_id"],
                    alias_anchor,
                    target_node_id=support_node["node_id"],
                    provenance={
                        "kind": "sanctioned_proteolytic_mass_construct_path",
                        "path_depth": 2,
                        "supporting_alias_edge_id": support_edge["edge_id"],
                        "required_evidence": [
                            {
                                "source_id": process_record["source_id"],
                                "anchor_quote": process_anchor,
                                "role": "direct_self_processing",
                            },
                            {
                                "source_id": alias_record["source_id"],
                                "anchor_quote": alias_anchor,
                                "role": "exact_product_mass_alias",
                            },
                        ],
                    },
                )

    candidate_targets = defaultdict(set)
    for edge in edges.values():
        if edge["relation_type"] in {
            "alias_of",
            "construct_of",
            "fragment_of",
            "truncation_of",
            "domain_of",
            "variant_of",
            "processed_product_of",
        }:
            candidate_targets[edge["source_node_id"]].add(
                edge["target_identity"]["target_id"]
            )
    for node_id, candidates in candidate_targets.items():
        ordered = sorted(candidates)
        ambiguous = len(ordered) > 1
        nodes[node_id]["ambiguous"] = ambiguous
        nodes[node_id]["candidate_target_ids"] = ordered
        for edge in edges.values():
            if edge["source_node_id"] == node_id:
                edge["ambiguous"] = ambiguous
                edge["candidate_target_ids"] = ordered

    passage_values = {}
    for record in records:
        # A source ID denotes one contiguous quote domain. Legacy duplicate IDs
        # are resolved deterministically to the first record, never joined.
        passage_values.setdefault(record["source_id"], record["text"])
    passage_index = {
        source_id: text for source_id, text in sorted(passage_values.items())
    }
    return {
        "schema_version": "doi_identity_v1",
        "doi": doi,
        "focus_target_id": focus["target_id"] if focus else "",
        "targets": [_target_identity(target) for target in targets],
        "nodes": sorted(nodes.values(), key=lambda node: node["node_id"]),
        "edges": sorted(edges.values(), key=lambda edge: edge["edge_id"]),
        "source_text": {key: passage_index[key] for key in sorted(passage_index)},
        "policy": dict(IDENTITY_POLICY),
    }


def _claim_evidence(claim):
    evidence = claim.get("evidence") or []
    if isinstance(evidence, dict):
        evidence = [evidence]
    if not evidence and (claim.get("quote") or claim.get("source_id")):
        evidence = [
            {
                "quote": claim.get("quote") or "",
                "source_id": claim.get("source_id") or "",
            }
        ]
    records = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        record = {
            "quote": str(item.get("quote") or ""),
            "source_id": str(item.get("source_id") or ""),
        }
        for key in (
            "reported_target_mention",
            "reported_species",
            "identity_anchor_ids",
        ):
            if key in item:
                raw_value = item.get(key)
                if key != "identity_anchor_ids":
                    record[key] = str(raw_value or "")
                else:
                    record[key] = [
                        str(value) for value in raw_value or [] if str(value)
                    ]
        records.append(record)
    return records


def _resolve_focus(graph, target_group_id):
    selector = target_group_id or graph.get("focus_target_id")
    candidates = graph.get("targets") or []
    if selector:
        matches = [
            target
            for target in candidates
            if selector == target["target_id"]
            or selector in target.get("group_ids", [])
            or str(selector).upper() == target.get("uniprot_accession")
        ]
        if len(matches) == 1:
            return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("target_group_id is required for a multi-target DOI graph")


def _claim_mentions(claim, evidence, nodes):
    explicit = (
        claim.get("reported_target_mention")
        or claim.get("mention")
        or claim.get("target_mention")
        or claim.get("protein")
        or ""
    )
    if explicit:
        return [str(explicit)]
    if any("reported_target_mention" in item for item in evidence):
        reported = sorted(
            {
                str(item.get("reported_target_mention") or "").strip()
                for item in evidence
                if str(item.get("reported_target_mention") or "").strip()
            }
        )
        # In v3 an explicitly empty report means unresolved. Do not replace it
        # with a model-adjacent guess inferred from another claim field.
        if reported or all(
            not str(item.get("reported_target_mention") or "").strip()
            for item in evidence
        ):
            return reported
    construct = str(claim.get("construct") or "").strip()
    generic_constructs = {
        "full length",
        "full-length",
        "full length protein",
        "full-length protein",
        "wild type",
        "wild-type",
        "wt",
    }
    node_labels = {
        _normalized(label)
        for node in nodes
        if node.get("node_type") in _PROTEIN_NODE_TYPES
        for label in node.get("identity_labels") or [node.get("label", "")]
    }
    if construct and _normalized(construct) not in generic_constructs and _normalized(
        construct
    ) in node_labels:
        return [construct]
    quoted = " ".join(item["quote"] for item in evidence)
    labels = []
    for node in nodes:
        if node["node_type"] not in _PROTEIN_NODE_TYPES:
            continue
        for label in node.get("identity_labels") or [node["label"]]:
            if _contains_term(quoted, label):
                labels.append(label)
    return sorted(set(labels), key=lambda value: (-len(value), _normalized(value), value))


def _node_target_relations(node, outgoing):
    relations = defaultdict(set)
    if node.get("node_type") in {"canonical_protein", "competing_protein"}:
        target_id = (node.get("target_identity") or {}).get("target_id")
        if target_id:
            relations[target_id].add("exact_protein")
    for edge in outgoing.get(node.get("node_id"), []):
        target_id = (edge.get("target_identity") or {}).get("target_id")
        if target_id:
            relations[target_id].add(edge.get("relation_type", ""))
    return relations


def _resolution(
    status,
    focus,
    checks,
    reason,
    mentions=None,
    matched_ids=None,
    candidate_ids=None,
    ambiguity="none",
    anchor_path=None,
    **extra
):
    result = {
        "status": status,
        "target_identity": focus,
        "matched_mentions": list(mentions or []),
        "matched_node_ids": sorted(matched_ids or []),
        "candidate_target_ids": sorted(candidate_ids or []),
        "ambiguity": ambiguity,
        "evidence_checks": checks,
        "anchor_path": list(anchor_path or []),
        "reason": reason,
    }
    result.update(extra)
    return result


def _anchor_quote_exact(identity_graph, source_id, quote):
    """Verify a graph anchor in any selected window from its base source."""
    if not source_id or not quote:
        return False
    sources = identity_graph.get("source_text") or {}
    window_index = identity_graph.get("source_windows_by_base") or {}
    candidate_ids = [source_id] + list(window_index.get(source_id) or [])
    return any(
        candidate_id in sources and quote in sources[candidate_id]
        for candidate_id in dict.fromkeys(candidate_ids)
    )


def _absent_partner_contrast(text, subject_labels, partner_labels):
    """Recognize only a typed subject contrasted with an explicitly absent partner."""
    raw = str(text or "")
    for subject in subject_labels:
        for partner in partner_labels:
            pattern = (
                _term_pattern(subject)
                + r"\s+can\s+\w*oligomer\w*(?:\s*in)?\s+the\s+absence\s+of\s+"
                + _term_pattern(partner)
            )
            if re.search(pattern, raw, re.I):
                return True
    return False


def _species_supported_by_safe_entry_title(focus, observed_species):
    """Accept an alternate organism name only when the exact-accession title states it."""
    titles = focus.get("entry_titles") or []
    observed = _strings(observed_species)
    if not titles or not observed:
        return False
    return all(
        any(
            _contains_term(
                title,
                re.sub(r"\s*\([^()]*\)\s*$", "", species).strip(),
            )
            for title in titles
        )
        for species in observed
    )


def resolve_claim_identity(claim, identity_graph, target_group_id=None):
    """Resolve one JSON claim against direct graph links, abstaining on ambiguity."""
    if not isinstance(claim, dict) or not isinstance(identity_graph, dict):
        raise TypeError("claim and identity_graph must be JSON objects")
    focus = _resolve_focus(identity_graph, target_group_id)
    evidence = _claim_evidence(claim)
    sources = identity_graph.get("source_text") or {}
    checks = []
    for item in evidence:
        exact = bool(
            item["quote"]
            and item["source_id"] in sources
            and item["quote"] in sources[item["source_id"]]
        )
        checks.append(dict(item, exact_source_quote=exact))
    exact_checks = [item for item in checks if item["exact_source_quote"]]
    if not exact_checks:
        return _resolution(
            "unlinked",
            focus,
            checks,
            "missing_or_nonexact_claim_evidence",
        )

    nodes = identity_graph.get("nodes") or []
    outgoing = defaultdict(list)
    edge_by_id = {}
    for edge in identity_graph.get("edges") or []:
        outgoing[edge["source_node_id"]].append(edge)
        edge_by_id[edge.get("edge_id", "")] = edge
    mentions = _claim_mentions(claim, exact_checks, nodes)
    quoted = " ".join(item["quote"] for item in exact_checks)
    contextual_construct_source_ids = []
    construct = str(claim.get("construct") or "").strip()
    construct_nodes = [
        node
        for node in nodes
        if node.get("node_type") in _CONSTRUCT_NODE_TYPES
        and _normalized(node.get("label")) == _normalized(construct)
        and any(
            edge.get("relation_type") in _CONSTRUCT_RELATION_TYPES
            for edge in outgoing.get(node.get("node_id"), [])
        )
    ]
    mentions_match_graph = any(
        _normalized(mention)
        == _normalized(label)
        for mention in mentions
        for node in nodes
        if node.get("node_type") in _PROTEIN_NODE_TYPES
        for label in node.get("identity_labels") or [node.get("label", "")]
    )
    mention_is_construct = bool(
        construct
        and any(
            _normalized(mention) == _normalized(construct) for mention in mentions
        )
    )
    if construct_nodes and (not mentions_match_graph or mention_is_construct):
        contextual_construct_source_ids = sorted(
            {
                item["source_id"]
                for item in exact_checks
                if item["source_id"] in sources
                and _contains_term(sources[item["source_id"]], construct)
            }
        )
        if contextual_construct_source_ids:
            mentions = [construct]
    has_reported_identity_fields = any(
        "reported_target_mention" in item for item in exact_checks
    )
    if has_reported_identity_fields and not mentions:
        return _resolution(
            "unlinked",
            focus,
            checks,
            "reported_target_mention_unresolved",
        )
    if (
        mentions
        and not contextual_construct_source_ids
        and not any(_contains_term(quoted, mention) for mention in mentions)
    ):
        return _resolution(
            "unlinked",
            focus,
            checks,
            "claim_mention_absent_from_exact_evidence",
            mentions=mentions,
        )

    focus_id = focus["target_id"]
    competing_ids = {
        target["target_id"]
        for target in identity_graph.get("targets") or []
        if target.get("target_id") != focus_id
    }
    focus_labels = {
        _normalized(label)
        for node in nodes
        if focus_id in _node_target_relations(node, outgoing)
        for label in node.get("identity_labels") or [node.get("label", "")]
        if label
    }
    observed_anchor_species = _anchor_species_mentions(quoted, focus)
    if observed_anchor_species and compare_species(
        focus.get("species"), observed_anchor_species
    )["status"] == "mismatch":
        return _resolution(
            "mismatch",
            focus,
            checks,
            "explicit_species_mismatch_in_exact_evidence",
            mentions=mentions,
            explicit_competing_species=observed_anchor_species,
        )
    explicit_competitor_nodes = []
    for node in nodes:
        if node.get("node_type") not in _PROTEIN_NODE_TYPES:
            continue
        labels = node.get("identity_labels") or [node.get("label", "")]
        present_labels = [
            label
            for label in labels
            if label
            and _contains_term(quoted, label)
            and _normalized(label) not in focus_labels
        ]
        if not present_labels:
            continue
        target_relations = _node_target_relations(node, outgoing)
        target_ids = set(target_relations)
        if target_ids and target_ids <= competing_ids:
            explicit_competitor_nodes.append(node)

    competing_species = []
    for target in identity_graph.get("targets") or []:
        if target.get("target_id") == focus_id:
            continue
        for species in target.get("species") or []:
            if _contains_term(quoted, species):
                competing_species.append(species)
    absent_competitor_nodes = [
        node
        for node in explicit_competitor_nodes
        if _absent_partner_contrast(
            quoted,
            mentions,
            node.get("identity_labels") or [node.get("label", "")],
        )
    ]
    explicit_competitor_nodes = [
        node for node in explicit_competitor_nodes if node not in absent_competitor_nodes
    ]
    if explicit_competitor_nodes or competing_species:
        target_ids = sorted(
            {
                target_id
                for node in explicit_competitor_nodes
                for target_id in _node_target_relations(node, outgoing)
            }
            | {
                target["target_id"]
                for target in identity_graph.get("targets") or []
                if any(
                    _normalized(species) == _normalized(observed)
                    for species in target.get("species") or []
                    for observed in competing_species
                )
            }
        )
        return _resolution(
            "mismatch",
            focus,
            checks,
            "explicit_competing_target_or_species_in_exact_evidence",
            mentions=mentions,
            matched_ids=[node["node_id"] for node in explicit_competitor_nodes],
            candidate_ids=target_ids,
            explicit_competing_mentions=sorted(
                {
                    node.get("label", "")
                    for node in explicit_competitor_nodes
                    if node.get("label")
                }
            ),
            explicit_competing_species=sorted(set(competing_species)),
        )

    matched_nodes = [
        node
        for node in nodes
        if node["node_type"] in _PROTEIN_NODE_TYPES
        and any(
            _normalized(mention) == _normalized(label)
            for mention in mentions
            for label in node.get("identity_labels") or [node["label"]]
        )
    ]

    candidates = defaultdict(set)
    relation_by_target = defaultdict(set)
    matched_ids = set()
    anchor_path = []
    for node in matched_nodes:
        matched_ids.add(node["node_id"])
        if node["node_type"] in {"canonical_protein", "competing_protein"}:
            identity = node.get("target_identity") or {}
            if identity.get("target_id"):
                candidates[node["node_id"]].add(identity["target_id"])
                relation_by_target[identity["target_id"]].add("exact_protein")
                anchor_path.append(
                    {
                        "mention": node["label"],
                        "node_id": node["node_id"],
                        "node_type": node["node_type"],
                        "relation_type": "exact_protein",
                        "edge_id": "",
                        "target_id": identity["target_id"],
                        "claim_source_ids": sorted(
                            {
                                item["source_id"]
                                for item in exact_checks
                                if _contains_term(item["quote"], node["label"])
                            }
                        ),
                        "anchor_source_id": "",
                        "anchor_quote": "",
                    }
                )
        for edge in outgoing.get(node["node_id"], []):
            source_id = edge.get("source_id", "")
            anchor_quote = edge.get("anchor_quote", "")
            manifest_edge = source_id.startswith("manifest:")
            if not manifest_edge and not _anchor_quote_exact(
                identity_graph, source_id, anchor_quote
            ):
                continue
            support_id = (edge.get("provenance") or {}).get(
                "supporting_alias_edge_id"
            )
            support = edge_by_id.get(support_id)
            support_valid = not support_id
            if support:
                support_source = support.get("source_id", "")
                support_quote = support.get("anchor_quote", "")
                support_valid = support_source.startswith("manifest:") or (
                    _anchor_quote_exact(
                        identity_graph, support_source, support_quote
                    )
                )
            if not support_valid:
                continue
            required_evidence = (edge.get("provenance") or {}).get(
                "required_evidence"
            ) or []
            if not all(
                _anchor_quote_exact(
                    identity_graph,
                    str(required.get("source_id") or ""),
                    str(required.get("anchor_quote") or ""),
                )
                for required in required_evidence
            ):
                continue
            target_id = edge["target_identity"]["target_id"]
            candidates[node["node_id"]].add(target_id)
            relation_by_target[target_id].add(edge["relation_type"])
            anchor_path.append(
                {
                    "mention": node["label"],
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "relation_type": edge["relation_type"],
                    "edge_id": edge.get("edge_id", ""),
                    "target_id": target_id,
                    "claim_source_ids": sorted(
                        {
                            item["source_id"]
                            for item in exact_checks
                            if any(
                                _contains_term(item["quote"], label)
                                for label in node.get("identity_labels")
                                or [node["label"]]
                            )
                        }
                    ),
                    "anchor_source_id": source_id,
                    "anchor_quote": anchor_quote,
                }
            )
            if support:
                support_source = support.get("source_id", "")
                support_quote = support.get("anchor_quote", "")
                if support_valid:
                    support_node = next(
                        (
                            candidate
                            for candidate in nodes
                            if candidate.get("node_id")
                            == support.get("source_node_id")
                        ),
                        {},
                    )
                    anchor_path.append(
                        {
                            "mention": support_node.get("label", ""),
                            "node_id": support_node.get("node_id", ""),
                            "node_type": support_node.get("node_type", ""),
                            "relation_type": support.get("relation_type", ""),
                            "edge_id": support.get("edge_id", ""),
                            "target_id": target_id,
                            "claim_source_ids": [],
                            "anchor_source_id": support_source,
                            "anchor_quote": support_quote,
                        }
                    )

    all_targets = sorted({target for values in candidates.values() for target in values})
    if any(len(values) > 1 for values in candidates.values()) or len(all_targets) > 1:
        return _resolution(
            "ambiguous",
            focus,
            checks,
            "explicit_abstention_on_ambiguous_alias_or_mentions",
            mentions=mentions,
            matched_ids=matched_ids,
            candidate_ids=all_targets,
            ambiguity="multiple_canonical_targets",
            anchor_path=anchor_path,
        )
    if not all_targets:
        return _resolution(
            "unlinked",
            focus,
            checks,
            "no_exact_name_or_explicit_identity_edge",
            mentions=mentions,
        )
    resolved_target = all_targets[0]
    if resolved_target != focus_id:
        return _resolution(
            "mismatch",
            focus,
            checks,
            "mention_resolves_to_competing_protein",
            mentions=mentions,
            matched_ids=matched_ids,
            candidate_ids=all_targets,
            anchor_path=anchor_path,
        )

    reported_species = sorted(
        {
            str(item.get("reported_species") or "").strip()
            for item in exact_checks
            if str(item.get("reported_species") or "").strip()
        }
    )
    observed_species = (
        claim.get("reported_species")
        or claim.get("species")
        or claim.get("organism")
        or reported_species
    )
    if (
        has_reported_identity_fields
        and focus.get("species")
        and not reported_species
    ):
        return _resolution(
            "unlinked",
            focus,
            checks,
            "reported_species_unresolved",
            mentions=mentions,
            matched_ids=matched_ids,
            candidate_ids=all_targets,
            anchor_path=anchor_path,
        )
    if observed_species:
        species_check = compare_species(focus.get("species"), observed_species)
        if (
            species_check["status"] == "mismatch"
            and not _species_supported_by_safe_entry_title(focus, observed_species)
        ):
            return _resolution(
                "mismatch",
                focus,
                checks,
                "explicit_species_mismatch",
                mentions=mentions,
                matched_ids=matched_ids,
                candidate_ids=all_targets,
                anchor_path=anchor_path,
                species_comparison=species_check,
            )
        if species_check["status"] == "mismatch":
            species_check = dict(
                species_check,
                status="exact",
                match_basis="safe_exact_accession_entry_title",
            )

    relations = relation_by_target[focus_id]
    construct_relations = relations & {
        "construct_of",
        "fragment_of",
        "truncation_of",
        "domain_of",
        "variant_of",
        "processed_product_of",
    }
    full_length_claim = _normalized(claim.get("construct")) in {
        "full length",
        "full-length",
        "full length protein",
        "full-length protein",
    } or bool(claim.get("requires_full_length"))
    construct_scoped = bool(
        construct_relations
        and not full_length_claim
        and _normalized(claim.get("construct"))
        and any(
            _normalized(claim.get("construct")) == _normalized(mention)
            for mention in mentions
        )
    )
    if construct_relations and full_length_claim:
        status = "mismatch"
        reason = "full_length_claim_resolves_only_to_construct"
    elif construct_relations and (
        construct_scoped
        or ("exact_protein" not in relations and "alias_of" not in relations)
    ):
        status = "construct_of_exact"
        reason = (
            "explicit_construct_relation_takes_alias_precedence"
            if construct_scoped
            else "explicit_one_hop_construct_relation"
        )
    else:
        status = "exact"
        reason = "exact_canonical_identity_or_direct_alias"
    return _resolution(
        status,
        focus,
        checks,
        reason,
        mentions=mentions,
        matched_ids=matched_ids,
        candidate_ids=all_targets,
        anchor_path=anchor_path,
        construct_scoped_claim=construct_scoped,
        contextual_construct_source_ids=contextual_construct_source_ids,
    )


def ground_claim_identity(
    manifest_rows,
    passages,
    claim,
    doi=None,
    target_group_id=None,
):
    """Convenience API returning both graph metadata and claim resolution."""
    graph = build_identity_graph(
        manifest_rows,
        passages,
        doi=doi,
        target_group_id=target_group_id,
    )
    return {
        "identity_graph": graph,
        "resolution": resolve_claim_identity(
            claim,
            graph,
            target_group_id=target_group_id,
        ),
    }


__all__ = [
    "CLAIM_STATUSES",
    "IDENTITY_POLICY",
    "NODE_TYPES",
    "build_identity_graph",
    "collect_doi_targets",
    "compare_organisms",
    "compare_species",
    "ground_claim_identity",
    "identity_graph_hashes",
    "resolve_claim_identity",
]
