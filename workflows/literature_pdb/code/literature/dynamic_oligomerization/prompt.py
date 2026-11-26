"""Versioned prompt rendering from canonical manifest groups only."""

import json
import re

from .aliases import (
    MAX_GENE_ALIASES_PER_ENTITY,
    MAX_RETRIEVAL_TERMS,
    bounded_match_labels,
    normalize_gene_aliases,
)
from .schema import SCHEMA_VERSION


PROMPT_VERSION = "dynamic_oligomerization_protein_prompt_v5"

SYSTEM_PROMPT = """You curate dynamic homo-oligomerization evidence.

The extraction unit is exactly one canonical paper-target group. Extract only
claims about the explicitly named TARGET. Never transfer evidence from another
protein, homologue, partner, or construct.

Scientific extraction rules:
1. Behaviors are non-exclusive. A protein may show coexistence, reversible
   equilibrium, and a condition-linked population shift in the same experiment.
2. Reporting one state under one condition does not prove global fixedness.
3. Distinguish discrete free homo-oligomers from heteromeric complexes,
   nucleic-acid-, membrane-, or partner-templated assembly, filaments,
   aggregation, and phase separation.
4. A conformational change is not an oligomerization change. Ligand binding at
   an interface is not evidence of a population shift unless state or population
   is directly compared. Association/dissociation means association/dissociation
   of TARGET protein subunits with each other. A 1:1 target:ligand or
   target:partner complex contains one target subunit, not two.
5. Direct SEC, SEC-MALS, or AUC measurement that verifies an oligomeric state is
   valid state evidence. Purification SEC, including merely pooling a fraction
   described as monomeric, is purification-only and is not state evidence.
   Buffer composition, concentration, pH, or metal conditions are not dynamic
   evidence unless oligomeric states or populations are measured or compared.
6. Record mutant, truncation, domain, tag, ligand-bound, and full-length
   constructs exactly. Do not transfer behavior between constructs.
7. Evidence origin is claim-specific. Prior citations may be recorded as
   prior_citation_about_target but do not count as direct evidence from this
   paper. Evidence about other or related proteins is never target evidence.
8. Every accepted claim requires a verbatim quote from one supplied original
   paper source block. source_id, section, and page must match its header.
9. Quantitative value_text must preserve the paper's wording. Fill parsed
   numeric fields only when unambiguous; otherwise use an empty string.
10. Therapeutic links require an explicit oligomerization-linked intervention
    or mechanistic opportunity. Generic drug-target relevance is insufficient.
11. Put tempting but invalid oligomerization claims in excluded_claims with the
    applicable reason.
12. Be conservative. Empty arrays are preferable to unsupported claims.
13. Use reversible_equilibrium only when an experiment or the authors
    explicitly establish equilibrium or reversible interconversion. Suspected
    concentration effects, coexistence, or crystal-versus-solution differences
    do not by themselves establish a reversible equilibrium.
14. Dynamic events must change or redistribute the number of TARGET protein
    subunits. Record target-subunit counts explicitly. A monomer conformational
    change belongs in excluded_claims unless tied to a genuine homo-oligomeric
    event.
15. Copy quotes exactly from one supplied source block. Do not paraphrase,
    silently repair text, join blocks, or insert ellipses.
16. Do not collapse a multi-step assembly pathway into its endpoints. Extract
    every explicitly stated intermediate target stoichiometry as a state and an
    event step. All steps in an ordered pathway share one nonempty pathway_id
    and consecutive one-based step_index values. Standalone events use
    pathway_id="" and step_index=0. For dimer -> transient monomer -> octamer,
    report dimer -> monomer and monomer -> octamer; do not invent a direct
    dimer -> octamer shortcut.

Protein-level grouping and grounding rules:
17. Produce one protein-level extraction for this canonical DOI-target group,
    not one extraction per PDB. Associated PDB records are construct-grounding
    metadata only and do not require separate PDB labels.
18. Use the construct field on every observation and event to distinguish
    full-length protein, domains, truncations, mutants, tags, homologues, and
    ligand-bound preparations.
19. PDB identity, entry titles, entity descriptions, and biological
    assembly metadata are not substitutes for quoted evidence from the paper.
20. Summarize the target at group level while retaining construct-level
    observations and conflicts. Do not output or imply applicable-PDB mappings.
21. The word "subunit" in a PDB description does not establish a monomer.
    subunit_count means target-protein stoichiometry in the complete assembly.
    Use 1 only for explicit monomer evidence and 0 when stoichiometry is unknown.
22. One molecule in a crystal asymmetric unit is not evidence of a monomer.
    Two or more molecules in an asymmetric unit are not evidence of a biological
    oligomer unless the paper explicitly supports that assembly. Put unsupported
    asymmetric-unit counts in excluded_claims.
23. In papers studying several proteins, homologues, partners, or constructs,
    every accepted observation and event must concern the exact named TARGET.
24. Ground every state and event to the canonical target species, gene/accession,
    and the exact assayed construct or variant. A human protein and a nonhuman
    homologue are different targets: never transfer evidence between them.
    The target need not be redundantly species-named in every quote, but any
    explicit species, gene, accession, homologue, or construct identity must
    agree with TARGET METADATA.
25. Every accepted state quote must itself support the asserted oligomer name or
    target-subunit count. Every accepted event quote must support an
    oligomerization, association, dissociation, equilibrium, coexistence, or
    population change. Nearby target text and conformational change alone are
    insufficient.
26. If two variants, mutants, truncations, domains, tags, species constructs, or
    ligand-bound constructs have different responses, emit separate states and
    events with one exact construct per item. Never combine their responses into
    one event or use a compound construct label.
27. Unknown stoichiometry uses count 0. A 0 -> 0 event is permitted only when
    the exact-target quote explicitly reports changed self-association,
    multimerization, oligomerization, assembly, or dissociation. Do not use
    0 -> 0 for a conformational or structural shift.
28. Citation language such as "previously reported", "earlier studies", or
    "reported by X et al." is prior-citation evidence even if the surrounding
    paragraph discusses TARGET; classify it accordingly.
29. Preserve quantitative value_text and units exactly as printed, including
    suspicious concentration units. Never silently convert, repair, or rescale
    a PDF-extracted unit.
30. A source block marked identity_anchor contains an explicit paper-local
    alias or construct relation. Use that relation only for the named target
    and construct; a domain, fragment, processed product, or truncation must
    never be projected to the full-length protein.
31. Other canonical targets from the same DOI are competing identities, not
    aliases of TARGET. If a claim quote names one of them or its manifest
    species, do not treat the quote as TARGET evidence.
32. For every evidence object, copy reported_target_mention and
    reported_species literally from that evidence quote. Propose the exact
    identity_anchor_ids from IDENTITY CONTEXT that support the reported
    identity. These IDs are proposals only: deterministic verification, not
    the model's choice of ID, controls acceptance.
33. If target identity, species, or an identity anchor is unresolved, use an
    empty string or [] as required and abstain or place the candidate in
    excluded_claims. Never guess an alias, species, protein, or anchor ID.
34. Set construct_relation on every state and event. Explicitly distinguish
    full-length protein from exact variants, domains, truncations, fragments,
    processed products, and other constructs. Construct evidence does not
    establish the behavior of its full-length parent.
35. Competing proteins and species in IDENTITY CONTEXT are exclusion controls.
    Evidence for a related protein, homologue, or wrong species must not be
    attached to an accepted TARGET claim.
36. Source headers are binding. Never quote from a source block whose
    REQUIRES_ABSTENTION value is true; unresolved layout can make apparently
    contiguous text scientifically unauditable.
"""

_ENTITY_FIELDS = (
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


def concise_target_metadata(group):
    """Return allow-listed identity/construct metadata, excluding label fields."""
    target_accession = str(group["uniprot_accession"]).upper()
    entities = []
    for raw in group.get("entities") or []:
        entity = {field: raw.get(field) for field in _ENTITY_FIELDS}
        entity["pdb_id"] = str(entity.get("pdb_id") or "").upper()
        for field in ("chains", "source_organisms"):
            entity[field] = sorted({str(value) for value in entity.get(field) or []})
        genes, gene_audit = normalize_gene_aliases(
            entity.get("source_genes"),
            limit=MAX_GENE_ALIASES_PER_ENTITY,
            prior_audit=entity.get("source_gene_alias_audit"),
        )
        entity["source_genes"] = genes
        entity["source_gene_alias_audit"] = gene_audit
        entity["entry_distinct_uniprot_accessions"] = sorted(
            {
                str(value).strip().upper()
                for value in entity.get("entry_distinct_uniprot_accessions") or []
                if str(value).strip()
            }
        )
        title = entity.get("entry_title")
        if (
            entity.get("entry_polymer_entity_count") == 1
            and entity["entry_distinct_uniprot_accessions"] == [target_accession]
            and isinstance(title, str)
            and title.strip()
        ):
            entity["entry_title"] = title.strip()
        else:
            entity["entry_title"] = None
        entities.append(entity)
    entities.sort(key=lambda row: (row["pdb_id"], str(row.get("entity_id") or "")))
    target_names, target_name_audit = bounded_match_labels(
        group.get("target_names") or [], 32
    )
    return {
        "group_id": str(group["group_id"]),
        "doi": str(group["doi"]),
        "uniprot_accession": target_accession,
        "target_names": target_names,
        "target_name_alias_audit": target_name_audit,
        "pdb_ids": sorted({str(value).upper() for value in group.get("pdb_ids") or []}),
        "entities": entities,
    }


def target_terms_with_audit(group):
    """Return priority-ordered bounded retrieval labels and truncation audit."""
    metadata = concise_target_metadata(group)
    categories = [
        ("accession", [metadata["uniprot_accession"]]),
        ("pdb", metadata["pdb_ids"]),
        ("target_name", metadata["target_names"]),
        (
            "safe_entry_title",
            [
                str(entity.get("entry_title") or "")
                for entity in metadata["entities"]
                if entity.get("entry_title")
            ],
        ),
        (
            "entity_description",
            [
            str(entity.get("description") or "")
            for entity in metadata["entities"]
            if entity.get("description")
            ],
        ),
        (
            "gene",
            [
                gene
                for entity in metadata["entities"]
                for gene in entity.get("source_genes") or []
            ],
        ),
    ]
    description_tokens = []
    for entity in metadata["entities"]:
        description = str(entity.get("description") or "")
        description_tokens.extend(
            token
            for token in re.split(r"[\s,/;()]+", description)
            if len(token) >= 3
            and (any(char.isdigit() for char in token) or token.isupper())
        )
    categories.append(("description_identifier", description_tokens))
    # RCSB gene/protein aliases sometimes use Arabic numerals while papers use
    # Roman numerals (for example CAMK2 versus CaMKII).
    roman_aliases = []
    for _, values in categories:
        for value in values:
            if str(value).endswith("2"):
                roman_aliases.append(str(value)[:-1] + "II")
    categories.append(("roman_numeral_variant", roman_aliases))

    selected = []
    seen = set()
    category_audits = {}
    total_candidates = 0
    for category, values in categories:
        bounded, category_audit = bounded_match_labels(
            values, MAX_RETRIEVAL_TERMS
        )
        category_audits[category] = category_audit
        total_candidates += category_audit["input_label_count"]
        for value in bounded:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            if len(selected) < MAX_RETRIEVAL_TERMS:
                selected.append(value.lower())
    truncated_count = max(0, len(seen) - len(selected))
    source_gene_audits = [
        entity.get("source_gene_alias_audit") or {}
        for entity in metadata["entities"]
    ]
    source_gene_truncated = any(
        item.get("truncated") for item in source_gene_audits
    )
    audit = {
        "policy": (
            "accession,pdb,target_name,safe_entry_title,entity_description,"
            "gene,description_identifier,roman_numeral_variant"
        ),
        "input_candidate_count": total_candidates,
        "valid_unique_term_count": len(seen),
        "returned_term_count": len(selected),
        "truncated_term_count": truncated_count,
        "truncated": any(
            item.get("truncated") for item in category_audits.values()
        )
        or bool(truncated_count)
        or source_gene_truncated,
        "complete": False,
        "limit": MAX_RETRIEVAL_TERMS,
        "categories": category_audits,
        "source_gene_aliases": {
            "entity_count": len(source_gene_audits),
            "entities_with_truncation": sum(
                bool(item.get("truncated")) for item in source_gene_audits
            ),
            "truncated": source_gene_truncated,
            "complete": not source_gene_truncated,
            "audits": source_gene_audits,
        },
    }
    audit["complete"] = not audit["truncated"]
    return selected, audit


def target_terms(group):
    """Compatibility wrapper returning only bounded target terms."""
    terms, _ = target_terms_with_audit(group)
    return terms


def compact_identity_context(identity_graph):
    """Render only auditable graph fields needed by the extraction model."""
    if not isinstance(identity_graph, dict):
        return {
            "focus_target": {},
            "competing_same_doi_targets": [],
            "identity_edges": [],
        }
    focus_id = str(identity_graph.get("focus_target_id") or "")
    targets = identity_graph.get("targets") or []

    def target_record(target):
        return {
            "target_id": str(target.get("target_id") or ""),
            "uniprot_accession": str(target.get("uniprot_accession") or ""),
            "target_names": sorted(
                {str(value) for value in target.get("target_names") or []}
            ),
            "genes": sorted({str(value) for value in target.get("genes") or []}),
            "species": sorted(
                {str(value) for value in target.get("species") or []}
            ),
        }

    focus = next(
        (target_record(target) for target in targets if target.get("target_id") == focus_id),
        {},
    )
    competing = sorted(
        (
            target_record(target)
            for target in targets
            if target.get("target_id") != focus_id
        ),
        key=lambda item: (item["uniprot_accession"], item["target_id"]),
    )
    nodes = {
        str(node.get("node_id") or ""): node
        for node in identity_graph.get("nodes") or []
    }
    edges = []
    for edge in identity_graph.get("edges") or []:
        source_id = str(edge.get("source_id") or "")
        source_node = nodes.get(str(edge.get("source_node_id") or ""), {})
        edges.append(
            {
                "edge_id": str(edge.get("edge_id") or ""),
                "source_label": str(source_node.get("label") or ""),
                "source_type": str(source_node.get("node_type") or ""),
                "relation": str(edge.get("relation_type") or ""),
                "target_id": str(
                    (edge.get("target_identity") or {}).get("target_id") or ""
                ),
                "anchor_source": source_id,
                "anchor_source_type": (
                    "manifest_metadata"
                    if source_id.startswith("manifest:")
                    else "original_paper"
                ),
                "anchor_quote": str(edge.get("anchor_quote") or ""),
            }
        )
    return {
        "focus_target": focus,
        "competing_same_doi_targets": competing,
        "identity_edges": sorted(edges, key=lambda item: item["edge_id"]),
    }


def render_user_prompt(group, passages, identity_graph=None):
    metadata = concise_target_metadata(group)
    identity_context = compact_identity_context(identity_graph)
    rendered = []
    for passage in passages:
        channels = passage.get("retrieval_families") or [
            passage.get("retrieval_family", "unknown")
        ]
        rendered.append(
            "[SOURCE_ID={source_id} | BASE_SOURCE_ID={base_source_id} | "
            "SECTION={section} | PAGE={page} | "
            "CHAR_OFFSETS={char_start}:{char_end} | "
            "SENTENCE_OFFSETS={sentence_start}:{sentence_end} | "
            "REQUIRES_ABSTENTION={requires_abstention} | "
            "RETRIEVAL={retrieval_family} | RETRIEVAL_CHANNELS={channels}]\n"
            "{text}".format(
                channels=",".join(sorted(set(channels))),
                source_id=passage.get("source_id", ""),
                base_source_id=passage.get(
                    "base_source_id", passage.get("source_id", "")
                ),
                section=passage.get("section", ""),
                page=passage.get("page", ""),
                char_start=passage.get("char_start", 0),
                char_end=passage.get("char_end", len(str(passage.get("text") or ""))),
                sentence_start=passage.get("sentence_start", 1),
                sentence_end=passage.get("sentence_end", 1),
                requires_abstention=str(
                    bool(passage.get("requires_abstention", False))
                ).lower(),
                retrieval_family=passage.get("retrieval_family", "unknown"),
                text=passage.get("text", ""),
            )
        )
    return """EXTRACTION UNIT: canonical manifest group
SCHEMA VERSION: {schema_version}
PROMPT VERSION: {prompt_version}
TARGET METADATA (identity/construct grounding only):
{metadata}

IDENTITY CONTEXT (auditable deterministic graph; edge IDs are proposals only):
{identity_context}

Produce one protein-level result for this exact target and paper. Keep
construct-specific observations separate. Do not classify individual PDBs.

ORIGINAL PAPER SOURCE BLOCKS:
{passages}
""".format(
        schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        metadata=json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2),
        identity_context=json.dumps(
            identity_context, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ),
        passages="\n\n".join(rendered),
    )
