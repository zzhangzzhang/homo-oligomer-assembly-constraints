"""Strict paper-level schema for contextual dynamic-oligomerization extraction.

This contract is deliberately separate from ``schema.py``.  The historical
protein-level v2/v3 registry and its active aliases must remain unchanged.
"""

import hashlib

from .serialization import canonical_json


CONTEXTUAL_SCHEMA_VERSION = "dynamic_oligomerization_paper_v6_1"
SCHEMA_VERSION = CONTEXTUAL_SCHEMA_VERSION


def _object(properties):
    """Build an OpenAI strict-mode object with every property required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


def _enum(*values):
    return {"type": "string", "enum": list(values)}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}
STRING_ARRAY = _array(STRING)

QUANTITATIVE_VALUE_SCHEMA_V6 = _object(
    {
        "name": STRING,
        "value_text": STRING,
        "numeric_value": STRING,
        "unit": STRING,
    }
)

EVIDENCE_REFERENCE_SCHEMA_V6 = _object(
    {
        "quote": STRING,
        "source_id": STRING,
        "page": STRING,
        "section": STRING,
        "occurrence_index": INTEGER,
        "evidence_role": _enum(
            "result",
            "identity_coreference",
            "method",
            "construct_identity",
            "target_attribution",
            "exclusion",
            "citation_scope",
        ),
    }
)

CANDIDATE_REFERENCE_SCHEMA_V6 = _object(
    {
        "originating_group_id": STRING,
        "candidate_kind": _enum("state", "event"),
        "ordinal": INTEGER,
    }
)

COREFERENCE_STEP_SCHEMA_V6 = _object(
    {
        "step_id": STRING,
        "mention_text": STRING,
        "mention_type": _enum(
            "canonical_name",
            "paper_alias",
            "construct",
            "species_name",
            "pronoun",
            "definite_description",
            "implicit_subject",
            "figure_or_table_label",
        ),
        "antecedent_step_id": STRING,
        "relation": _enum(
            "same_entity",
            "alias_of",
            "construct_of",
            "subject_continuity",
            "caption_panel_link",
        ),
        "evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    }
)

TARGET_ATTRIBUTION_SCHEMA_V6 = _object(
    {
        "assignment_status": _enum("selected", "ambiguous", "unassigned"),
        "selected_group_id": STRING,
        "target_attribution_status": _enum(
            "exact_target",
            "construct_of_target",
            "competing_target",
            "nonmanifest_protein",
            "ambiguous",
            "unresolved",
        ),
        "candidate_target_ids": STRING_ARRAY,
        "subject_text": STRING,
        "attribution_basis": STRING,
        "coreference_chain": _array(COREFERENCE_STEP_SCHEMA_V6),
        "attribution_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
        "competing_mentions": STRING_ARRAY,
        "ambiguity_reason": STRING,
    }
)

PAPER_TARGET_IDENTITY_SCHEMA_V6 = _object(
    {
        "group_id": STRING,
        "paper_role": _enum(
            "primary_experimental_target",
            "secondary_experimental_target",
            "experimental_comparator",
            "methods_only",
            "mentioned_only",
            "absent",
            "ambiguous",
            "unresolved",
        ),
        "paper_local_mentions": STRING_ARRAY,
        "species": STRING_ARRAY,
        "identity_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    }
)

CONSTRUCT_SCHEMA_V6 = _object(
    {
        "construct_id": STRING,
        "selected_group_id": STRING,
        "relation_to_parent": _enum(
            "full_length",
            "domain",
            "fragment",
            "truncation",
            "variant",
            "processed",
            "tagged",
            "ligand_bound",
            "other",
        ),
        "parent_construct_id": STRING,
        "construct_text": STRING,
        "modifications": STRING_ARRAY,
        "chain_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
        "resolution_status": _enum("resolved", "ambiguous", "unresolved"),
    }
)

CITATION_SCOPE_SCHEMA_V6 = _object(
    {
        "status": _enum(
            "current_paper",
            "prior_cited_work",
            "mixed_separable",
            "mixed_unseparable",
            "unclear",
        ),
        "citation_marker_texts": STRING_ARRAY,
        "reference_ids": STRING_ARRAY,
        "scope_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    }
)

CLAIM_PROVENANCE_PROPERTIES = {
    "claim_id": STRING,
    "candidate_references": _array(CANDIDATE_REFERENCE_SCHEMA_V6),
    "candidate_relationship": _enum(
        "independently_recovered",
        "confirms_candidate",
        "corrects_candidate",
        "contradicts_candidate",
        "no_candidate",
    ),
    "candidate_semantic_correction": STRING,
    "target_attribution": TARGET_ATTRIBUTION_SCHEMA_V6,
    "construct_id": STRING,
    "result_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    "identity_context_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    "method": STRING,
    "method_evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    "method_result_link_type": _enum(
        "same_quote",
        "same_source_block",
        "explicit_cross_reference",
        "separate_context_only",
        "unresolved",
    ),
    "claim_origin": _enum(
        "current_paper_experiment",
        "current_paper_interpretation",
        "prior_citation",
        "methods_only",
        "background_uncited",
        "unknown",
    ),
    "citation_scope": CITATION_SCOPE_SCHEMA_V6,
}

ASSEMBLY_CLASS = _enum(
    "discrete_free_homo_oligomer",
    "templated_by_nucleic_acid",
    "membrane_or_surface_assembly",
    "partner_templated",
    "filament",
    "aggregation",
    "phase_separation",
    "other",
    "unknown",
)

STATE_CLAIM_SCHEMA_V6 = _object(
    {
        **CLAIM_PROVENANCE_PROPERTIES,
        "state_label": STRING,
        "subunit_count": INTEGER,
        "subunit_count_text": STRING,
        "assembly_class": ASSEMBLY_CLASS,
        "population_text": STRING,
        "quantitative_values": _array(QUANTITATIVE_VALUE_SCHEMA_V6),
        "biological_context": _enum(
            "purified_solution",
            "cellular",
            "in_vivo",
            "crystallographic",
            "mixed",
            "unknown",
        ),
        "conditions": STRING_ARRAY,
    }
)

EVENT_CLAIM_SCHEMA_V6 = _object(
    {
        **CLAIM_PROVENANCE_PROPERTIES,
        "pathway_id": STRING,
        "pathway_step_index": INTEGER,
        "direction_processes": _array(
            _enum(
                "association",
                "dissociation",
                "redistribution",
                "no_net_direction",
                "unknown",
            )
        ),
        "dynamic_regime": _enum(
            "reversible_equilibrium",
            "coexistence",
            "irreversible",
            "unresolved_heterogeneity",
            "unknown",
        ),
        "condition_linked": BOOLEAN,
        "from_states": STRING_ARRAY,
        "to_states": STRING_ARRAY,
        "from_target_subunit_counts": _array(INTEGER),
        "to_target_subunit_counts": _array(INTEGER),
        "from_assembly_classes": _array(ASSEMBLY_CLASS),
        "to_assembly_classes": _array(ASSEMBLY_CLASS),
        "trigger_type": _enum(
            "concentration",
            "ligand",
            "substrate",
            "product",
            "cofactor",
            "drug_or_inhibitor",
            "ph",
            "salt_or_ionic_strength",
            "ion_or_metal",
            "redox",
            "ptm",
            "mutation",
            "temperature",
            "membrane_or_lipid",
            "glycan_or_carbohydrate",
            "nucleic_acid",
            "binding_partner",
            "proteolysis",
            "localization",
            "crowding",
            "other",
            "none",
        ),
        "trigger_name": STRING,
        "compared_conditions": STRING_ARRAY,
        "quantitative_values": _array(QUANTITATIVE_VALUE_SCHEMA_V6),
    }
)

EXCLUDED_CANDIDATE_SCHEMA_V6 = _object(
    {
        "candidate_references": _array(CANDIDATE_REFERENCE_SCHEMA_V6),
        "target_attribution": TARGET_ATTRIBUTION_SCHEMA_V6,
        "construct_id": STRING,
        "subject_text": STRING,
        "reason": _enum(
            "conformational_change_only",
            "interface_binding_without_state_change",
            "noncomparative_method_condition",
            "purification_only",
            "crystal_only",
            "asymmetric_unit_only",
            "heteromeric_not_homo",
            "non_discrete_assembly_misclassified",
            "different_protein",
            "unmatched_construct",
            "prior_citation_only",
            "speculative",
            "identity_unresolved",
            "method_result_link_unproven",
            "quote_or_source_invalid",
            "duplicate_candidate",
            "other",
        ),
        "reason_detail": STRING,
        "evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    }
)

UNASSIGNED_CANDIDATE_SCHEMA_V6 = _object(
    {
        "candidate_references": _array(CANDIDATE_REFERENCE_SCHEMA_V6),
        "subject_text": STRING,
        "candidate_target_ids": STRING_ARRAY,
        "reason": _enum(
            "ambiguous_between_targets",
            "nonmanifest_protein",
            "identity_context_missing",
            "construct_parent_unresolved",
            "subject_unresolved",
            "other",
        ),
        "reason_detail": STRING,
        "evidence": _array(EVIDENCE_REFERENCE_SCHEMA_V6),
    }
)

CONTEXTUAL_OUTPUT_SCHEMA = _object(
    {
        "paper_doi": STRING,
        "paper_target_identities": _array(PAPER_TARGET_IDENTITY_SCHEMA_V6),
        "constructs": _array(CONSTRUCT_SCHEMA_V6),
        "state_claims": _array(STATE_CLAIM_SCHEMA_V6),
        "event_claims": _array(EVENT_CLAIM_SCHEMA_V6),
        "excluded_candidates": _array(EXCLUDED_CANDIDATE_SCHEMA_V6),
        "unassigned_candidates": _array(UNASSIGNED_CANDIDATE_SCHEMA_V6),
        "extraction_summary": STRING,
    }
)
OUTPUT_SCHEMA = CONTEXTUAL_OUTPUT_SCHEMA


def contextual_schema_sha256():
    """Return the canonical checksum of the immutable v6 schema."""
    return hashlib.sha256(
        canonical_json(CONTEXTUAL_OUTPUT_SCHEMA).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CANDIDATE_REFERENCE_SCHEMA_V6",
    "CITATION_SCOPE_SCHEMA_V6",
    "COREFERENCE_STEP_SCHEMA_V6",
    "CONSTRUCT_SCHEMA_V6",
    "CONTEXTUAL_OUTPUT_SCHEMA",
    "CONTEXTUAL_SCHEMA_VERSION",
    "EVIDENCE_REFERENCE_SCHEMA_V6",
    "EVENT_CLAIM_SCHEMA_V6",
    "PAPER_TARGET_IDENTITY_SCHEMA_V6",
    "STATE_CLAIM_SCHEMA_V6",
    "OUTPUT_SCHEMA",
    "SCHEMA_VERSION",
    "TARGET_ATTRIBUTION_SCHEMA_V6",
    "contextual_schema_sha256",
]
