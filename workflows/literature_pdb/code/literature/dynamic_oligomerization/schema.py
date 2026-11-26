"""Versioned strict schemas for protein-level literature extraction.

The v2 object is retained byte-for-byte in the registry.  New requests use v3;
callers validating persisted output must explicitly select its recorded version.
"""

import copy
import hashlib
import json
from types import MappingProxyType


SCHEMA_V2_VERSION = "dynamic_oligomerization_protein_v2"
SCHEMA_V3_VERSION = "dynamic_oligomerization_protein_v3"


QUANTITATIVE_VALUE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value_text": {"type": "string"},
        "numeric_value": {"type": "string"},
        "unit": {"type": "string"},
    },
    "required": ["name", "value_text", "numeric_value", "unit"],
    "additionalProperties": False,
}

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "quote": {"type": "string", "minLength": 1},
        "source_id": {"type": "string", "minLength": 1},
        "section": {"type": "string"},
        "page": {"type": "string"},
        "evidence_origin": {
            "type": "string",
            "enum": [
                "direct_target_experiment",
                "target_author_interpretation",
                "prior_citation_about_target",
                "different_or_related_protein",
                "insufficient",
            ],
        },
    },
    "required": ["quote", "source_id", "section", "page", "evidence_origin"],
    "additionalProperties": False,
}

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "state_label": {"type": "string"},
        "subunit_count": {"type": "integer", "minimum": 0},
        "assembly_type": {
            "type": "string",
            "enum": [
                "discrete_free_homo_oligomer",
                "templated_by_nucleic_acid",
                "membrane_or_surface_assembly",
                "partner_templated",
                "filament",
                "aggregation",
                "phase_separation",
                "other",
                "unknown",
            ],
        },
        "population_text": {"type": "string"},
        "quantitative_values": {
            "type": "array",
            "items": QUANTITATIVE_VALUE_SCHEMA,
        },
        "method": {"type": "string"},
        "construct": {"type": "string", "minLength": 1},
        "biological_context": {
            "type": "string",
            "enum": [
                "purified_in_vitro",
                "cellular",
                "in_vivo",
                "crystallographic",
                "mixed",
                "unknown",
            ],
        },
        "conditions": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": EVIDENCE_SCHEMA},
    },
    "required": [
        "state_label",
        "subunit_count",
        "assembly_type",
        "population_text",
        "quantitative_values",
        "method",
        "construct",
        "biological_context",
        "conditions",
        "evidence",
    ],
    "additionalProperties": False,
}

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "pathway_id": {"type": "string"},
        "step_index": {"type": "integer", "minimum": 0},
        "event_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "coexistence",
                    "reversible_equilibrium",
                    "condition_linked_shift",
                    "association",
                    "dissociation",
                    "irreversible_assembly_or_disassembly",
                    "unresolved_heterogeneity",
                ],
            },
        },
        "from_states": {"type": "array", "items": {"type": "string"}},
        "to_states": {"type": "array", "items": {"type": "string"}},
        "from_target_subunit_counts": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "to_target_subunit_counts": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "trigger_type": {
            "type": "string",
            "enum": [
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
            ],
        },
        "trigger_name": {"type": "string"},
        "compared_conditions": {"type": "array", "items": {"type": "string"}},
        "direction": {"type": "string"},
        "reversible": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "quantitative_values": {
            "type": "array",
            "items": QUANTITATIVE_VALUE_SCHEMA,
        },
        "method": {"type": "string"},
        "construct": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": EVIDENCE_SCHEMA},
    },
    "required": [
        "pathway_id",
        "step_index",
        "event_types",
        "from_states",
        "to_states",
        "from_target_subunit_counts",
        "to_target_subunit_counts",
        "trigger_type",
        "trigger_name",
        "compared_conditions",
        "direction",
        "reversible",
        "quantitative_values",
        "method",
        "construct",
        "evidence",
    ],
    "additionalProperties": False,
}


def _evidenced_item(properties, required):
    props = dict(properties)
    props["evidence"] = {"type": "array", "items": EVIDENCE_SCHEMA}
    return {
        "type": "object",
        "properties": props,
        "required": list(required) + ["evidence"],
        "additionalProperties": False,
    }


CONFLICT_SCHEMA = _evidenced_item(
    {"conflict": {"type": "string"}, "author_resolution": {"type": "string"}},
    ["conflict", "author_resolution"],
)
FUNCTION_SCHEMA = _evidenced_item(
    {
        "linked_state_or_event": {"type": "string"},
        "effect_or_role": {"type": "string"},
        "effect_direction": {"type": "string"},
        "relationship": {
            "type": "string",
            "enum": ["experimental", "correlative", "author_proposed"],
        },
    },
    ["linked_state_or_event", "effect_or_role", "effect_direction", "relationship"],
)
DISEASE_SCHEMA = _evidenced_item(
    {
        "disease_or_phenotype": {"type": "string"},
        "variant_or_context": {"type": "string"},
        "linked_state_or_event": {"type": "string"},
        "pathogenic_or_protective": {
            "type": "string",
            "enum": ["pathogenic", "protective", "mixed", "unknown"],
        },
        "evidence_tier": {
            "type": "string",
            "enum": [
                "biochemical",
                "cellular",
                "animal",
                "clinical",
                "author_proposed",
            ],
        },
    },
    [
        "disease_or_phenotype",
        "variant_or_context",
        "linked_state_or_event",
        "pathogenic_or_protective",
        "evidence_tier",
    ],
)
THERAPEUTIC_SCHEMA = _evidenced_item(
    {
        "intervention_or_opportunity": {"type": "string"},
        "modality": {"type": "string"},
        "oligomerization_action": {
            "type": "string",
            "enum": [
                "stabilize",
                "disrupt",
                "trap",
                "shift_equilibrium",
                "prevent_assembly",
                "promote_assembly",
                "other",
                "unknown",
            ],
        },
        "target_state_interface_or_pocket": {"type": "string"},
        "evidence_tier": {
            "type": "string",
            "enum": [
                "biochemical",
                "cellular",
                "animal",
                "clinical",
                "explicit_mechanistic_opportunity",
            ],
        },
        "potency_or_outcome": {"type": "string"},
        "selectivity_or_resistance": {"type": "string"},
    },
    [
        "intervention_or_opportunity",
        "modality",
        "oligomerization_action",
        "target_state_interface_or_pocket",
        "evidence_tier",
        "potency_or_outcome",
        "selectivity_or_resistance",
    ],
)
EXCLUDED_SCHEMA = _evidenced_item(
    {
        "candidate_claim": {"type": "string"},
        "reason": {
            "type": "string",
            "enum": [
                "conformational_change_only",
                "interface_binding_without_state_change",
                "noncomparative_method_condition",
                "purification_only",
                "crystal_only",
                "asymmetric_unit_only",
                "heteromeric_not_homo",
                "non_discrete_assembly",
                "different_protein",
                "unmatched_construct",
                "prior_citation_only",
                "speculative",
                "other",
            ],
        },
    },
    ["candidate_claim", "reason"],
)

OUTPUT_SCHEMA = {
    "$id": SCHEMA_V2_VERSION,
    "type": "object",
    "properties": {
        "target_identity_supported": {"type": "boolean"},
        "direct_target_state_evidence": {"type": "boolean"},
        "state_observations": {"type": "array", "items": STATE_SCHEMA},
        "dynamic_events": {"type": "array", "items": EVENT_SCHEMA},
        "conflicting_evidence": {"type": "array", "items": CONFLICT_SCHEMA},
        "functional_links": {"type": "array", "items": FUNCTION_SCHEMA},
        "disease_links": {"type": "array", "items": DISEASE_SCHEMA},
        "therapeutic_links": {"type": "array", "items": THERAPEUTIC_SCHEMA},
        "excluded_claims": {"type": "array", "items": EXCLUDED_SCHEMA},
        "summary": {"type": "string"},
    },
    "required": [
        "target_identity_supported",
        "direct_target_state_evidence",
        "state_observations",
        "dynamic_events",
        "conflicting_evidence",
        "functional_links",
        "disease_links",
        "therapeutic_links",
        "excluded_claims",
        "summary",
    ],
    "additionalProperties": False,
}

# Freeze the historical contract before constructing v3.  Never edit the v2
# definitions above in place: old parsed outputs are validated against this
# exact schema and are never migrated.
OUTPUT_SCHEMA_V2 = copy.deepcopy(OUTPUT_SCHEMA)

CONSTRUCT_RELATION_ENUM = [
    "full_length",
    "exact_variant",
    "exact_domain",
    "truncation",
    "fragment",
    "processed_product",
    "other_construct",
    "unresolved",
]

EVIDENCE_SCHEMA_V3 = copy.deepcopy(EVIDENCE_SCHEMA)
EVIDENCE_SCHEMA_V3["properties"].update(
    {
        "reported_target_mention": {"type": "string"},
        "reported_species": {"type": "string"},
        "identity_anchor_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    }
)
EVIDENCE_SCHEMA_V3["required"].extend(
    ["reported_target_mention", "reported_species", "identity_anchor_ids"]
)

OUTPUT_SCHEMA_V3 = copy.deepcopy(OUTPUT_SCHEMA_V2)
OUTPUT_SCHEMA_V3["$id"] = SCHEMA_V3_VERSION
for _field_name in (
    "state_observations",
    "dynamic_events",
    "conflicting_evidence",
    "functional_links",
    "disease_links",
    "therapeutic_links",
    "excluded_claims",
):
    _item_schema = OUTPUT_SCHEMA_V3["properties"][_field_name]["items"]
    _item_schema["properties"]["evidence"]["items"] = copy.deepcopy(
        EVIDENCE_SCHEMA_V3
    )

for _field_name in ("state_observations", "dynamic_events"):
    _item_schema = OUTPUT_SCHEMA_V3["properties"][_field_name]["items"]
    _item_schema["properties"]["construct_relation"] = {
        "type": "string",
        "enum": list(CONSTRUCT_RELATION_ENUM),
    }
    _item_schema["required"].append("construct_relation")

_v3_exclusion_reasons = OUTPUT_SCHEMA_V3["properties"]["excluded_claims"]["items"][
    "properties"
]["reason"]["enum"]
for _reason in (
    "target_alias_unlinked",
    "alias_ambiguous_multiple_targets",
    "species_unresolved",
    "species_mismatch",
    "construct_parent_unproven",
    "method_result_link_unproven",
    "quote_noncontiguous_layout_abstention",
    "quote_noncontiguous/layout_abstention",
    "related_protein_only",
):
    if _reason not in _v3_exclusion_reasons:
        _v3_exclusion_reasons.append(_reason)

SCHEMA_REGISTRY = MappingProxyType(
    {
        SCHEMA_V2_VERSION: OUTPUT_SCHEMA_V2,
        SCHEMA_V3_VERSION: OUTPUT_SCHEMA_V3,
    }
)


def schema_for_version(version):
    """Return the exact registered schema, rejecting unknown versions."""
    try:
        return SCHEMA_REGISTRY[str(version)]
    except KeyError as exc:
        raise ValueError("unknown schema version: %s" % version) from exc


def schema_sha256(version):
    """Return the canonical checksum for one registered schema."""
    schema = schema_for_version(version)
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Public active-contract aliases.  Historical callers can use
# OUTPUT_SCHEMA_V2/schema_for_version without mutating parsed JSON.
SCHEMA_VERSION = SCHEMA_V3_VERSION
OUTPUT_SCHEMA = OUTPUT_SCHEMA_V3
EVIDENCE_SCHEMA = EVIDENCE_SCHEMA_V3
STATE_SCHEMA = OUTPUT_SCHEMA["properties"]["state_observations"]["items"]
EVENT_SCHEMA = OUTPUT_SCHEMA["properties"]["dynamic_events"]["items"]
CONFLICT_SCHEMA = OUTPUT_SCHEMA["properties"]["conflicting_evidence"]["items"]
FUNCTION_SCHEMA = OUTPUT_SCHEMA["properties"]["functional_links"]["items"]
DISEASE_SCHEMA = OUTPUT_SCHEMA["properties"]["disease_links"]["items"]
THERAPEUTIC_SCHEMA = OUTPUT_SCHEMA["properties"]["therapeutic_links"]["items"]
EXCLUDED_SCHEMA = OUTPUT_SCHEMA["properties"]["excluded_claims"]["items"]
