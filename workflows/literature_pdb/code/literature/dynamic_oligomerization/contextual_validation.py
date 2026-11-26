"""Strict offline validation and tiering for contextual paper-level v6 output."""

import copy
import re
from collections import Counter
from functools import lru_cache

from .contextual_schema import (
    CONTEXTUAL_OUTPUT_SCHEMA,
    CONTEXTUAL_SCHEMA_VERSION,
    contextual_schema_sha256,
)
from .doi import normalize_doi
from .quote_normalization import normalized_quote_text

DIRECT_CLAIM_ORIGINS = {
    "current_paper_experiment",
    "current_paper_interpretation",
}
DIRECT_CITATION_SCOPES = {"current_paper", "mixed_separable"}
SELECTED_ATTRIBUTIONS = {"exact_target", "construct_of_target"}
AMBIGUOUS_ATTRIBUTIONS = {"ambiguous", "unresolved"}
HARD_EXCLUSION_REASONS = {
    "conformational_change_only",
    "purification_only",
    "crystal_only",
    "asymmetric_unit_only",
    "heteromeric_not_homo",
    "different_protein",
    "prior_citation_only",
}
EVIDENCE_ARRAY_ROLES = {
    "identity_evidence": "identity_coreference",
    "chain_evidence": "construct_identity",
    "attribution_evidence": "target_attribution",
    "result_evidence": "result",
    "identity_context_evidence": "identity_coreference",
    "method_evidence": "method",
    "scope_evidence": "citation_scope",
}
EVENT_DIRECTION_PROCESSES = {
    "association",
    "dissociation",
    "redistribution",
    "no_net_direction",
    "unknown",
}


class ContextualValidationError(ValueError):
    """A deterministic schema, semantic, accounting, or evidence failure."""

    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = str(code)
        self.details = details or {}


def _fail(code, message, **details):
    raise ContextualValidationError(code, message, details)


@lru_cache(maxsize=1)
def _compiled_validator():
    try:
        import fastjsonschema
    except ImportError as fast_error:
        try:
            from jsonschema.validators import validator_for
        except ImportError as json_error:
            raise RuntimeError(
                "contextual validation requires fastjsonschema or jsonschema "
                "(fastjsonschema import failed: %s; jsonschema import failed: %s)"
                % (fast_error, json_error)
            ) from json_error
        validator_class = validator_for(CONTEXTUAL_OUTPUT_SCHEMA)
        validator_class.check_schema(CONTEXTUAL_OUTPUT_SCHEMA)
        return "jsonschema", validator_class(CONTEXTUAL_OUTPUT_SCHEMA)
    return "fastjsonschema", fastjsonschema.compile(CONTEXTUAL_OUTPUT_SCHEMA)


def validate_contextual_schema(parsed, expected_schema_sha256=None):
    """Validate a model object against the immutable strict v6 schema."""
    if (
        expected_schema_sha256
        and expected_schema_sha256 != contextual_schema_sha256()
    ):
        _fail(
            "schema_identity_mismatch",
            "recorded contextual schema checksum does not match active v6",
        )
    backend, validator = _compiled_validator()
    if backend == "fastjsonschema":
        try:
            validator(parsed)
        except Exception as error:
            parts = list(getattr(error, "path", []) or [])
            if parts and parts[0] == "data":
                parts.pop(0)
            path = "$"
            for part in parts:
                path += "[%d]" % part if isinstance(part, int) else ".%s" % part
            _fail(
                "schema_validation",
                "model output failed contextual schema validation at %s: %s"
                % (path, getattr(error, "message", str(error))),
                path=path,
            )
    else:
        errors = sorted(
            validator.iter_errors(parsed),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            error = errors[0]
            path = "$"
            for part in error.absolute_path:
                path += "[%d]" % part if isinstance(part, int) else ".%s" % part
            _fail(
                "schema_validation",
                "model output failed contextual schema validation at %s: %s"
                % (path, error.message),
                path=path,
            )
    return parsed


def candidate_reference_key(reference):
    return (
        str(reference.get("originating_group_id") or ""),
        str(reference.get("candidate_kind") or ""),
        reference.get("ordinal"),
    )


def supplied_candidate_keys(dossier):
    keys = []
    for bundle in dossier.get("candidate_bundles") or []:
        group_id = str(bundle.get("originating_group_id") or "")
        for candidate in bundle.get("candidates") or []:
            keys.append(
                (
                    group_id,
                    str(candidate.get("candidate_kind") or ""),
                    candidate.get("candidate_ordinal"),
                )
            )
    return keys


def _find_occurrences(text, quote):
    starts = []
    offset = 0
    while quote and offset <= len(text):
        found = text.find(quote, offset)
        if found < 0:
            break
        starts.append(found)
        offset = found + 1
    return starts


def verify_evidence_reference(reference, source_blocks, expected_role, path="$"):
    """Verify one reference without source correction or fuzzy acceptance."""
    source_id = str(reference.get("source_id") or "")
    source = source_blocks.get(source_id)
    if source is None:
        _fail(
            "evidence_source_missing",
            "%s names an unknown dossier source_id %r" % (path, source_id),
            path=path,
            source_id=source_id,
        )
    for field in ("page", "section"):
        expected = str(source.get(field, "unknown"))
        actual = str(reference.get(field, ""))
        if actual != expected:
            _fail(
                "evidence_provenance_mismatch",
                "%s %s does not match dossier source (%r != %r)"
                % (path, field, actual, expected),
                path=path,
                field=field,
                actual=actual,
                expected=expected,
            )
    role = str(reference.get("evidence_role") or "")
    if role != expected_role:
        _fail(
            "evidence_role_mismatch",
            "%s evidence_role must be %s, got %s"
            % (path, expected_role, role),
            path=path,
            expected_role=expected_role,
            actual_role=role,
        )
    quote = str(reference.get("quote") or "")
    if not quote:
        _fail("evidence_quote_empty", "%s has an empty quote" % path, path=path)
    occurrence_index = reference.get("occurrence_index")
    if (
        isinstance(occurrence_index, bool)
        or not isinstance(occurrence_index, int)
        or occurrence_index < 0
    ):
        _fail(
            "evidence_occurrence_invalid",
            "%s occurrence_index must be a non-negative integer" % path,
            path=path,
        )
    text = str(source.get("text") or "")
    exact_starts = _find_occurrences(text, quote)
    if exact_starts:
        if occurrence_index >= len(exact_starts):
            _fail(
                "evidence_occurrence_invalid",
                "%s occurrence_index %d exceeds %d literal occurrences"
                % (path, occurrence_index, len(exact_starts)),
                path=path,
                occurrence_count=len(exact_starts),
            )
        start = exact_starts[occurrence_index]
        return {
            "verified": True,
            "match_type": "literal_exact",
            "source_id": source_id,
            "occurrence_index": occurrence_index,
            "occurrence_count": len(exact_starts),
            "character_start": start,
            "character_end": start + len(quote),
        }

    normalized_quote = normalized_quote_text(quote)
    normalized_source = normalized_quote_text(text)
    normalized_starts = _find_occurrences(normalized_source, normalized_quote)
    if not normalized_quote or not normalized_starts:
        _fail(
            "evidence_quote_not_exact",
            "%s quote is neither literal nor allowlisted normalized exact" % path,
            path=path,
            source_id=source_id,
        )
    if occurrence_index >= len(normalized_starts):
        _fail(
            "evidence_occurrence_invalid",
            "%s occurrence_index %d exceeds %d normalized exact occurrences"
            % (path, occurrence_index, len(normalized_starts)),
            path=path,
            occurrence_count=len(normalized_starts),
        )
    return {
        "verified": True,
        "match_type": "allowlisted_normalized_exact",
        "normalization": "normalized_quote_text_v1",
        "source_id": source_id,
        "occurrence_index": occurrence_index,
        "occurrence_count": len(normalized_starts),
        "normalized_character_start": normalized_starts[occurrence_index],
        "normalized_character_end": (
            normalized_starts[occurrence_index] + len(normalized_quote)
        ),
    }


def _verify_evidence_array(values, source_blocks, expected_role, path, summary):
    for index, evidence in enumerate(values):
        check = verify_evidence_reference(
            evidence,
            source_blocks,
            expected_role,
            "%s[%d]" % (path, index),
        )
        evidence["verification"] = check
        summary[check["match_type"]] += 1


def _validate_coreference(attribution, source_blocks, path, summary):
    steps = attribution.get("coreference_chain") or []
    seen = set()
    for index, step in enumerate(steps):
        step_path = "%s.coreference_chain[%d]" % (path, index)
        step_id = str(step.get("step_id") or "")
        if not step_id or step_id in seen:
            _fail(
                "coreference_step_invalid",
                "%s has an empty or duplicate step_id" % step_path,
                path=step_path,
            )
        antecedent = str(step.get("antecedent_step_id") or "")
        relation = str(step.get("relation") or "")
        if index == 0:
            if antecedent or relation != "same_entity":
                _fail(
                    "coreference_root_invalid",
                    "%s root must have empty antecedent and same_entity relation"
                    % step_path,
                    path=step_path,
                )
        elif not antecedent or antecedent not in seen:
            _fail(
                "coreference_antecedent_invalid",
                "%s must reference an earlier step" % step_path,
                path=step_path,
                antecedent_step_id=antecedent,
            )
        elif relation == "same_entity":
            _fail(
                "coreference_relation_invalid",
                "%s uses root-only same_entity relation" % step_path,
                path=step_path,
            )
        if not step.get("evidence"):
            _fail(
                "coreference_evidence_missing",
                "%s has no identity/coreference evidence" % step_path,
                path=step_path,
            )
        seen.add(step_id)
        _verify_evidence_array(
            step.get("evidence") or [],
            source_blocks,
            "identity_coreference",
            step_path + ".evidence",
            summary,
        )


def _validate_attribution(
    attribution,
    target_ids,
    source_blocks,
    path,
    summary,
):
    assignment = attribution["assignment_status"]
    selected = str(attribution.get("selected_group_id") or "")
    status = attribution["target_attribution_status"]
    candidates = [str(value) for value in attribution.get("candidate_target_ids") or []]
    if len(candidates) != len(set(candidates)) or any(
        candidate not in target_ids for candidate in candidates
    ):
        _fail(
            "target_attribution_candidate_ids",
            "%s has duplicate or unknown candidate_target_ids" % path,
            path=path,
        )
    if assignment == "selected":
        if selected not in target_ids:
            _fail(
                "target_attribution_selected_id",
                "%s selected_group_id is not a supplied target" % path,
                path=path,
                selected_group_id=selected,
            )
        if status not in {
            "exact_target",
            "construct_of_target",
            "competing_target",
        }:
            _fail(
                "target_attribution_assignment",
                "%s selected assignment has incompatible attribution status" % path,
                path=path,
            )
        if not attribution.get("coreference_chain"):
            _fail(
                "coreference_chain_missing",
                "%s selected assignment has no coreference chain" % path,
                path=path,
            )
        if not attribution.get("attribution_evidence"):
            _fail(
                "target_attribution_evidence_missing",
                "%s selected assignment has no target-attribution evidence" % path,
                path=path,
            )
    else:
        if selected:
            _fail(
                "target_attribution_assignment",
                "%s non-selected assignment must have empty selected_group_id" % path,
                path=path,
            )
        allowed = (
            {"ambiguous", "unresolved"}
            if assignment == "ambiguous"
            else {"ambiguous", "unresolved", "nonmanifest_protein"}
        )
        if status not in allowed:
            _fail(
                "target_attribution_assignment",
                "%s non-selected assignment has incompatible attribution status"
                % path,
                path=path,
            )
    _validate_coreference(attribution, source_blocks, path, summary)
    _verify_evidence_array(
        attribution.get("attribution_evidence") or [],
        source_blocks,
        "target_attribution",
        path + ".attribution_evidence",
        summary,
    )


def _construct_index(constructs, target_ids, source_blocks, summary):
    indexed = {}
    for index, construct in enumerate(constructs):
        path = "$.constructs[%d]" % index
        construct_id = str(construct.get("construct_id") or "")
        selected = str(construct.get("selected_group_id") or "")
        if not construct_id or construct_id in indexed:
            _fail(
                "construct_id_invalid",
                "%s has an empty or duplicate construct_id" % path,
                path=path,
            )
        if selected not in target_ids:
            _fail(
                "construct_target_invalid",
                "%s selected_group_id is not a supplied target" % path,
                path=path,
            )
        if (
            construct.get("resolution_status") == "resolved"
            and not construct.get("chain_evidence")
        ):
            _fail(
                "construct_evidence_missing",
                "%s resolved construct has no chain evidence" % path,
                path=path,
            )
        indexed[construct_id] = construct
        _verify_evidence_array(
            construct.get("chain_evidence") or [],
            source_blocks,
            "construct_identity",
            path + ".chain_evidence",
            summary,
        )
    for construct_id, construct in indexed.items():
        parent = str(construct.get("parent_construct_id") or "")
        if parent:
            if parent == construct_id or parent not in indexed:
                _fail(
                    "construct_parent_invalid",
                    "construct %s has an unknown or self parent" % construct_id,
                    construct_id=construct_id,
                    parent_construct_id=parent,
                )
            if (
                indexed[parent]["selected_group_id"]
                != construct["selected_group_id"]
            ):
                _fail(
                    "construct_parent_target_mismatch",
                    "construct %s and its parent resolve to different targets"
                    % construct_id,
                    construct_id=construct_id,
                )
        visited = {construct_id}
        cursor = parent
        while cursor:
            if cursor in visited:
                _fail(
                    "construct_parent_cycle",
                    "construct parent cycle includes %s" % construct_id,
                    construct_id=construct_id,
                )
            visited.add(cursor)
            cursor = str(indexed[cursor].get("parent_construct_id") or "")
    return indexed


def _validate_claim_relationship(claim, path):
    references = claim.get("candidate_references") or []
    relationship = claim.get("candidate_relationship")
    if references and relationship in {"independently_recovered", "no_candidate"}:
        _fail(
            "candidate_relationship_invalid",
            "%s has candidate references but a no-candidate relationship" % path,
            path=path,
        )
    if not references and relationship not in {
        "independently_recovered",
        "no_candidate",
    }:
        _fail(
            "candidate_relationship_invalid",
            "%s has no candidate references but claims candidate confirmation"
            % path,
            path=path,
        )


def _validate_event_semantics(event, path):
    directions = list(event.get("direction_processes") or [])
    direction_set = set(directions)
    if not directions or len(directions) != len(direction_set):
        _fail(
            "event_direction_process_invalid",
            "%s direction_processes must be nonempty and unique" % path,
            path=path,
        )
    if not direction_set <= EVENT_DIRECTION_PROCESSES:
        _fail(
            "event_direction_process_invalid",
            "%s has an unknown direction/process value" % path,
            path=path,
        )
    if len(directions) > 1 and (
        direction_set != {"association", "dissociation"}
        or not str(event.get("pathway_id") or "")
    ):
        _fail(
            "event_direction_process_invalid",
            "%s may use multiple direction/process values only for an explicit "
            "pathway step containing association and dissociation" % path,
            path=path,
        )

    condition_linked = event.get("condition_linked")
    trigger_type = event.get("trigger_type")
    trigger_name = str(event.get("trigger_name") or "")
    compared_conditions = list(event.get("compared_conditions") or [])
    if condition_linked:
        if (
            trigger_type == "none"
            or not trigger_name
            or len(compared_conditions) < 2
        ):
            _fail(
                "event_condition_link_invalid",
                "%s condition-linked event requires a named non-none trigger "
                "and at least two compared conditions" % path,
                path=path,
            )
    elif trigger_type != "none" or trigger_name or compared_conditions:
        _fail(
            "event_condition_link_invalid",
            "%s non-condition-linked event must use trigger_type=none and empty "
            "trigger_name/compared_conditions" % path,
            path=path,
        )

    for endpoint in ("from", "to"):
        states = list(event.get(endpoint + "_states") or [])
        counts = list(event.get(endpoint + "_target_subunit_counts") or [])
        assemblies = list(event.get(endpoint + "_assembly_classes") or [])
        if not states or len(states) != len(counts) or len(states) != len(assemblies):
            _fail(
                "event_endpoint_shape_invalid",
                "%s %s state, target-subunit-count, and assembly arrays must be "
                "nonempty and aligned" % (path, endpoint),
                path=path,
                endpoint=endpoint,
            )
    return {
        "verified": True,
        "dimensions_independent": True,
        "direction_process_count": len(directions),
        "dynamic_regime": event["dynamic_regime"],
        "condition_linked": condition_linked,
    }


def _validate_event_pathways(events):
    pathways = {}
    for index, event in enumerate(events):
        path = "$.event_claims[%d]" % index
        pathway_id = str(event.get("pathway_id") or "")
        step = event.get("pathway_step_index")
        if pathway_id:
            if step < 1:
                _fail(
                    "event_pathway_step_invalid",
                    "%s explicit pathway step index must be one-based" % path,
                    path=path,
                )
            pathways.setdefault(pathway_id, []).append((step, path))
        elif step != 0:
            _fail(
                "event_pathway_step_invalid",
                "%s standalone event must use pathway_step_index=0" % path,
                path=path,
            )
    for pathway_id, members in pathways.items():
        steps = sorted(step for step, _ in members)
        expected = list(range(1, len(steps) + 1))
        if steps != expected:
            _fail(
                "event_pathway_sequence_invalid",
                "pathway %s must use unique consecutive one-based steps"
                % pathway_id,
                pathway_id=pathway_id,
                actual=steps,
                expected=expected,
            )


def verify_contextual_result(parsed, dossier, expected_schema_sha256=None):
    """Validate schema, semantics, candidate accounting, and every EvidenceRef."""
    validate_contextual_schema(parsed, expected_schema_sha256)
    verified = copy.deepcopy(parsed)
    try:
        expected_doi = normalize_doi(dossier["paper_identity"]["doi"])
        actual_doi = normalize_doi(verified["paper_doi"])
    except Exception as error:
        _fail("paper_doi_invalid", "model or dossier DOI is invalid: %s" % error)
    if actual_doi != expected_doi:
        _fail(
            "paper_doi_mismatch",
            "model paper_doi does not match dossier DOI",
            expected=expected_doi,
            actual=actual_doi,
        )

    target_ids = [
        str(target.get("group_id") or "")
        for target in dossier.get("canonical_targets") or []
    ]
    output_target_ids = [
        str(target.get("group_id") or "")
        for target in verified["paper_target_identities"]
    ]
    if output_target_ids != target_ids or len(output_target_ids) != len(
        set(output_target_ids)
    ):
        _fail(
            "paper_target_identities_mismatch",
            "paper_target_identities must contain every supplied group exactly once "
            "in supplied order",
            expected=target_ids,
            actual=output_target_ids,
        )

    source_blocks = {
        str(block["source_id"]): block for block in dossier.get("source_blocks") or []
    }
    evidence_summary = Counter()
    for index, identity in enumerate(verified["paper_target_identities"]):
        _verify_evidence_array(
            identity.get("identity_evidence") or [],
            source_blocks,
            "identity_coreference",
            "$.paper_target_identities[%d].identity_evidence" % index,
            evidence_summary,
        )

    constructs = _construct_index(
        verified["constructs"],
        set(target_ids),
        source_blocks,
        evidence_summary,
    )
    claim_ids = set()
    referenced_candidates = []
    for field in ("state_claims", "event_claims"):
        for index, claim in enumerate(verified[field]):
            path = "$.%s[%d]" % (field, index)
            claim_id = str(claim.get("claim_id") or "")
            if not claim_id or claim_id in claim_ids:
                _fail(
                    "claim_id_invalid",
                    "%s has an empty or duplicate claim_id" % path,
                    path=path,
                )
            claim_ids.add(claim_id)
            _validate_claim_relationship(claim, path)
            if field == "event_claims":
                claim["event_semantics_validation"] = _validate_event_semantics(
                    claim, path
                )
            attribution = claim["target_attribution"]
            _validate_attribution(
                attribution,
                set(target_ids),
                source_blocks,
                path + ".target_attribution",
                evidence_summary,
            )
            construct_id = str(claim.get("construct_id") or "")
            if construct_id not in constructs:
                _fail(
                    "claim_construct_invalid",
                    "%s references an unknown construct_id" % path,
                    path=path,
                    construct_id=construct_id,
                )
            selected = str(attribution.get("selected_group_id") or "")
            if (
                attribution["assignment_status"] == "selected"
                and constructs[construct_id]["selected_group_id"] != selected
            ):
                _fail(
                    "claim_construct_target_mismatch",
                    "%s construct and target attribution resolve to different groups"
                    % path,
                    path=path,
                )
            for array_name in (
                "result_evidence",
                "identity_context_evidence",
                "method_evidence",
            ):
                _verify_evidence_array(
                    claim.get(array_name) or [],
                    source_blocks,
                    EVIDENCE_ARRAY_ROLES[array_name],
                    path + "." + array_name,
                    evidence_summary,
                )
            _verify_evidence_array(
                claim["citation_scope"].get("scope_evidence") or [],
                source_blocks,
                "citation_scope",
                path + ".citation_scope.scope_evidence",
                evidence_summary,
            )
            referenced_candidates.extend(
                candidate_reference_key(reference)
                for reference in claim.get("candidate_references") or []
            )
    _validate_event_pathways(verified["event_claims"])

    for field in ("excluded_candidates", "unassigned_candidates"):
        expected_role = (
            "exclusion"
            if field == "excluded_candidates"
            else "target_attribution"
        )
        for index, disposition in enumerate(verified[field]):
            path = "$.%s[%d]" % (field, index)
            if field == "excluded_candidates":
                _validate_attribution(
                    disposition["target_attribution"],
                    set(target_ids),
                    source_blocks,
                    path + ".target_attribution",
                    evidence_summary,
                )
                construct_id = str(disposition.get("construct_id") or "")
                if construct_id and construct_id not in constructs:
                    _fail(
                        "excluded_construct_invalid",
                        "%s references an unknown construct_id" % path,
                        path=path,
                    )
            else:
                candidate_ids = [
                    str(value)
                    for value in disposition.get("candidate_target_ids") or []
                ]
                if len(candidate_ids) != len(set(candidate_ids)) or any(
                    value not in target_ids for value in candidate_ids
                ):
                    _fail(
                        "unassigned_target_ids_invalid",
                        "%s has duplicate or unknown candidate_target_ids" % path,
                        path=path,
                    )
            _verify_evidence_array(
                disposition.get("evidence") or [],
                source_blocks,
                expected_role,
                path + ".evidence",
                evidence_summary,
            )
            referenced_candidates.extend(
                candidate_reference_key(reference)
                for reference in disposition.get("candidate_references") or []
            )

    supplied = supplied_candidate_keys(dossier)
    supplied_set = set(supplied)
    counts = Counter(referenced_candidates)
    unknown = sorted(key for key in counts if key not in supplied_set)
    duplicate = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(supplied_set - set(counts))
    if unknown or duplicate or missing:
        _fail(
            "candidate_accounting",
            "candidate references are not complete exactly once",
            supplied_count=len(supplied),
            accounted_count=sum(
                count for key, count in counts.items() if key in supplied_set
            ),
            unknown=[list(key) for key in unknown],
            duplicate=[list(key) for key in duplicate],
            missing=[list(key) for key in missing],
        )
    verified["verification_summary"] = {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "schema_sha256": contextual_schema_sha256(),
        "paper_doi_verified": True,
        "target_identities_verified": len(target_ids),
        "constructs_verified": len(constructs),
        "claims_verified": len(claim_ids),
        "evidence_references_verified": sum(evidence_summary.values()),
        "literal_exact_quotes": evidence_summary["literal_exact"],
        "allowlisted_normalized_exact_quotes": evidence_summary[
            "allowlisted_normalized_exact"
        ],
        "candidate_references_supplied": len(supplied),
        "candidate_references_accounted": len(referenced_candidates),
        "candidate_accounting_exact": True,
    }
    return verified


def _diagnostic(error, default_path, component_path, severity="error"):
    if isinstance(error, ContextualValidationError):
        code = error.code
        details = copy.deepcopy(error.details)
    else:
        code = type(error).__name__
        details = {}
    path = str(details.get("path") or default_path)
    details.setdefault("path", path)
    return {
        "severity": severity,
        "error_code": code,
        "error_type": type(error).__name__,
        "error": str(error),
        "path": path,
        "component_path": component_path,
        "details": details,
    }


def _append_diagnostic(errors, error, path, component_path):
    diagnostic = _diagnostic(error, path, component_path)
    errors.append(diagnostic)
    return diagnostic


def _component_validation(errors, warnings=None):
    warnings = list(warnings or [])
    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "error_codes": sorted({item["error_code"] for item in errors}),
        "warning_codes": sorted({item["error_code"] for item in warnings}),
        "errors": errors,
        "warnings": warnings,
    }


def _verify_evidence_array_tolerant(
    values,
    source_blocks,
    expected_role,
    path,
    component_path,
    errors,
    evidence_summary,
):
    for index, evidence in enumerate(values):
        evidence_path = "%s[%d]" % (path, index)
        evidence_summary["total"] += 1
        try:
            check = verify_evidence_reference(
                evidence,
                source_blocks,
                expected_role,
                evidence_path,
            )
        except Exception as error:
            diagnostic = _append_diagnostic(
                errors,
                error,
                evidence_path,
                component_path,
            )
            evidence["verification"] = {
                "verified": False,
                "match_type": "invalid",
                "error_code": diagnostic["error_code"],
                "path": diagnostic["path"],
                "details": diagnostic["details"],
            }
            evidence_summary["invalid"] += 1
            evidence_summary["invalid_by_code"][diagnostic["error_code"]] += 1
        else:
            evidence["verification"] = check
            evidence_summary["valid"] += 1
            evidence_summary[check["match_type"]] += 1


def _validate_coreference_tolerant(
    attribution,
    source_blocks,
    path,
    component_path,
    errors,
    evidence_summary,
):
    steps = attribution.get("coreference_chain") or []
    seen = set()
    for index, step in enumerate(steps):
        step_path = "%s.coreference_chain[%d]" % (path, index)
        step_id = str(step.get("step_id") or "")
        if not step_id or step_id in seen:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "coreference_step_invalid",
                    "%s has an empty or duplicate step_id" % step_path,
                    {"path": step_path},
                ),
                step_path,
                component_path,
            )
        antecedent = str(step.get("antecedent_step_id") or "")
        relation = str(step.get("relation") or "")
        if index == 0:
            if antecedent or relation != "same_entity":
                _append_diagnostic(
                    errors,
                    ContextualValidationError(
                        "coreference_root_invalid",
                        "%s root must have empty antecedent and same_entity relation"
                        % step_path,
                        {"path": step_path},
                    ),
                    step_path,
                    component_path,
                )
        elif not antecedent or antecedent not in seen:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "coreference_antecedent_invalid",
                    "%s must reference an earlier step" % step_path,
                    {
                        "path": step_path,
                        "antecedent_step_id": antecedent,
                    },
                ),
                step_path,
                component_path,
            )
        elif relation == "same_entity":
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "coreference_relation_invalid",
                    "%s uses root-only same_entity relation" % step_path,
                    {"path": step_path},
                ),
                step_path,
                component_path,
            )
        if not step.get("evidence"):
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "coreference_evidence_missing",
                    "%s has no identity/coreference evidence" % step_path,
                    {"path": step_path},
                ),
                step_path,
                component_path,
            )
        if step_id and step_id not in seen:
            seen.add(step_id)
        _verify_evidence_array_tolerant(
            step.get("evidence") or [],
            source_blocks,
            "identity_coreference",
            step_path + ".evidence",
            component_path,
            errors,
            evidence_summary,
        )


def _validate_attribution_tolerant(
    attribution,
    target_ids,
    source_blocks,
    path,
    component_path,
    errors,
    evidence_summary,
):
    assignment = attribution["assignment_status"]
    selected = str(attribution.get("selected_group_id") or "")
    status = attribution["target_attribution_status"]
    candidates = [str(value) for value in attribution.get("candidate_target_ids") or []]
    if len(candidates) != len(set(candidates)) or any(
        candidate not in target_ids for candidate in candidates
    ):
        _append_diagnostic(
            errors,
            ContextualValidationError(
                "target_attribution_candidate_ids",
                "%s has duplicate or unknown candidate_target_ids" % path,
                {"path": path},
            ),
            path,
            component_path,
        )
    if assignment == "selected":
        if selected not in target_ids:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "target_attribution_selected_id",
                    "%s selected_group_id is not a supplied target" % path,
                    {"path": path, "selected_group_id": selected},
                ),
                path,
                component_path,
            )
        if status not in {
            "exact_target",
            "construct_of_target",
            "competing_target",
        }:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "target_attribution_assignment",
                    "%s selected assignment has incompatible attribution status"
                    % path,
                    {"path": path},
                ),
                path,
                component_path,
            )
        if not attribution.get("coreference_chain"):
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "coreference_chain_missing",
                    "%s selected assignment has no coreference chain" % path,
                    {"path": path},
                ),
                path,
                component_path,
            )
        if not attribution.get("attribution_evidence"):
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "target_attribution_evidence_missing",
                    "%s selected assignment has no target-attribution evidence"
                    % path,
                    {"path": path},
                ),
                path,
                component_path,
            )
    else:
        if selected:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "target_attribution_assignment",
                    "%s non-selected assignment must have empty selected_group_id"
                    % path,
                    {"path": path},
                ),
                path,
                component_path,
            )
        allowed = (
            {"ambiguous", "unresolved"}
            if assignment == "ambiguous"
            else {"ambiguous", "unresolved", "nonmanifest_protein"}
        )
        if status not in allowed:
            _append_diagnostic(
                errors,
                ContextualValidationError(
                    "target_attribution_assignment",
                    "%s non-selected assignment has incompatible attribution status"
                    % path,
                    {"path": path},
                ),
                path,
                component_path,
            )
    _validate_coreference_tolerant(
        attribution,
        source_blocks,
        path,
        component_path,
        errors,
        evidence_summary,
    )
    _verify_evidence_array_tolerant(
        attribution.get("attribution_evidence") or [],
        source_blocks,
        "target_attribution",
        path + ".attribution_evidence",
        component_path,
        errors,
        evidence_summary,
    )


def _add_component_error(component, code, message, path, **details):
    error = ContextualValidationError(
        code,
        message,
        dict(details, path=path),
    )
    diagnostic = _diagnostic(error, path, component["_component_path"])
    component["_component_errors"].append(diagnostic)
    return diagnostic


def _finish_component(component):
    component_path = component["_component_path"]
    component["component_validation"] = _component_validation(
        component.pop("_component_errors"),
        component.pop("_component_warnings"),
    )
    component["component_validation"]["component_path"] = component_path
    component.pop("_component_path", None)


def _prepare_component(component, path):
    component["_component_path"] = path
    component["_component_errors"] = []
    component["_component_warnings"] = []


def _validate_constructs_tolerant(
    constructs,
    target_ids,
    identity_valid,
    source_blocks,
    evidence_summary,
):
    id_counts = Counter(str(item.get("construct_id") or "") for item in constructs)
    by_id = {}
    for index, construct in enumerate(constructs):
        path = "$.constructs[%d]" % index
        _prepare_component(construct, path)
        construct_id = str(construct.get("construct_id") or "")
        selected = str(construct.get("selected_group_id") or "")
        if not construct_id:
            _add_component_error(
                construct,
                "construct_id_invalid",
                "%s has an empty construct_id" % path,
                path,
            )
        elif id_counts[construct_id] > 1:
            _add_component_error(
                construct,
                "construct_id_invalid",
                "%s has a duplicate construct_id" % path,
                path,
                construct_id=construct_id,
            )
        else:
            by_id[construct_id] = construct
        if selected not in target_ids:
            _add_component_error(
                construct,
                "construct_target_invalid",
                "%s selected_group_id is not a supplied target" % path,
                path,
                selected_group_id=selected,
            )
        elif not identity_valid.get(selected, False):
            _add_component_error(
                construct,
                "dependent_identity_invalid",
                "%s depends on an invalid paper target identity" % path,
                path,
                selected_group_id=selected,
            )
        if (
            construct.get("resolution_status") == "resolved"
            and not construct.get("chain_evidence")
        ):
            _add_component_error(
                construct,
                "construct_evidence_missing",
                "%s resolved construct has no chain evidence" % path,
                path,
            )
        _verify_evidence_array_tolerant(
            construct.get("chain_evidence") or [],
            source_blocks,
            "construct_identity",
            path + ".chain_evidence",
            path,
            construct["_component_errors"],
            evidence_summary,
        )

    for construct in constructs:
        path = construct["_component_path"]
        construct_id = str(construct.get("construct_id") or "")
        parent = str(construct.get("parent_construct_id") or "")
        if parent:
            if parent == construct_id or parent not in by_id:
                _add_component_error(
                    construct,
                    "construct_parent_invalid",
                    "%s has an unknown or self parent" % path,
                    path,
                    construct_id=construct_id,
                    parent_construct_id=parent,
                )
            elif (
                by_id[parent].get("selected_group_id")
                != construct.get("selected_group_id")
            ):
                _add_component_error(
                    construct,
                    "construct_parent_target_mismatch",
                    "%s and its parent resolve to different targets" % path,
                    path,
                    construct_id=construct_id,
                    parent_construct_id=parent,
                )
        visited = {construct_id}
        cursor = parent
        while cursor and cursor in by_id:
            if cursor in visited:
                _add_component_error(
                    construct,
                    "construct_parent_cycle",
                    "construct parent cycle includes %s" % construct_id,
                    path,
                    construct_id=construct_id,
                )
                break
            visited.add(cursor)
            cursor = str(by_id[cursor].get("parent_construct_id") or "")

    changed = True
    while changed:
        changed = False
        for construct in constructs:
            parent = str(construct.get("parent_construct_id") or "")
            if (
                parent in by_id
                and by_id[parent]["_component_errors"]
                and not any(
                    item["error_code"] == "dependent_parent_construct_invalid"
                    for item in construct["_component_errors"]
                )
            ):
                _add_component_error(
                    construct,
                    "dependent_parent_construct_invalid",
                    "%s depends on an invalid parent construct"
                    % construct["_component_path"],
                    construct["_component_path"],
                    parent_construct_id=parent,
                )
                changed = True
    valid = {
        construct_id: not construct["_component_errors"]
        for construct_id, construct in by_id.items()
    }
    for construct in constructs:
        _finish_component(construct)
    return by_id, valid


def _validate_event_pathways_tolerant(events):
    pathways = {}
    for index, event in enumerate(events):
        path = "$.event_claims[%d]" % index
        pathway_id = str(event.get("pathway_id") or "")
        step = event.get("pathway_step_index")
        if pathway_id:
            if step < 1:
                _add_component_error(
                    event,
                    "event_pathway_step_invalid",
                    "%s explicit pathway step index must be one-based" % path,
                    path,
                )
            pathways.setdefault(pathway_id, []).append((step, event))
        elif step != 0:
            _add_component_error(
                event,
                "event_pathway_step_invalid",
                "%s standalone event must use pathway_step_index=0" % path,
                path,
            )
    for pathway_id, members in pathways.items():
        steps = sorted(step for step, _ in members)
        expected = list(range(1, len(steps) + 1))
        if steps != expected:
            for _, event in members:
                _add_component_error(
                    event,
                    "event_pathway_sequence_invalid",
                    "pathway %s must use unique consecutive one-based steps"
                    % pathway_id,
                    event["_component_path"],
                    pathway_id=pathway_id,
                    actual=steps,
                    expected=expected,
                )


def _add_candidate_warning(component, code, message, key):
    path = component.get("_component_path") or component.get(
        "component_validation", {}
    ).get("component_path", "$")
    diagnostic = {
        "severity": "warning",
        "error_code": code,
        "error_type": "CandidateAccountingWarning",
        "error": message,
        "path": path,
        "component_path": path,
        "details": {"path": path, "candidate_reference": list(key)},
    }
    component["component_validation"]["warnings"].append(diagnostic)
    validation = component["component_validation"]
    validation["warning_count"] = len(validation["warnings"])
    validation["warning_codes"] = sorted(
        {item["error_code"] for item in validation["warnings"]}
    )


def verify_contextual_result_tolerant(
    parsed,
    dossier,
    expected_schema_sha256=None,
):
    """Verify v6 output component-by-component while retaining strict failures.

    JSON/schema and DOI failures still raise. All later diagnostics are attached
    to the smallest identity, construct, claim, or disposition component.
    """
    validate_contextual_schema(parsed, expected_schema_sha256)
    verified = copy.deepcopy(parsed)
    try:
        expected_doi = normalize_doi(dossier["paper_identity"]["doi"])
        actual_doi = normalize_doi(verified["paper_doi"])
    except Exception as error:
        _fail("paper_doi_invalid", "model or dossier DOI is invalid: %s" % error)
    if actual_doi != expected_doi:
        _fail(
            "paper_doi_mismatch",
            "model paper_doi does not match dossier DOI",
            expected=expected_doi,
            actual=actual_doi,
        )

    target_ids = [
        str(target.get("group_id") or "")
        for target in dossier.get("canonical_targets") or []
    ]
    target_set = set(target_ids)
    source_blocks = {
        str(block["source_id"]): block for block in dossier.get("source_blocks") or []
    }
    evidence_summary = Counter()
    evidence_summary["invalid_by_code"] = Counter()
    component_diagnostics = []
    paper_warnings = []

    identities = verified["paper_target_identities"]
    identity_counts = Counter(str(item.get("group_id") or "") for item in identities)
    identity_by_id = {}
    for index, identity in enumerate(identities):
        path = "$.paper_target_identities[%d]" % index
        _prepare_component(identity, path)
        group_id = str(identity.get("group_id") or "")
        if group_id not in target_set:
            _add_component_error(
                identity,
                "paper_target_identity_unknown",
                "%s group_id is not a supplied target" % path,
                path,
                group_id=group_id,
            )
        elif identity_counts[group_id] > 1:
            _add_component_error(
                identity,
                "paper_target_identity_duplicate",
                "%s duplicates a supplied target identity" % path,
                path,
                group_id=group_id,
            )
        else:
            identity_by_id[group_id] = identity
            if index >= len(target_ids) or group_id != target_ids[index]:
                _add_component_error(
                    identity,
                    "paper_target_identity_order",
                    "%s is not in supplied target order" % path,
                    path,
                    group_id=group_id,
                )
        _verify_evidence_array_tolerant(
            identity.get("identity_evidence") or [],
            source_blocks,
            "identity_coreference",
            path + ".identity_evidence",
            path,
            identity["_component_errors"],
            evidence_summary,
        )
    for group_id in target_ids:
        if group_id not in identity_by_id:
            diagnostic = {
                "severity": "error",
                "error_code": "paper_target_identity_missing",
                "error_type": "ContextualValidationError",
                "error": "paper_target_identities omits supplied group %s" % group_id,
                "path": "$.paper_target_identities",
                "component_path": "$.paper_target_identities",
                "details": {
                    "path": "$.paper_target_identities",
                    "group_id": group_id,
                },
            }
            component_diagnostics.append(diagnostic)
    identity_valid = {
        group_id: (
            group_id in identity_by_id
            and not identity_by_id[group_id]["_component_errors"]
        )
        for group_id in target_ids
    }
    for identity in identities:
        component_diagnostics.extend(identity["_component_errors"])
        _finish_component(identity)

    constructs, construct_valid = _validate_constructs_tolerant(
        verified["constructs"],
        target_set,
        identity_valid,
        source_blocks,
        evidence_summary,
    )
    for construct in verified["constructs"]:
        component_diagnostics.extend(
            construct["component_validation"]["errors"]
        )

    all_claims = verified["state_claims"] + verified["event_claims"]
    claim_counts = Counter(str(item.get("claim_id") or "") for item in all_claims)
    candidate_owners = []
    for field in ("state_claims", "event_claims"):
        for index, claim in enumerate(verified[field]):
            path = "$.%s[%d]" % (field, index)
            _prepare_component(claim, path)
            claim_id = str(claim.get("claim_id") or "")
            if not claim_id or claim_counts[claim_id] > 1:
                _add_component_error(
                    claim,
                    "claim_id_invalid",
                    "%s has an empty or duplicate claim_id" % path,
                    path,
                    claim_id=claim_id,
                )
            try:
                _validate_claim_relationship(claim, path)
            except Exception as error:
                _append_diagnostic(
                    claim["_component_errors"],
                    error,
                    path,
                    path,
                )
            if field == "event_claims":
                try:
                    claim["event_semantics_validation"] = _validate_event_semantics(
                        claim,
                        path,
                    )
                except Exception as error:
                    diagnostic = _append_diagnostic(
                        claim["_component_errors"],
                        error,
                        path,
                        path,
                    )
                    claim["event_semantics_validation"] = {
                        "verified": False,
                        "error_codes": [diagnostic["error_code"]],
                    }
            attribution = claim["target_attribution"]
            _validate_attribution_tolerant(
                attribution,
                target_set,
                source_blocks,
                path + ".target_attribution",
                path,
                claim["_component_errors"],
                evidence_summary,
            )
            selected = str(attribution.get("selected_group_id") or "")
            if (
                attribution.get("assignment_status") == "selected"
                and selected in target_set
                and not identity_valid.get(selected, False)
            ):
                _add_component_error(
                    claim,
                    "dependent_identity_invalid",
                    "%s depends on an invalid paper target identity" % path,
                    path,
                    selected_group_id=selected,
                )
            construct_id = str(claim.get("construct_id") or "")
            if construct_id not in constructs:
                _add_component_error(
                    claim,
                    "claim_construct_invalid",
                    "%s references an unknown construct_id" % path,
                    path,
                    construct_id=construct_id,
                )
            elif not construct_valid.get(construct_id, False):
                _add_component_error(
                    claim,
                    "dependent_construct_invalid",
                    "%s depends on an invalid construct" % path,
                    path,
                    construct_id=construct_id,
                )
            elif (
                attribution.get("assignment_status") == "selected"
                and constructs[construct_id].get("selected_group_id") != selected
            ):
                _add_component_error(
                    claim,
                    "claim_construct_target_mismatch",
                    "%s construct and target attribution resolve to different groups"
                    % path,
                    path,
                    construct_id=construct_id,
                    selected_group_id=selected,
                )
            for array_name in (
                "result_evidence",
                "identity_context_evidence",
                "method_evidence",
            ):
                _verify_evidence_array_tolerant(
                    claim.get(array_name) or [],
                    source_blocks,
                    EVIDENCE_ARRAY_ROLES[array_name],
                    path + "." + array_name,
                    path,
                    claim["_component_errors"],
                    evidence_summary,
                )
            _verify_evidence_array_tolerant(
                claim["citation_scope"].get("scope_evidence") or [],
                source_blocks,
                "citation_scope",
                path + ".citation_scope.scope_evidence",
                path,
                claim["_component_errors"],
                evidence_summary,
            )
            for reference in claim.get("candidate_references") or []:
                candidate_owners.append(
                    (candidate_reference_key(reference), claim, path)
                )
    _validate_event_pathways_tolerant(verified["event_claims"])
    for claim in all_claims:
        component_diagnostics.extend(claim["_component_errors"])
        _finish_component(claim)

    for field in ("excluded_candidates", "unassigned_candidates"):
        expected_role = (
            "exclusion"
            if field == "excluded_candidates"
            else "target_attribution"
        )
        for index, disposition in enumerate(verified[field]):
            path = "$.%s[%d]" % (field, index)
            _prepare_component(disposition, path)
            if field == "excluded_candidates":
                _validate_attribution_tolerant(
                    disposition["target_attribution"],
                    target_set,
                    source_blocks,
                    path + ".target_attribution",
                    path,
                    disposition["_component_errors"],
                    evidence_summary,
                )
                construct_id = str(disposition.get("construct_id") or "")
                if construct_id:
                    if construct_id not in constructs:
                        _add_component_error(
                            disposition,
                            "excluded_construct_invalid",
                            "%s references an unknown construct_id" % path,
                            path,
                            construct_id=construct_id,
                        )
                    elif not construct_valid.get(construct_id, False):
                        _add_component_error(
                            disposition,
                            "dependent_construct_invalid",
                            "%s depends on an invalid construct" % path,
                            path,
                            construct_id=construct_id,
                        )
            else:
                candidate_ids = [
                    str(value)
                    for value in disposition.get("candidate_target_ids") or []
                ]
                if len(candidate_ids) != len(set(candidate_ids)) or any(
                    value not in target_set for value in candidate_ids
                ):
                    _add_component_error(
                        disposition,
                        "unassigned_target_ids_invalid",
                        "%s has duplicate or unknown candidate_target_ids" % path,
                        path,
                    )
            _verify_evidence_array_tolerant(
                disposition.get("evidence") or [],
                source_blocks,
                expected_role,
                path + ".evidence",
                path,
                disposition["_component_errors"],
                evidence_summary,
            )
            for reference in disposition.get("candidate_references") or []:
                candidate_owners.append(
                    (candidate_reference_key(reference), disposition, path)
                )
            component_diagnostics.extend(disposition["_component_errors"])
            _finish_component(disposition)

    supplied = supplied_candidate_keys(dossier)
    supplied_set = set(supplied)
    owners_by_key = {}
    for key, component, path in candidate_owners:
        owners_by_key.setdefault(key, []).append((component, path))
    missing = sorted(supplied_set - set(owners_by_key))
    duplicate = sorted(
        key
        for key, owners in owners_by_key.items()
        if key in supplied_set and len(owners) > 1
    )
    unknown = sorted(key for key in owners_by_key if key not in supplied_set)
    exact_verified = []
    unverified = []
    for key in sorted(supplied_set & set(owners_by_key)):
        owners = owners_by_key[key]
        if len(owners) == 1:
            component = owners[0][0]
            if component["component_validation"]["valid"]:
                exact_verified.append(key)
            else:
                unverified.append(key)
                _add_candidate_warning(
                    component,
                    "candidate_reference_unverified_component",
                    "candidate is referenced once but its component is invalid",
                    key,
                )
    for key in duplicate:
        for component, _ in owners_by_key[key]:
            _add_candidate_warning(
                component,
                "candidate_reference_duplicate",
                "candidate is referenced more than once",
                key,
            )
    for key in unknown:
        for component, _ in owners_by_key[key]:
            _add_candidate_warning(
                component,
                "candidate_reference_unknown",
                "candidate reference is not supplied by the dossier",
                key,
            )
    for key in missing:
        paper_warnings.append(
            {
                "severity": "warning",
                "error_code": "candidate_reference_missing",
                "error_type": "CandidateAccountingWarning",
                "error": "supplied candidate is not referenced by any output component",
                "path": "$",
                "component_path": "$",
                "details": {"path": "$", "candidate_reference": list(key)},
            }
        )
    accounting = {
        "supplied_count": len(supplied),
        "output_reference_occurrence_count": len(candidate_owners),
        "accounted_once_count": sum(
            len(owners_by_key.get(key, [])) == 1 for key in supplied_set
        ),
        "verified_exactly_once_count": len(exact_verified),
        "missing_count": len(missing),
        "duplicate_count": len(duplicate),
        "unverified_count": len(unverified),
        "unknown_count": len(unknown),
        "exact": not missing and not duplicate and not unverified and not unknown,
        "missing": [list(key) for key in missing],
        "duplicate": [list(key) for key in duplicate],
        "unverified": [list(key) for key in unverified],
        "unknown": [list(key) for key in unknown],
    }

    all_components = (
        identities
        + verified["constructs"]
        + all_claims
        + verified["excluded_candidates"]
        + verified["unassigned_candidates"]
    )
    all_warnings = list(paper_warnings)
    for component in all_components:
        all_warnings.extend(component["component_validation"]["warnings"])
    usable_items = [
        item
        for item in (
            all_claims
            + verified["excluded_candidates"]
            + verified["unassigned_candidates"]
        )
        if item["component_validation"]["valid"]
    ]
    if not component_diagnostics and accounting["exact"]:
        usability = "fully_clean"
    elif usable_items:
        usability = "partially_usable"
    else:
        usability = "wholly_unusable"
    invalid_by_code = dict(sorted(evidence_summary["invalid_by_code"].items()))
    verified["candidate_accounting"] = accounting
    verified["component_diagnostics"] = sorted(
        component_diagnostics + all_warnings,
        key=lambda item: (
            item["severity"],
            item["component_path"],
            item["path"],
            item["error_code"],
        ),
    )
    verified["verification_summary"] = {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "schema_sha256": contextual_schema_sha256(),
        "paper_doi_verified": True,
        "verification_mode": "component_tolerant_v1",
        "paper_usability": usability,
        "fully_clean": usability == "fully_clean",
        "partially_usable": usability == "partially_usable",
        "wholly_unusable": usability == "wholly_unusable",
        "component_count": len(all_components),
        "component_error_count": len(component_diagnostics),
        "component_warning_count": len(all_warnings),
        "usable_claim_or_disposition_count": len(usable_items),
        "evidence_references_total": evidence_summary["total"],
        "evidence_references_valid": evidence_summary["valid"],
        "evidence_references_invalid": evidence_summary["invalid"],
        "literal_exact_quotes": evidence_summary["literal_exact"],
        "allowlisted_normalized_exact_quotes": evidence_summary[
            "allowlisted_normalized_exact"
        ],
        "evidence_invalid_by_code": invalid_by_code,
        "candidate_accounting_exact": accounting["exact"],
    }
    return verified


def _all_result_evidence_exact(claim):
    evidence = claim.get("result_evidence") or []
    return bool(evidence) and all(
        item.get("verification", {}).get("match_type")
        in {"literal_exact", "allowlisted_normalized_exact"}
        for item in evidence
    )


def _claim_hard_exclusions(claim):
    reasons = []
    attribution = claim["target_attribution"]
    status = attribution["target_attribution_status"]
    if status == "competing_target":
        reasons.append("competing_target")
    if status == "nonmanifest_protein":
        reasons.append("nonmanifest_target")
    if claim["claim_origin"] == "prior_citation":
        reasons.append("prior_citation_only")
    scope = claim["citation_scope"]["status"]
    if scope == "prior_cited_work":
        reasons.append("prior_citation_only")
    if scope == "mixed_unseparable":
        reasons.append("mixed_unseparable")
    if claim.get("biological_context") == "crystallographic":
        reasons.append("crystallographic_only")
    if claim.get("assembly_class") == "partner_templated":
        reasons.append("heteromeric_or_partner_templated")
    text = " ".join(
        (
            " ".join(claim.get("direction_processes") or [])
            if field == "direction_processes"
            else str(claim.get(field) or "")
        )
        for field in (
            "method",
            "state_label",
            "direction_processes",
            "dynamic_regime",
            "trigger_name",
        )
    ).casefold()
    for name, pattern in (
        ("asymmetric_unit_only", r"\basymmetric unit\b|\basu\b"),
        ("purification_only", r"\bpurification[- ]only\b|\bpooled fractions?\b"),
        ("conformational_change_only", r"\bconformational change only\b"),
        ("heteromeric_not_homo", r"\bheteromer(?:ic|ization|isation)\b"),
    ):
        if re.search(pattern, text):
            reasons.append(name)
    return sorted(set(reasons))


def derive_claim_tier(claim, construct):
    """Derive one initial deterministic policy tier from verified v6 fields."""
    exact_result = _all_result_evidence_exact(claim)
    hard = _claim_hard_exclusions(claim)
    attribution = claim["target_attribution"]
    status = attribution["target_attribution_status"]
    event_semantics_valid = (
        "direction_processes" not in claim
        or claim.get("event_semantics_validation", {}).get("verified") is True
    )
    if hard:
        tier = "excluded_by_policy"
        reasons = hard
    elif (
        attribution["assignment_status"] == "selected"
        and status in SELECTED_ATTRIBUTIONS
        and claim["claim_origin"] in DIRECT_CLAIM_ORIGINS
        and claim["citation_scope"]["status"] in DIRECT_CITATION_SCOPES
        and exact_result
        and claim["method_result_link_type"] != "unresolved"
        and construct.get("resolution_status") == "resolved"
        and event_semantics_valid
    ):
        tier = "llm_validated"
        reasons = ["all_llm_validation_gates_passed"]
    elif (
        (
            attribution["assignment_status"] in {"ambiguous", "unassigned"}
            or status in AMBIGUOUS_ATTRIBUTIONS
        )
        and exact_result
    ):
        tier = "ambiguous_candidate"
        reasons = ["ambiguous_or_unresolved_target_attribution"]
    else:
        tier = "excluded_by_policy"
        reasons = []
        if not exact_result:
            reasons.append("result_evidence_not_exact")
        if claim["claim_origin"] not in DIRECT_CLAIM_ORIGINS:
            reasons.append("unsupported_claim_origin")
        if claim["citation_scope"]["status"] not in DIRECT_CITATION_SCOPES:
            reasons.append("citation_scope_not_direct_and_separable")
        if claim["method_result_link_type"] == "unresolved":
            reasons.append("method_result_link_unresolved")
        if construct.get("resolution_status") != "resolved":
            reasons.append("construct_unresolved")
        if not event_semantics_valid:
            reasons.append("orthogonal_event_semantics_unverified")
        if status not in SELECTED_ATTRIBUTIONS | AMBIGUOUS_ATTRIBUTIONS:
            reasons.append("target_attribution_excluded")
        if not reasons:
            reasons.append("llm_validation_gate_failed")
    return {
        "initial_tier": tier,
        "tier_reasons": sorted(set(reasons)),
        "deterministic_high_confidence": False,
        "deterministic_high_confidence_status": "unassessed",
    }


def derive_contextual_tiers(verified):
    """Attach deterministic tiers while retaining the complete verified result."""
    tiered = copy.deepcopy(verified)
    constructs = {
        str(item["construct_id"]): item for item in tiered.get("constructs") or []
    }
    for field in ("state_claims", "event_claims"):
        for claim in tiered[field]:
            claim["deterministic_tiering"] = derive_claim_tier(
                claim, constructs[claim["construct_id"]]
            )
    for item in tiered["excluded_candidates"]:
        reasons = [str(item.get("reason") or "excluded_candidate")]
        if item.get("reason") in HARD_EXCLUSION_REASONS:
            reasons.append("hard_exclusion")
        item["deterministic_tiering"] = {
            "initial_tier": "excluded_by_policy",
            "tier_reasons": sorted(set(reasons)),
            "deterministic_high_confidence": False,
            "deterministic_high_confidence_status": "unassessed",
        }
    for item in tiered["unassigned_candidates"]:
        nonmanifest = item.get("reason") == "nonmanifest_protein"
        item["deterministic_tiering"] = {
            "initial_tier": (
                "excluded_by_policy" if nonmanifest else "ambiguous_candidate"
            ),
            "tier_reasons": [str(item.get("reason") or "unassigned_candidate")],
            "deterministic_high_confidence": False,
            "deterministic_high_confidence_status": "unassessed",
        }
    return tiered


def _technical_tiering(component):
    validation = component.get("component_validation") or {}
    return {
        "initial_tier": "technical_error",
        "tier_reasons": validation.get("error_codes") or [
            "component_validation_failed"
        ],
        "deterministic_high_confidence": False,
        "deterministic_high_confidence_status": "unassessed",
    }


def derive_contextual_tiers_tolerant(verified):
    """Derive mixed per-component tiers from tolerant verification output."""
    tiered = copy.deepcopy(verified)
    constructs = {
        str(item.get("construct_id") or ""): item
        for item in tiered.get("constructs") or []
        if str(item.get("construct_id") or "")
    }
    for field in ("state_claims", "event_claims"):
        for claim in tiered[field]:
            if not claim.get("component_validation", {}).get("valid"):
                claim["deterministic_tiering"] = _technical_tiering(claim)
                continue
            construct = constructs.get(str(claim.get("construct_id") or ""))
            if not construct or not construct.get("component_validation", {}).get(
                "valid"
            ):
                claim["deterministic_tiering"] = _technical_tiering(claim)
                continue
            claim["deterministic_tiering"] = derive_claim_tier(claim, construct)
    for item in tiered["excluded_candidates"]:
        if not item.get("component_validation", {}).get("valid"):
            item["deterministic_tiering"] = _technical_tiering(item)
            continue
        reasons = [str(item.get("reason") or "excluded_candidate")]
        if item.get("reason") in HARD_EXCLUSION_REASONS:
            reasons.append("hard_exclusion")
        item["deterministic_tiering"] = {
            "initial_tier": "excluded_by_policy",
            "tier_reasons": sorted(set(reasons)),
            "deterministic_high_confidence": False,
            "deterministic_high_confidence_status": "unassessed",
        }
    for item in tiered["unassigned_candidates"]:
        if not item.get("component_validation", {}).get("valid"):
            item["deterministic_tiering"] = _technical_tiering(item)
            continue
        nonmanifest = item.get("reason") == "nonmanifest_protein"
        item["deterministic_tiering"] = {
            "initial_tier": (
                "excluded_by_policy" if nonmanifest else "ambiguous_candidate"
            ),
            "tier_reasons": [str(item.get("reason") or "unassigned_candidate")],
            "deterministic_high_confidence": False,
            "deterministic_high_confidence_status": "unassessed",
        }
    return tiered


def tier_counts(tiered):
    counts = Counter()
    for field in (
        "state_claims",
        "event_claims",
        "excluded_candidates",
        "unassigned_candidates",
    ):
        for item in tiered.get(field) or []:
            counts[item["deterministic_tiering"]["initial_tier"]] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "ContextualValidationError",
    "candidate_reference_key",
    "derive_claim_tier",
    "derive_contextual_tiers",
    "derive_contextual_tiers_tolerant",
    "supplied_candidate_keys",
    "tier_counts",
    "validate_contextual_schema",
    "verify_contextual_result",
    "verify_contextual_result_tolerant",
    "verify_evidence_reference",
]
