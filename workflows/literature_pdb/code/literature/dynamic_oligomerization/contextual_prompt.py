"""Validated prompt rendering for paper-level contextual v6 dossiers."""

import hashlib
import json
import re

from .contextual import (
    DOSSIER_SCHEMA_VERSION,
    MAX_TOKEN_CAP,
    MIN_TOKEN_CAP,
    ContextualIndexError,
    assert_predictor_blind,
)
from .serialization import canonical_json


CONTEXTUAL_PROMPT_VERSION = "dynamic_oligomerization_paper_prompt_v6_2"
PROMPT_VERSION = CONTEXTUAL_PROMPT_VERSION


class ContextualDossierRefusal(ValueError):
    """Raised when a dossier cannot safely be rendered into a request."""


CONTEXTUAL_SYSTEM_PROMPT = """You extract paper-level dynamic homo-oligomerization evidence.

Adjudicate every canonical DOI target jointly. Supplied candidates are retrieval hints,
never facts, and their originating group is not an adjudicated target.
Independently scan every supplied source block for additional direct state and
event evidence for every canonical target, including evidence with no candidate.

Identity, attribution, and constructs:
1. Return exactly one paper_target_identities item for every supplied group_id,
   in supplied order. Report its paper role, all supported paper-local mentions,
   species, and exact identity evidence. Do not invent absent mentions.
2. Build an explicit, auditable coreference_chain from the grammatical subject
   back to a paper-local protein mention and then to a canonical group. Every
   discourse step must have a stable step_id, literal mention_text, typed
   mention_type, relation, and exact evidence references. antecedent_step_id is
   empty only for the root, whose relation is same_entity; every later step
   points to an earlier step and must not use same_entity. Implicit subjects
   require their own
   implicit_subject step and preceding subject continuity evidence. Candidate
   origins never resolve identity.
3. Use target attribution status exact_target, construct_of_target,
   competing_target, nonmanifest_protein, ambiguous, or unresolved. Set
   assignment_status=selected with exactly one selected_group_id only when the
   chain resolves; otherwise use ambiguous or unassigned, an empty
   selected_group_id, candidate target IDs, and an explicit ambiguity reason.
   Record attribution_evidence separately and list literal competing_mentions;
   do not hide competing proteins inside free-text attribution_basis.
4. Define each assayed construct once. Assign deterministic construct IDs in
   first-evidence document order as construct:<group_id>:001, :002, and so on.
   Reuse the same ID for the same paper construct. Record full_length, domain,
   fragment, truncation, variant, processed, tagged, ligand_bound, or other,
   its optional parent ID (empty when none), and exact chain evidence.
5. Never project behavior from a domain, fragment, processed product,
   truncation, mutation, tag, ligand-bound preparation, species homologue, or
   other construct to a full-length parent. Abstain if the parent relation is
   missing or ambiguous.

Claims and candidate accounting:
6. A state or event must have one selected group or an explicit ambiguous or
   unassigned attribution. It must name one exact construct_id.
7. Candidate references use originating_group_id, candidate_kind, and ordinal;
   ordinal is the dossier's candidate_ordinal. Empty candidate_references means
   the observation was independently recovered. State/event claims, exclusions,
   and unassigned observations may all use an empty candidate_references array.
   Extract such new evidence whenever the paper supports it.
8. State whether a recovered claim confirms, corrects, contradicts, or is
   independent of candidate semantics. Correct wrong state labels,
   target-subunit counts, direction/process, dynamic regime, condition linkage,
   and triggers instead of copying hints.
9. Account for every supplied candidate exactly once: reference it from a state
   or event claim, excluded_candidates, or unassigned_candidates. Exclusions
   and unassigned items require evidence and a concrete reason. An independently
   recovered exclusion or unassigned observation has no candidate reference and
   does not discharge any supplied candidate.
10. Do not emit a model-controlled acceptance decision. Deterministic
    postprocessing will derive evidence tiers.

Evidence and provenance:
11. Every evidence reference is one contiguous verbatim quote from one supplied
    source block. Never paraphrase, repair text, join blocks, join sentences
    across blocks, or insert ellipses. source_id, page, and section must match
    the block header.
12. occurrence_index is zero-based among exact occurrences of that quote in the
    named source block. Use separate evidence arrays for result,
    identity/coreference context, and method. Context or identity evidence can
    support attribution but can never replace result evidence. Set
    evidence_role from the containing field exactly:
    paper_target_identities.identity_evidence, every coreference step evidence,
    and claim identity_context_evidence use identity_coreference;
    target_attribution.attribution_evidence and unassigned-candidate evidence use
    target_attribution; construct chain_evidence uses construct_identity;
    result_evidence uses result; method_evidence uses method; citation
    scope_evidence uses citation_scope; and excluded-candidate evidence uses
    exclusion.
13. Result evidence must itself state the asserted oligomeric state,
    target-subunit count, self-association/dissociation, coexistence,
    equilibrium, heterogeneity, or population change. A nearby target name,
    construct statement, method, or conformational observation is insufficient.
14. Method evidence must identify the method actually supporting the result.
    Record whether method and result are in the same quote/block, explicitly
    cross-referenced, only separate context, or unresolved. Do not infer a
    method-result link from generic Methods text.
15. Distinguish current_paper_experiment, current_paper_interpretation,
    prior_citation, methods_only, background_uncited, and unknown for each
    claim. citation_scope is an evidence-bearing object, not a label: classify
    it as current_paper, prior_cited_work, mixed_separable, mixed_unseparable,
    or unclear; copy citation_marker_texts literally, record reference_ids, and
    cite exact scope_evidence. Citation language such as "previously reported"
    remains prior_citation even in a target paragraph. Prior citations may be
    recorded but are never direct current-paper results. If current and prior
    assertions cannot be separated, use mixed_unseparable and abstain from
    treating the statement as a direct current-paper result.
16. If required identity, construct, subject, method-result, or quote context is
    missing, abstain, exclude, or leave the candidate unassigned. Never guess.

Scientific semantics:
17. Represent event semantics on three orthogonal dimensions:
    direction_processes, dynamic_regime, and condition_linked. Never encode
    these dimensions in one flat list or infer one dimension from another.
18. direction_processes is a nonempty array using association, dissociation,
    redistribution, no_net_direction, or unknown. Use multiple values only
    when one explicitly ordered pathway step has both association and
    dissociation. Use redistribution for a supported population shift without
    a uniquely directed association or dissociation, and no_net_direction for
    explicit coexistence without a net shift.
19. dynamic_regime is exactly one of reversible_equilibrium, coexistence,
    irreversible, unresolved_heterogeneity, or unknown. Use
    reversible_equilibrium only when equilibrium or reversible
    interconversion is explicit. Coexistence alone is not equilibrium.
20. condition_linked is true only when the event is explicitly compared across
    conditions; then record the trigger type/name and compared conditions.
    A condition-linked association is not automatically an equilibrium.
    If condition_linked is false, use trigger_type=none, an empty trigger_name,
    and an empty compared_conditions array.
21. Reporting one state under one condition does not establish global fixedness.
    Keep direction/process, reversibility/regime, and condition dependence
    independent even when one experiment supports more than one dimension.
22. Distinguish discrete free homo-oligomers from nucleic-acid-, membrane-,
    surface-, or partner-templated assemblies, filaments, aggregation, and phase
    separation. Preserve the assembly class on both sides of events.
23. Association/dissociation counts TARGET protein subunits, not ligand or
    heteromeric partner stoichiometry. Unknown target stoichiometry is 0. A
    0-to-0 event is valid only for explicit changed self-association or assembly.
24. A conformational change or interface binding is not an oligomerization
    event without measured state/population change. Exclude it when tempting.
25. Direct analytical SEC, SEC-MALS, AUC, or another solution measurement can
    support a state. Purification-only SEC, pooled fractions, buffer recipes,
    and unmeasured concentration, pH, salt, or metal conditions do not.
26. Crystal asymmetric-unit counts and packing do not establish a biological
    oligomer. One ASU molecule is not monomer evidence; multiple ASU molecules
    are not oligomer evidence without explicit biological support.
27. Preserve conditions, exact quantitative wording and units, biological
    context, trigger, direction, reversibility, and pathway fields. Never
    silently convert suspicious units.
28. Do not collapse ordered pathways. Extract each explicit intermediate and
    step with one shared nonempty pathway_id and consecutive one-based step
    indices. Standalone events use an empty pathway_id and index 0.

Use only the predictor-blind dossier below. PDB/entity metadata grounds identity
and constructs but cannot replace quoted paper evidence. Be conservative."""
SYSTEM_PROMPT = CONTEXTUAL_SYSTEM_PROMPT


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()))


def _dossier_content_hash(dossier):
    payload = json.loads(canonical_json(dossier))
    payload.pop("token_audit", None)
    hashes = payload.get("hashes")
    if isinstance(hashes, dict):
        hashes.pop("dossier_content_sha256", None)
    return _sha256_text(canonical_json(payload))


def _serialized_token_count(dossier):
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ContextualDossierRefusal(
            "cannot verify dossier token cap without tiktoken"
        ) from exc
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(canonical_json(dossier)))


def _refuse(message):
    raise ContextualDossierRefusal(message)


def validate_contextual_dossier(dossier):
    """Verify provenance, blindness, context, and cap invariants before render."""
    if not isinstance(dossier, dict):
        _refuse("dossier must be an object")
    if dossier.get("dossier_schema_version") != DOSSIER_SCHEMA_VERSION:
        _refuse("unsupported contextual dossier schema version")

    try:
        fresh_blind_audit = assert_predictor_blind(dossier)
    except ContextualIndexError as exc:
        raise ContextualDossierRefusal(str(exc)) from exc
    recorded_blind_audit = dossier.get("predictor_blind_audit")
    if (
        not isinstance(recorded_blind_audit, dict)
        or recorded_blind_audit.get("passed") is not True
        or recorded_blind_audit.get("violations")
        or not fresh_blind_audit["passed"]
    ):
        _refuse("predictor-blind audit is absent or failed")

    paper = dossier.get("paper_identity")
    hashes = dossier.get("hashes")
    if not isinstance(paper, dict) or not isinstance(hashes, dict):
        _refuse("dossier checksum policy metadata is missing")
    paper_sha256 = str(paper.get("paper_sha256") or "").lower()
    if paper.get("checksum_verified") is not True or not _is_sha256(paper_sha256):
        _refuse("paper checksum is not verified")
    if str(hashes.get("paper_sha256") or "").lower() != paper_sha256:
        _refuse("paper checksum disagrees with dossier hashes")
    required_hashes = (
        "index_inputs_sha256",
        "index_code_sha256",
        "index_policy_sha256",
        "dossier_code_sha256",
        "dossier_policy_sha256",
        "dossier_content_sha256",
    )
    if any(not _is_sha256(hashes.get(key)) for key in required_hashes):
        _refuse("dossier hash policy is incomplete")
    if hashes["dossier_content_sha256"] != _dossier_content_hash(dossier):
        _refuse("dossier content checksum mismatch")

    context = dossier.get("context_audit")
    if not isinstance(context, dict):
        _refuse("context audit is missing")
    if context.get("required_context_unresolved"):
        _refuse("required candidate or identity context is unresolved")
    if context.get("required_candidate_or_anchor_omitted") is not False:
        _refuse("required candidate or identity context was omitted")
    if context.get("cap_exceeded_due_to_required_context") is not False:
        _refuse("required context exceeds the dossier token cap")
    if context.get("selection_mode") not in {
        "full_nonbibliography_text",
        "adaptive_over_cap",
    }:
        _refuse("unknown contextual selection mode")

    token = dossier.get("token_audit")
    if not isinstance(token, dict):
        _refuse("token audit is missing")
    cap = token.get("token_cap")
    recorded_count = token.get("serialized_dossier_token_count")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap < MIN_TOKEN_CAP
        or cap > MAX_TOKEN_CAP
    ):
        _refuse("dossier token cap is outside the configured 120k-200k range")
    if token.get("encoding") != "o200k_base" or token.get("within_cap") is not True:
        _refuse("dossier token audit failed")
    if (
        isinstance(recorded_count, bool)
        or not isinstance(recorded_count, int)
        or recorded_count > cap
    ):
        _refuse("recorded dossier token count exceeds the cap")
    actual_count = _serialized_token_count(dossier)
    if actual_count != recorded_count or actual_count > cap:
        _refuse("dossier token count does not match its audit")

    targets = dossier.get("canonical_targets")
    bundles = dossier.get("candidate_bundles")
    blocks = dossier.get("source_blocks")
    if not isinstance(targets, list) or not targets:
        _refuse("dossier has no canonical targets")
    if not isinstance(bundles, list) or not isinstance(blocks, list) or not blocks:
        _refuse("dossier candidates or source blocks are malformed")
    group_ids = [str(item.get("group_id") or "") for item in targets if isinstance(item, dict)]
    if len(group_ids) != len(targets) or any(not value for value in group_ids):
        _refuse("canonical target group IDs are malformed")
    if len(group_ids) != len(set(group_ids)):
        _refuse("canonical target group IDs are duplicated")

    candidate_keys = set()
    for bundle in bundles:
        if not isinstance(bundle, dict):
            _refuse("candidate bundle is malformed")
        group_id = str(bundle.get("originating_group_id") or "")
        if group_id not in set(group_ids):
            _refuse("candidate bundle references an unknown group")
        for pointer in bundle.get("candidates") or []:
            if not isinstance(pointer, dict):
                _refuse("candidate pointer is malformed")
            kind = pointer.get("candidate_kind")
            ordinal = pointer.get("candidate_ordinal")
            key = (group_id, kind, ordinal)
            if (
                kind not in {"state", "event"}
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or key in candidate_keys
            ):
                _refuse("candidate pointer identity is invalid or duplicated")
            candidate_keys.add(key)

    source_ids = []
    document_indices = []
    for block in blocks:
        if not isinstance(block, dict):
            _refuse("source block is malformed")
        source_id = str(block.get("source_id") or "")
        document_index = block.get("document_index")
        if (
            not source_id
            or isinstance(document_index, bool)
            or not isinstance(document_index, int)
            or not isinstance(block.get("text"), str)
        ):
            _refuse("source block identity, order, or text is invalid")
        source_ids.append(source_id)
        document_indices.append(document_index)
    if len(source_ids) != len(set(source_ids)):
        _refuse("source IDs are duplicated")
    if len(document_indices) != len(set(document_indices)):
        _refuse("source document indices are duplicated")

    return {
        "doi": str(paper.get("doi") or ""),
        "dossier_sha256": hashes["dossier_content_sha256"],
        "target_group_ids": group_ids,
        "candidate_count": len(candidate_keys),
        "source_block_count": len(blocks),
        "serialized_dossier_token_count": actual_count,
        "token_cap": cap,
    }


def _target_preamble(target):
    allowlisted = {
        "group_id": target.get("group_id"),
        "uniprot_accession": target.get("uniprot_accession"),
        "target_names": target.get("target_names") or [],
        "pdb_ids": target.get("pdb_ids") or [],
        "entities": target.get("entities") or [],
        "same_paper_competitor_group_ids": (
            target.get("same_paper_competitor_group_ids") or []
        ),
    }
    return canonical_json(allowlisted)


def _candidate_preamble(bundle):
    return canonical_json(
        {
            "originating_group_id": bundle.get("originating_group_id"),
            "originating_group_role": bundle.get("originating_group_role"),
            "bundle_status": bundle.get("bundle_status"),
            "candidates": bundle.get("candidates") or [],
        }
    )


def render_contextual_user_prompt(dossier):
    """Render one compact audited preamble followed by ordered source blocks."""
    validation = validate_contextual_dossier(dossier)
    targets = "\n".join(
        "TARGET {index}: {value}".format(index=index, value=_target_preamble(target))
        for index, target in enumerate(dossier["canonical_targets"], 1)
    )
    candidates = "\n".join(
        "CANDIDATE_BUNDLE {index}: {value}".format(
            index=index, value=_candidate_preamble(bundle)
        )
        for index, bundle in enumerate(dossier["candidate_bundles"], 1)
    )
    if not candidates:
        candidates = "CANDIDATE_BUNDLES: []"
    anchors = canonical_json(dossier.get("raw_identity_anchor_candidates") or [])

    rendered_blocks = []
    for block in sorted(
        dossier["source_blocks"],
        key=lambda item: (item["document_index"], item["source_id"]),
    ):
        rendered_blocks.append(
            "[SOURCE_ID={source_id} | DOCUMENT_INDEX={document_index} | "
            "SECTION={section} | PAGE={page} | BASE_SOURCE_ID={base_source_id} | "
            "REQUIRES_ABSTENTION={requires_abstention}]\n{text}".format(
                source_id=block["source_id"],
                document_index=block["document_index"],
                section=block.get("section", "unknown"),
                page=block.get("page", "unknown"),
                base_source_id=block.get("base_source_id", block["source_id"]),
                requires_abstention=str(
                    bool(block.get("requires_abstention", False))
                ).lower(),
                text=block["text"],
            )
        )

    return """EXTRACTION UNIT: one DOI, all canonical targets jointly
SCHEMA VERSION: dynamic_oligomerization_paper_v6
PROMPT VERSION: dynamic_oligomerization_paper_prompt_v6
DOI: {doi}
DOSSIER HASHES: {hashes}
CONTEXT AUDIT: {context_audit}
TOKEN AUDIT: {token_audit}
VALIDATED COUNTS: {validation}

CANONICAL TARGETS:
{targets}

CANDIDATE HINTS (not truth; candidate_ordinal maps to output ordinal):
{candidates}

RAW IDENTITY ANCHOR CANDIDATES (not adjudicated mappings):
{anchors}

Return all target identities, constructs, claims, exclusions, and unassigned
candidates under the strict schema. Independently recover uncued evidence.

ORIGINAL PAPER SOURCE BLOCKS IN CANONICAL DOCUMENT ORDER:
{source_blocks}
""".format(
        doi=validation["doi"],
        hashes=canonical_json(dossier["hashes"]),
        context_audit=canonical_json(dossier["context_audit"]),
        token_audit=canonical_json(dossier["token_audit"]),
        validation=canonical_json(validation),
        targets=targets,
        candidates=candidates,
        anchors=anchors,
        source_blocks="\n\n".join(rendered_blocks),
    )


def contextual_system_prompt_sha256():
    return _sha256_text(CONTEXTUAL_SYSTEM_PROMPT)


__all__ = [
    "CONTEXTUAL_PROMPT_VERSION",
    "CONTEXTUAL_SYSTEM_PROMPT",
    "ContextualDossierRefusal",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "contextual_system_prompt_sha256",
    "render_contextual_user_prompt",
    "validate_contextual_dossier",
]
