"""Conservative target-level state classification from accepted claims."""

import re


_MONOMER_PATTERN = re.compile(r"\bmonomer(?:ic|s)?\b", re.IGNORECASE)
_OLIGOMER_PATTERN = re.compile(
    r"\b(?:"
    r"dimer(?:ic|s)?|trimer(?:ic|s)?|tetramer(?:ic|s)?|"
    r"pentamer(?:ic|s)?|hexamer(?:ic|s)?|heptamer(?:ic|s)?|"
    r"octamer(?:ic|s)?|nonamer(?:ic|s)?|decamer(?:ic|s)?|"
    r"dodecamer(?:ic|s)?|oligomer(?:ic|s)?|multimer(?:ic|s)?"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_OLIGOMER_PATTERN = re.compile(
    r"\b(?:"
    r"failed to assemble|loss of|decreased|reduced|destabilized|"
    r"balance|unknown|unspecified"
    r")\b",
    re.IGNORECASE,
)


def _is_accepted(claim):
    return (
        claim.get("deterministic_tiering", {}).get("initial_tier")
        == "llm_validated"
    )


def _selected_group_id(claim):
    attribution = claim.get("target_attribution") or {}
    return str(attribution.get("selected_group_id") or "")


def _state_signals(label, subunit_count, assembly_class):
    if str(assembly_class or "") == "partner_templated":
        return set()
    count = int(subunit_count or 0)
    signals = set()
    if count == 1:
        signals.add("monomer")
    elif count > 1:
        signals.add("oligomer")
    text = str(label or "")
    if _MONOMER_PATTERN.search(text):
        signals.add("monomer")
    if (
        _OLIGOMER_PATTERN.search(text)
        and not _NEGATED_OLIGOMER_PATTERN.search(text)
    ):
        signals.add("oligomer")
    return signals


def _event_endpoint_signals(claim, prefix):
    labels = list(claim.get(prefix + "_states") or [])
    counts = list(claim.get(prefix + "_target_subunit_counts") or [])
    assemblies = list(claim.get(prefix + "_assembly_classes") or [])
    signals = set()
    for index in range(max(len(labels), len(counts), len(assemblies))):
        label = labels[index] if index < len(labels) else ""
        count = counts[index] if index < len(counts) else 0
        assembly = assemblies[index] if index < len(assemblies) else ""
        signals.update(_state_signals(label, count, assembly))
    return signals


def reported_state_classifications(result):
    """Return one conservative state classification per selected target group."""
    signals = {}
    for claim in result.get("state_claims") or []:
        if not _is_accepted(claim):
            continue
        group_id = _selected_group_id(claim)
        if not group_id:
            continue
        signals.setdefault(group_id, set()).update(
            _state_signals(
                claim.get("state_label"),
                claim.get("subunit_count"),
                claim.get("assembly_class"),
            )
        )
    for claim in result.get("event_claims") or []:
        if not _is_accepted(claim):
            continue
        group_id = _selected_group_id(claim)
        if not group_id:
            continue
        group_signals = signals.setdefault(group_id, set())
        group_signals.update(_event_endpoint_signals(claim, "from"))
        group_signals.update(_event_endpoint_signals(claim, "to"))

    classifications = {}
    for group_id, observed in signals.items():
        if observed == {"monomer", "oligomer"}:
            value = "Multiple states"
        elif observed == {"monomer"}:
            value = "Monomer-only"
        elif observed == {"oligomer"}:
            value = "Oligomer-only"
        else:
            value = "Unknown"
        classifications[group_id] = value
    return classifications


__all__ = ["reported_state_classifications"]
