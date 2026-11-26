"""Quote verification and deterministic curation gates."""

import copy
import difflib
import json
import re
from functools import lru_cache

from .identity import resolve_claim_identity
from .prompt import target_terms
from .quote_normalization import normalized_quote_text, normalized_text
from .schema import SCHEMA_VERSION, schema_for_version, schema_sha256


FUZZY_QUOTE_THRESHOLD = 0.70
DIRECT_ORIGINS = {"direct_target_experiment", "target_author_interpretation"}
OLIGOMER_WORD_COUNTS = {
    "monomer": 1,
    "dimer": 2,
    "trimer": 3,
    "tetramer": 4,
    "pentamer": 5,
    "hexamer": 6,
    "heptamer": 7,
    "octamer": 8,
    "nonamer": 9,
    "decamer": 10,
}
SPECIES_ALIASES = {
    "homo sapiens": {"human", "humans", "homo sapiens"},
    "mus musculus": {"mouse", "murine", "mice", "mus musculus"},
    "rattus norvegicus": {"rat", "rats", "rattus norvegicus"},
    "escherichia coli": {"e. coli", "e coli", "escherichia coli"},
    "saccharomyces cerevisiae": {
        "yeast",
        "budding yeast",
        "s. cerevisiae",
        "s cerevisiae",
        "saccharomyces cerevisiae",
    },
    "drosophila melanogaster": {
        "fly",
        "fruit fly",
        "drosophila",
        "drosophila melanogaster",
    },
    "danio rerio": {"zebrafish", "danio rerio"},
    "caenorhabditis elegans": {"c. elegans", "c elegans", "caenorhabditis elegans"},
}
PRIOR_CITATION_PATTERNS = (
    r"\b(?:was|were|has been|have been)\s+(?:previously|formerly)\s+(?:reported|shown|described|observed|found)\b",
    r"\b(?:previous|prior|earlier)\s+(?:work|study|studies|reports?|investigations?)\b",
    r"\b(?:previous|prior|earlier)\b.{0,60}\bresults?\b.{0,40}\b(?:reported|showed|shown|found|described|observed)\b",
    r"\b(?:as|consistent with)\s+(?:previously|formerly)\s+(?:reported|shown|described)\b",
    r"\b(?:reported|shown|described|observed|found)\s+(?:previously|earlier|by\s+[A-Z][A-Za-z-]+\s+et\s+al\.?)\b",
    r"\b(?:according to|as reported by)\b",
    r"(?:\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]|\(\s*\d+(?:\s*[-,]\s*\d+)*\s*\))"
    r"\s*(?:also\s+)?(?:reported|showed|found|described|observed)\b",
    r"\b(?:reported|showed|found|described|observed)\b.{0,100}"
    r"(?:\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]|\(\s*\d+(?:\s*[-,]\s*\d+)*\s*\))",
)
EVENT_CHANGE_PATTERNS = (
    r"\b(?:increase[ds]?|decrease[ds]?|shift(?:ed|s|ing)?|redistribut(?:e[ds]?|ed|ion)|"
    r"convert(?:s|ed)?|interconvert(?:s|ed)?|transition(?:s|ed)?|dissociat(?:e[ds]?|ed|ion)|"
    r"equilibrat(?:e[ds]?|ed|ing|ion)|"
    r"associat(?:e[ds]?|ed|ion)|oligomeri[sz](?:e[ds]?|ed|ation)|multimeri[sz](?:e[ds]?|ed|ation)|"
    r"assembl(?:e[ds]?|ed|y)|disassembl(?:e[ds]?|ed|y)|form(?:s|ed|ation)?)\b",
    r"\b(?:equilibrium|coexist(?:s|ed|ence)|population(?:s)?)\b.{0,80}\b(?:monomer|dimer|trimer|tetramer|"
    r"pentamer|hexamer|heptamer|octamer|nonamer|decamer|oligomer|multimer)\w*\b",
    r"\b(?:monomer|dimer|trimer|tetramer|pentamer|hexamer|heptamer|octamer|nonamer|decamer|"
    r"oligomer|multimer)\w*\b.{0,80}\b(?:equilibrium|coexist(?:s|ed|ence)|population(?:s)?)\b",
)
ASSOCIATION_CHANGE_PATTERN = re.compile(
    r"\b(?:associat(?:e[ds]?|ed|ion)|dissociat(?:e[ds]?|ed|ion)|"
    r"oligomeri[sz](?:e[ds]?|ed|ation)|multimeri[sz](?:e[ds]?|ed|ation)|"
    r"assembl(?:e[ds]?|ed|y)|disassembl(?:e[ds]?|ed|y))\b",
    re.I,
)
CONFORMATION_ONLY_PATTERN = re.compile(
    r"\b(?:conformation(?:al)?|fold(?:ing|ed)?|structural rearrangement|domain movement)\b",
    re.I,
)
SPECULATIVE_EVENT_PATTERN = re.compile(
    r"\b(?:may|could|might)\b|\bwe\s+believe\b",
    re.I,
)
DIRECT_POPULATION_MEASUREMENT_PATTERN = re.compile(
    r"\b(?:population(?:s)?|distribution|fraction(?:s)?|percentage|proportion|"
    r"sec(?:-mals)?|size[- ]exclusion chromatography|analytical ultracentrifugation|"
    r"sedimentation|native mass spectrometry|mass photometry|measured molecular "
    r"(?:mass|weight)|chromatograph(?:y|ic) peaks?)\b",
    re.I,
)
DIRECT_STATE_MEASUREMENT_PATTERN = re.compile(
    r"\b(?:nmr|sec(?:-mals)?|size[- ]exclusion chromatography|gel filtration|auc|"
    r"analytical ultracentrifugation|sedimentation|native mass spectrometry|"
    r"mass photometry|light scattering|(?:native\s+)?page|tem|"
    r"transmission electron microscopy|measured molecular (?:mass|weight))\b",
    re.I,
)
PURIFICATION_POOLING_PATTERN = re.compile(
    r"\b(?:monomeric|dimeric|oligomeric)?\s*(?:fraction|fractions|peak|pool)\b"
    r".{0,40}\b(?:pooled|collected|combined|concentrated)\b|"
    r"\b(?:pooled|collected|combined)\b.{0,40}"
    r"\b(?:monomeric|dimeric|oligomeric)?\s*(?:fraction|fractions|peak|pool)\b",
    re.I,
)
STATE_VERIFICATION_ASSERTION_PATTERN = re.compile(
    r"\b(?:showed|demonstrated|measured|determined|revealed|confirmed|verified|"
    r"correspond(?:ed|ing)\s+to)\b|"
    r"\b(?:was|is|remained|formed)\b.{0,35}"
    r"\b(?:monomer|dimer|trimer|tetramer|oligomer|multimer|rings?)\w*\b",
    re.I,
)
CONTEXTUAL_TARGET_SUBJECT_PATTERN = re.compile(
    r"\b(?:the|this|isolated|purified)\s+(?:[a-z0-9-]+\s+){0,3}(?:target\s+)?"
    r"(?:protein|enzyme|domain|construct|fragment|sample|preparation)\b",
    re.I,
)
STABLE_DIMER_ASSOCIATION_PATTERN = re.compile(
    r"\b(?:stable|persistent|long[- ]lived)\s+(?:homo[- ]?)?dimer\b|"
    r"\b(?:homo[- ]?)?dimer(?:ic)?\b.{0,80}"
    r"\b(?:remained|persisted|was stable|retained|time[- ]course)\b|"
    r"\b(?:remained|persisted)\s+(?:a\s+)?(?:stable\s+)?"
    r"(?:homo[- ]?)?dimer(?:ic)?\b"
    r".{0,80}\b(?:time[- ]course|hours?|days?|sedimentation)\b|"
    r"\b(?:prepared|formed|generated)\b.{0,50}\b(?:homo[- ]?)?dimer\b"
    r".{0,80}\b(?:remained|persisted|stable|time[- ]course)\b",
    re.I,
)
FREE_TARGET_PATTERN = re.compile(
    r"\b(?:free|unbound|soluble)\s+(?:target\s+)?(?:protein|monomer|dimer|trimer|"
    r"tetramer|oligomer|multimer)|\bin\s+solution\b|\bafter\s+(?:partner|dna|rna|"
    r"nucleic acid|membrane|lipid)\s+removal\b",
    re.I,
)
EXCLUDED_EVENT_SCOPE_PATTERNS = (
    r"\bhetero(?:dimer|trimer|tetramer|oligomer|multimer)(?:ic|ization|isation|s)?\b",
    r"\b(?:dna|rna|nucleic[- ]acid)[- ](?:bound|templated|scaffolded|induced)\b",
    r"\b(?:on|along|across)\s+(?:the\s+)?(?:dna|rna|nucleic[- ]acid)\b",
    r"\b(?:membrane|lipid|bilayer|surface)[- ](?:bound|templated|associated|inserted)\b",
    r"\b(?:in|on|at)\s+(?:the\s+)?(?:membrane|lipid bilayer|membrane surface)\b",
    r"\b(?:partner|receptor|scaffold)[- ](?:templated|bound|scaffolded|induced)\b",
    r"\b(?:co[- ]?assembl|co[- ]?oligomeri[sz]|mixed[- ]protein complex)\w*\b",
    r"\bcomplex(?:es)?\s+(?:with|between)\s+(?:another|a second|the partner)\s+protein\b",
)
EVIDENCE_FIELDS = (
    "state_observations",
    "dynamic_events",
    "conflicting_evidence",
    "functional_links",
    "disease_links",
    "therapeutic_links",
    "excluded_claims",
)


class ParsedOutputValidationError(ValueError):
    """Raised when parsed model output does not satisfy the strict schema."""


@lru_cache(maxsize=None)
def _compiled_schema_validator(schema_version):
    """Compile one validator per process; validation runs can contain 70k rows."""
    schema = schema_for_version(schema_version)
    try:
        import fastjsonschema
    except ImportError as fastjsonschema_error:
        try:
            from jsonschema.validators import validator_for
        except ImportError as jsonschema_error:
            raise RuntimeError(
                "parsed output validation requires a usable 'fastjsonschema' "
                "or 'jsonschema' package (fastjsonschema import failed: %s; "
                "jsonschema import failed: %s)"
                % (fastjsonschema_error, jsonschema_error)
            ) from jsonschema_error
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        return "jsonschema", validator_class(schema)
    return "fastjsonschema", fastjsonschema.compile(schema)


def validate_parsed_output(
    parsed,
    schema_version=SCHEMA_VERSION,
    expected_schema_sha256=None,
):
    """Validate without migration against an explicitly selected schema."""
    actual_hash = schema_sha256(schema_version)
    if expected_schema_sha256 and expected_schema_sha256 != actual_hash:
        raise ParsedOutputValidationError(
            "recorded schema checksum does not match registered %s" % schema_version
        )
    backend, validator = _compiled_schema_validator(schema_version)
    if backend == "fastjsonschema":
        try:
            validator(parsed)
        except Exception as error:
            parts = list(getattr(error, "path", []) or [])
            if parts and parts[0] == "data":
                parts.pop(0)
            path = "$"
            for part in parts:
                if isinstance(part, int) or str(part).isdigit():
                    path += "[%d]" % int(part)
                else:
                    path += ".%s" % part
            raise ParsedOutputValidationError(
                "model output failed schema validation at %s: %s"
                % (path, getattr(error, "message", str(error)))
            ) from error
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
            raise ParsedOutputValidationError(
                "model output failed schema validation at %s: %s"
                % (path, error.message)
            )
    return parsed


def _fuzzy_score(quote, text):
    query = normalized_text(quote)
    normalized = normalized_text(text)
    if not query:
        return 0.0
    if query in normalized:
        return 1.0
    words = normalized.split()
    width = len(query.split())
    candidates = [normalized]
    candidates.extend(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip()
    )
    if width >= 5:
        stride = max(1, width // 5)
        for extra in (-4, 0, 4):
            size = max(5, width + extra)
            candidates.extend(
                " ".join(words[start : start + size])
                for start in range(0, max(1, len(words) - size + 1), stride)
            )
    return max(
        (difflib.SequenceMatcher(None, query, candidate).ratio() for candidate in candidates),
        default=0.0,
    )


def verify_quote(quote, passages, claimed_source_id=""):
    """Classify literal exact matches separately from normalized/fuzzy matches."""
    quote = str(quote or "")
    by_source = {passage["source_id"]: passage for passage in passages}
    claimed = by_source.get(claimed_source_id)
    claimed_exact = bool(claimed and quote and quote in claimed["text"])
    exact_sources = sorted(
        passage["source_id"]
        for passage in passages
        if quote and quote in passage["text"]
    )
    exact = bool(exact_sources)
    normalized_query = normalized_quote_text(quote)
    normalized_exact_sources = sorted(
        passage["source_id"]
        for passage in passages
        if normalized_query
        and normalized_query in normalized_quote_text(passage["text"])
    )
    normalized_exact = bool(normalized_exact_sources)
    normalized_only_sources = sorted(set(normalized_exact_sources) - set(exact_sources))
    claimed_normalized_exact = bool(
        claimed
        and normalized_query
        and normalized_query in normalized_quote_text(claimed["text"])
    )
    if exact or normalized_exact:
        best_score = 1.0
        best_source = (
            exact_sources[0] if exact_sources else normalized_exact_sources[0]
        )
        claimed_fuzzy = 1.0 if claimed_normalized_exact else (
            _fuzzy_score(quote, claimed["text"]) if claimed else 0.0
        )
    else:
        best_score = 0.0
        best_source = ""
        for passage in passages:
            score = _fuzzy_score(quote, passage["text"])
            if score > best_score:
                best_score, best_source = score, passage["source_id"]
        claimed_fuzzy = _fuzzy_score(quote, claimed["text"]) if claimed else 0.0
    status = (
        "exact"
        if exact
        else "normalized_exact"
        if normalized_exact
        else "fuzzy"
        if best_score >= FUZZY_QUOTE_THRESHOLD
        else "unverified"
    )
    corrected_source_id = (
        exact_sources[0]
        if not claimed_exact and len(exact_sources) == 1
        else ""
    )
    effective_source_id = corrected_source_id or str(claimed_source_id or "")
    return {
        "match_type": status,
        "exact_match": exact,
        "exact_source_ids": exact_sources,
        "normalized_exact_match": normalized_exact and not exact,
        "normalized_exact_source_ids": normalized_only_sources,
        "fuzzy_match_score": round(best_score, 3),
        "best_fuzzy_source_id": best_source,
        "claimed_source_exact": claimed_exact,
        "claimed_source_normalized_exact": claimed_normalized_exact and not claimed_exact,
        "claimed_source_fuzzy_score": round(claimed_fuzzy, 3),
        "claimed_source_id": str(claimed_source_id or ""),
        "corrected_source_id": corrected_source_id,
        "effective_source_id": effective_source_id,
        "effective_source_exact": bool(claimed_exact or corrected_source_id),
    }


def _iter_items(result):
    for field in EVIDENCE_FIELDS:
        for index, item in enumerate(result.get(field, [])):
            yield field, index, item


def _term_in_text(term, text):
    # Exact token boundaries stop short aliases from matching inside words.
    normalized = normalized_text(text)
    for form in _term_forms(term):
        pattern = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(form)
        if re.search(pattern, normalized):
            return True
        if re.fullmatch(r"[a-z0-9-]{3,}", form):
            compact_form = form.replace("-", "")
            compact_text = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", "", normalized)
            if re.search(
                r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(compact_form),
                compact_text,
            ):
                return True
    return False


def _term_forms(term):
    normalized = normalized_text(term)
    forms = {normalized}
    greek_to_letter = {"alpha": "a", "beta": "b", "gamma": "g", "delta": "d"}
    for word, letter in greek_to_letter.items():
        if word in normalized:
            forms.add(normalized.replace(word, letter))
    return {form for form in forms if form}


def _alias_tokens(value):
    return [
        token
        for token in re.findall(
            r"[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ωΔδ-]{1,24}",
            str(value or ""),
        )
        if len(token) >= 2
    ]


def _document_target_terms(group, passages):
    """Build a per-group alias graph from canonical terms and source context."""
    terms = set(target_terms(group))
    ignored = {
        "also",
        "called",
        "domain",
        "human",
        "protein",
        "terminal",
        "construct",
        "fragment",
    }
    for passage in passages:
        text = str(passage.get("text") or "")
        contexts = [
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+|\n+", text)
            if part.strip()
        ]
        changed = True
        while changed:
            changed = False
            for context in contexts:
                tied_terms = [term for term in terms if _term_in_text(term, context)]
                if not tied_terms:
                    continue
                candidates = set()
                for parenthetical_match in re.finditer(r"\(([^()]{1,60})\)", context):
                    parenthetical = parenthetical_match.group(1)
                    left = context[: parenthetical_match.start()]
                    normalized_left = normalized_text(left).rstrip()
                    directly_follows_target = any(
                        normalized_left.endswith(form)
                        for term in tied_terms
                        for form in _term_forms(term)
                    )
                    construct_descriptor = bool(
                        re.search(
                            r"\b(?:domain|terminus|terminal|construct|fragment|"
                            r"component|chain)\s*$",
                            normalized_left,
                        )
                    )
                    if directly_follows_target or construct_descriptor:
                        candidates.update(_alias_tokens(parenthetical))
                    if any(
                        _term_in_text(term, parenthetical) for term in tied_terms
                    ):
                        preceding = _alias_tokens(left[-40:])
                        if preceding:
                            candidates.add(preceding[-1])
                for slash_group in re.findall(
                    r"[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ωΔδ-]*"
                    r"(?:/[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ωΔδ-]*)+",
                    context,
                ):
                    candidates.update(_alias_tokens(slash_group))
                for match in re.finditer(
                    r"\b(?:also\s+known\s+as|also\s+called|hereafter|designated|"
                    r"abbreviated\s+as)\s+"
                    r"([A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ωΔδ-]{1,24})",
                    context,
                    re.I,
                ):
                    candidates.add(match.group(1))
                for token in _alias_tokens(context):
                    token_compact = re.sub(r"[^a-z0-9]", "", normalized_text(token))
                    if any(
                        len(base) >= 3
                        and token_compact.startswith(base)
                        and token_compact != base
                        for term in tied_terms
                        for base in {
                            re.sub(r"[^a-z0-9]", "", form)
                            for form in _term_forms(term)
                        }
                    ):
                        candidates.add(token)
                for candidate in candidates:
                    normalized_candidate = normalized_text(candidate)
                    if (
                        normalized_candidate not in ignored
                        and normalized_candidate not in terms
                        and 2 <= len(normalized_candidate) <= 25
                    ):
                        terms.add(normalized_candidate)
                        changed = True
    return sorted(terms)


def _claim_text(field, item):
    ignored = {
        "evidence",
        "quantitative_values",
        "curation_validation",
        "identity_resolution",
    }
    values = [field]
    for key, value in item.items():
        if key in ignored:
            continue
        if isinstance(value, list):
            values.extend(str(part) for part in value)
        else:
            values.append(str(value))
    values.extend(str(e.get("quote") or "") for e in item.get("evidence", []))
    return " ".join(values)


def _canonical_identity(group, terms):
    organisms = set()
    genes = set()
    for entity in group.get("entities") or []:
        organisms.update(
            normalized_text(value) for value in entity.get("source_organisms") or []
        )
        for value in entity.get("source_genes") or []:
            genes.update(
                normalized_text(token)
                for token in re.split(r"[\s,;/]+", str(value))
                if token
            )
    return {
        "organisms": organisms,
        "genes": genes,
        "accession": normalized_text(group.get("uniprot_accession", "")),
        "terms": {normalized_text(value) for value in terms if len(value) >= 3},
    }


def _species_identifies_target(alias, text, identity):
    markers = identity["genes"] | identity["terms"] | {
        "protein",
        "enzyme",
        "homolog",
        "homologue",
        "ortholog",
        "paralog",
        "construct",
    }
    alias_pattern = re.escape(normalized_text(alias))
    identity_text = re.sub(
        r"(?<![a-z0-9])%s\s+(?:cells?|tissues?|model|assay|host)(?![a-z0-9])"
        % alias_pattern,
        "",
        text,
    )
    return any(
        re.search(
            r"(?<![a-z0-9])(?:%s).{0,50}(?:%s)(?![a-z0-9])|"
            r"(?<![a-z0-9])(?:%s).{0,50}(?:%s)(?![a-z0-9])"
            % (
                alias_pattern,
                re.escape(marker),
                re.escape(marker),
                alias_pattern,
            ),
            identity_text,
        )
        for marker in markers
    )


def _explicit_identity_mismatch(text, identity):
    normalized = normalized_text(text)
    canonical_species = set()
    mentioned_species = set()
    for organism, aliases in SPECIES_ALIASES.items():
        if organism in identity["organisms"]:
            canonical_species.add(organism)
        if any(
            _term_in_text(alias, normalized)
            and _species_identifies_target(alias, normalized, identity)
            for alias in aliases
        ):
            mentioned_species.add(organism)
    species_mismatch = bool(
        canonical_species and mentioned_species and not (canonical_species & mentioned_species)
    )
    accessions = {
        normalized_text(value)
        for value in re.findall(
            r"(?<![A-Z0-9])[A-Z][0-9][A-Z0-9]{3}[0-9](?![A-Z0-9])",
            text,
            re.I,
        )
    }
    accession_mismatch = bool(
        identity["accession"]
        and accessions
        and identity["accession"] not in accessions
    )
    homolog_mismatch = bool(
        re.search(
            r"\b(?:homolog(?:ue)?|ortholog|paralog)\s+(?:of|from)\b|"
            r"\b(?:related|different)\s+protein\b",
            normalized,
        )
        and not any(_term_in_text(gene, normalized) for gene in identity["genes"])
    )
    named_genes = {
        normalized_text(value)
        for value in re.findall(
            r"\bgene(?:\s+symbol)?\s+([A-Za-z][A-Za-z0-9-]{1,15})\b",
            text,
            re.I,
        )
    } - {"expression", "encoding", "family", "product", "transcription"}
    gene_mismatch = bool(
        identity["genes"]
        and named_genes
        and not (identity["genes"] & named_genes)
    )
    return {
        "explicit_species_mismatch": species_mismatch,
        "explicit_accession_mismatch": accession_mismatch,
        "explicit_homolog_mismatch": homolog_mismatch,
        "explicit_gene_mismatch": gene_mismatch,
        "explicit_target_mismatch": species_mismatch
        or accession_mismatch
        or homolog_mismatch
        or gene_mismatch,
    }


def _multi_protein_leakage(text, terms):
    """Detect compact sibling names such as MakA/MakB/MakE in one claim."""
    normalized = normalized_text(text)
    for term in terms:
        target = normalized_text(term)
        if not re.fullmatch(r"[a-z]{4,8}", target):
            continue
        prefix = target[:-1]
        siblings = set(
            re.findall(r"(?<![a-z0-9])%s[a-z0-9](?![a-z0-9])" % re.escape(prefix), normalized)
        )
        if target in siblings and siblings - {target}:
            absent_partners = {
                sibling
                for sibling in siblings - {target}
                if re.search(
                    r"(?<![a-z0-9])"
                    + re.escape(target)
                    + r"(?![a-z0-9])\s+can\s+\w*oligomer\w*"
                    + r"(?:\s*in)?\s+the\s+absence\s+of\s+"
                    + r"(?<![a-z0-9])"
                    + re.escape(sibling)
                    + r"(?![a-z0-9])",
                    normalized,
                )
            }
            if absent_partners == siblings - {target}:
                continue
            return True
    return False


def _likely_prior_citation(text):
    raw = str(text or "")
    normalized = normalized_text(raw)
    return any(re.search(pattern, raw, re.I) for pattern in PRIOR_CITATION_PATTERNS) or bool(
        re.search(r"\bet\s+al\.\s*(?:\(|\[)?\d{4}", raw, re.I)
        and re.search(r"\b(?:reported|showed|found|described|observed)\b", normalized)
    )


def _hydrated_identity_graph(identity_graph, passages):
    if not isinstance(identity_graph, dict):
        return None
    graph = copy.deepcopy(identity_graph)
    source_values = {}
    windows_by_anchor = {}
    for passage in passages:
        source_id = str(passage.get("source_id") or "")
        text = str(passage.get("text") or "")
        if source_id and text:
            source_values.setdefault(source_id, text)
            for anchor_id in {
                source_id,
                str(passage.get("base_source_id") or source_id),
                str(passage.get("source_passage_id") or source_id),
            }:
                if anchor_id:
                    windows_by_anchor.setdefault(anchor_id, set()).add(source_id)
    graph["source_text"] = {
        source_id: source_values[source_id] for source_id in sorted(source_values)
    }
    graph["source_windows_by_base"] = {
        anchor_id: sorted(window_ids)
        for anchor_id, window_ids in sorted(windows_by_anchor.items())
    }
    return graph


def _legacy_identity_resolution(group):
    return {
        "status": "legacy_identity_grounding",
        "target_identity": {
            "group_id": str(group.get("group_id") or ""),
            "doi": str(group.get("doi") or ""),
            "uniprot_accession": str(group.get("uniprot_accession") or ""),
        },
        "matched_mentions": [],
        "matched_node_ids": [],
        "candidate_target_ids": [],
        "ambiguity": "not_evaluated",
        "evidence_checks": [],
        "anchor_path": [],
        "reason": "prepared_metadata_has_no_identity_graph",
    }


def _model_anchor_validation(evidence, identity_graph):
    """Audit model-proposed edge IDs without using them to resolve identity."""
    proposed = [
        str(value)
        for value in evidence.get("identity_anchor_ids") or []
        if str(value)
    ]
    edge_by_id = {
        str(edge.get("edge_id") or ""): edge
        for edge in identity_graph.get("edges") or []
    }
    focus_id = str(identity_graph.get("focus_target_id") or "")
    sources = identity_graph.get("source_text") or {}
    windows = identity_graph.get("source_windows_by_base") or {}
    records = []
    for edge_id in proposed:
        edge = edge_by_id.get(edge_id)
        if edge is None:
            records.append(
                {
                    "edge_id": edge_id,
                    "known": False,
                    "target_id": "",
                    "maps_to_focus_target": False,
                    "anchor_source_id": "",
                    "anchor_quote": "",
                    "anchor_provenance_exact": False,
                    "anchor_provenance_type": "unknown",
                }
            )
            continue
        source_id = str(edge.get("source_id") or "")
        anchor_quote = str(edge.get("anchor_quote") or "")
        manifest = source_id.startswith("manifest:")
        candidate_ids = [source_id] + list(windows.get(source_id) or [])
        provenance_exact = manifest or any(
            candidate_id in sources and anchor_quote in sources[candidate_id]
            for candidate_id in dict.fromkeys(candidate_ids)
        )
        target_id = str(
            (edge.get("target_identity") or {}).get("target_id") or ""
        )
        records.append(
            {
                "edge_id": edge_id,
                "known": True,
                "target_id": target_id,
                "maps_to_focus_target": bool(target_id and target_id == focus_id),
                "relation_type": str(edge.get("relation_type") or ""),
                "anchor_source_id": source_id,
                "anchor_quote": anchor_quote,
                "anchor_provenance_exact": provenance_exact,
                "anchor_provenance_type": (
                    "manifest_metadata" if manifest else "original_paper"
                ),
            }
        )
    known = [record["edge_id"] for record in records if record["known"]]
    unknown = [record["edge_id"] for record in records if not record["known"]]
    valid = [
        record["edge_id"]
        for record in records
        if record["known"]
        and record["maps_to_focus_target"]
        and record["anchor_provenance_exact"]
    ]
    accepted = bool(proposed) and len(valid) == len(proposed)
    return {
        "proposed_identity_anchor_ids": proposed,
        "known_identity_anchor_ids": known,
        "unknown_identity_anchor_ids": unknown,
        "valid_focus_identity_anchor_ids": valid,
        "target_mappings": records,
        "all_proposed_known": bool(proposed) and not unknown,
        "all_map_to_focus_target": bool(proposed)
        and all(record["maps_to_focus_target"] for record in records),
        "all_anchor_provenance_exact": bool(proposed)
        and all(record["anchor_provenance_exact"] for record in records),
        "accepted_as_verified_proposals": accepted,
        "authoritative_for_identity_resolution": False,
    }


def _typed_identity_sufficient(resolution):
    return bool(
        resolution.get("status") == "exact"
        or (
            resolution.get("status") == "construct_of_exact"
            and resolution.get("construct_scoped_claim")
        )
    )


def verify_result(parsed, passages, group, identity_graph=None):
    """Return a verified copy; never mutate parsed model output."""
    verified = copy.deepcopy(parsed)
    terms = _document_target_terms(group, passages)
    identity = _canonical_identity(group, terms)
    hydrated_graph = _hydrated_identity_graph(identity_graph, passages)
    passage_by_id = {
        str(passage.get("source_id") or ""): passage for passage in passages
    }
    for field, _, item in _iter_items(verified):
        for evidence in item.get("evidence", []):
            claimed_source_id = str(evidence.get("source_id") or "")
            check = verify_quote(
                evidence.get("quote", ""), passages, claimed_source_id
            )
            corrected_source_id = check.get("corrected_source_id", "")
            if corrected_source_id:
                evidence["claimed_source_id"] = claimed_source_id
                evidence["corrected_source_id"] = corrected_source_id
                evidence["source_id"] = corrected_source_id
            effective_passage = passage_by_id.get(
                str(evidence.get("source_id") or "")
            )
            check["layout_requires_abstention"] = bool(
                effective_passage
                and effective_passage.get("requires_abstention", False)
            )
            evidence["verification"] = check
            if hydrated_graph is not None and "identity_anchor_ids" in evidence:
                evidence["model_identity_anchor_validation"] = (
                    _model_anchor_validation(evidence, hydrated_graph)
                )
        if hydrated_graph is None:
            item["identity_resolution"] = _legacy_identity_resolution(group)
        else:
            item["identity_resolution"] = resolve_claim_identity(
                item,
                hydrated_graph,
                target_group_id=str(group.get("group_id") or ""),
            )
        claim_text = _claim_text(field, item)
        claim_mismatch = _explicit_identity_mismatch(claim_text, identity)
        construct = str(item.get("construct") or "")
        for evidence in item.get("evidence", []):
            check = evidence["verification"]
            quote = evidence.get("quote", "")
            quote_target_grounded = any(_term_in_text(term, quote) for term in terms)
            construct_target_grounded = bool(
                CONTEXTUAL_TARGET_SUBJECT_PATTERN.search(quote)
            ) and any(
                _term_in_text(term, construct) for term in terms
            )
            check["exact_target_grounding"] = bool(
                quote_target_grounded or construct_target_grounded
            )
            check["target_grounding_basis"] = (
                "quote"
                if quote_target_grounded
                else "claim_construct_alias"
                if construct_target_grounded
                else "none"
            )
            evidence_identity = None
            if hydrated_graph is not None:
                evidence_claim = {
                    key: copy.deepcopy(value)
                    for key, value in item.items()
                    if key not in {"evidence", "identity_resolution"}
                }
                evidence_claim["evidence"] = [copy.deepcopy(evidence)]
                evidence_identity = resolve_claim_identity(
                    evidence_claim,
                    hydrated_graph,
                    target_group_id=str(group.get("group_id") or ""),
                )
                check["identity_resolution"] = evidence_identity
                check["exact_target_grounding"] = _typed_identity_sufficient(
                    evidence_identity
                )
                check["target_grounding_basis"] = (
                    "typed_exact_identity_path"
                    if evidence_identity.get("status") == "exact"
                    else "typed_construct_identity_path"
                    if _typed_identity_sufficient(evidence_identity)
                    else "none"
                )
            quote_mismatch = _explicit_identity_mismatch(quote, identity)
            multi_protein_leakage = _multi_protein_leakage(quote, terms)
            check.update(
                {
                    key: bool(
                        claim_mismatch[key]
                        or quote_mismatch[key]
                        or (key == "explicit_target_mismatch" and multi_protein_leakage)
                    )
                    for key in claim_mismatch
                }
            )
            if evidence_identity and evidence_identity.get("status") in {
                "ambiguous",
                "mismatch",
                "unlinked",
            }:
                check["explicit_target_mismatch"] = bool(
                    check["explicit_target_mismatch"]
                    or evidence_identity.get("status") == "mismatch"
                )
            check["multi_protein_leakage"] = multi_protein_leakage
            check["likely_prior_citation_only"] = _likely_prior_citation(
                quote
            )
    return verified


def _evidence_gate(evidence):
    origins = {item.get("evidence_origin", "") for item in evidence}
    sufficient = [item for item in evidence if _sufficient_exact_direct_evidence(item)]
    return {
        "all_quotes_exact": bool(evidence)
        and all(item.get("verification", {}).get("exact_match") for item in evidence),
        "all_quotes_exact_or_normalized_exact": bool(evidence)
        and all(
            item.get("verification", {}).get("exact_match")
            or item.get("verification", {}).get("normalized_exact_match")
            for item in evidence
        ),
        "all_claimed_sources_exact": bool(evidence)
        and all(
            item.get("verification", {}).get("claimed_source_exact")
            for item in evidence
        ),
        "all_claimed_sources_exact_or_normalized_exact": bool(evidence)
        and all(
            item.get("verification", {}).get("claimed_source_exact")
            or item.get("verification", {}).get("claimed_source_normalized_exact")
            for item in evidence
        ),
        "all_effective_sources_exact": bool(evidence)
        and all(
            item.get("verification", {}).get("effective_source_exact")
            or item.get("verification", {}).get("claimed_source_normalized_exact")
            for item in evidence
        ),
        "exact_target_grounding": bool(evidence)
        and all(
            item.get("verification", {}).get("exact_target_grounding")
            for item in evidence
        ),
        "direct_target_evidence_present": bool(origins & DIRECT_ORIGINS),
        "sufficient_exact_direct_evidence_present": bool(sufficient),
        "sufficient_exact_direct_evidence_count": len(sufficient),
        "likely_prior_citation_only": bool(evidence)
        and all(
            item.get("verification", {}).get("likely_prior_citation_only")
            for item in evidence
        ),
        "explicit_target_mismatch": any(
            item.get("verification", {}).get("explicit_target_mismatch")
            for item in evidence
        ),
        **{
            key: any(
                item.get("verification", {}).get(key)
                for item in evidence
            )
            for key in (
                "explicit_species_mismatch",
                "explicit_accession_mismatch",
                "explicit_homolog_mismatch",
                "explicit_gene_mismatch",
            )
        },
        "wrong_target_evidence_present": any(
            item.get("evidence_origin") == "different_or_related_protein"
            or item.get("verification", {}).get("explicit_target_mismatch")
            for item in evidence
        ),
        "layout_abstention_present": any(
            item.get("verification", {}).get("layout_requires_abstention")
            for item in evidence
        ),
        "model_identity_anchor_proposals_present": bool(evidence)
        and all("identity_anchor_ids" in item for item in evidence),
        "model_identity_anchors_verified": bool(evidence)
        and all(
            (
                item.get("model_identity_anchor_validation") is None
                or item.get("model_identity_anchor_validation", {}).get(
                    "accepted_as_verified_proposals"
                )
            )
            for item in evidence
        ),
        "evidence_origins": sorted(origins),
    }


def _sufficient_exact_direct_evidence(evidence):
    verification = evidence.get("verification", {})
    return bool(
        evidence.get("evidence_origin") in DIRECT_ORIGINS
        and (
            verification.get("exact_match")
            or verification.get("normalized_exact_match")
        )
        and (
            verification.get("effective_source_exact")
            or verification.get("claimed_source_normalized_exact")
        )
        and verification.get("exact_target_grounding")
        and not verification.get("likely_prior_citation_only")
        and not verification.get("explicit_target_mismatch")
        and not verification.get("layout_requires_abstention")
    )


def _quoted_text(item, direct_only=False, sufficient_only=False):
    return normalized_text(
        " ".join(
            str(value.get("quote") or "")
            for value in item.get("evidence", [])
            if (not sufficient_only or _sufficient_exact_direct_evidence(value))
            and (
                not direct_only
            or (
                value.get("evidence_origin") in DIRECT_ORIGINS
                and not value.get("verification", {}).get(
                    "likely_prior_citation_only", False
                )
            )
            )
        )
    )


def _normalize_oligomer_text(text):
    names = tuple(OLIGOMER_WORD_COUNTS)
    normalized = normalized_text(text)
    # Repair only unambiguous OCR joins such as "monomerdimer".
    joined = r"\b(%s)(?=(?:%s)\b)" % ("|".join(names), "|".join(names))
    while re.search(joined, normalized):
        normalized = re.sub(joined, r"\1 ", normalized)
    return re.sub(
        r"\bhomo[- ]?(%s)\b" % "|".join(names),
        r"\1",
        normalized,
    )


def _count_supported(count, text):
    text = _normalize_oligomer_text(text)
    if int(count or 0) == 0:
        return bool(
            re.search(r"\b(?:oligomer|multimer|assembl|associat|rings?)\w*\b", text)
        )
    names = [name for name, value in OLIGOMER_WORD_COUNTS.items() if value == int(count)]
    patterns = [r"\b(?:homo[- ]?)?%s(?:ic|s)?\b" % name for name in names]
    patterns.extend(
        [
            r"\b%s[- ]mer\b" % int(count),
            r"\b%s\s+(?:target\s+)?(?:protein\s+)?(?:subunits?|molecules?)\b"
            % int(count),
        ]
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _state_entailment(item):
    text = _quoted_text(item, sufficient_only=True)
    count = int(item.get("subunit_count", 0) or 0)
    label = normalized_text(item.get("state_label", ""))
    explicit_state = _count_supported(count, text)
    if count == 0:
        explicit_state = explicit_state or any(
            _term_in_text(term, text)
            for term in ("monomer", "dimer", "oligomer", "multimer", "heterogeneous")
            if _term_in_text(term, label)
        )
    return explicit_state


def _state_purification_gate(item):
    text = _normalize_oligomer_text(_quoted_text(item, sufficient_only=True))
    pooling_only = bool(
        PURIFICATION_POOLING_PATTERN.search(text)
        and not (
            DIRECT_STATE_MEASUREMENT_PATTERN.search(text)
            and STATE_VERIFICATION_ASSERTION_PATTERN.search(text)
        )
    )
    return {
        "purification_fraction_only": pooling_only,
        "direct_state_measurement_supported": bool(
            DIRECT_STATE_MEASUREMENT_PATTERN.search(text)
            and STATE_VERIFICATION_ASSERTION_PATTERN.search(text)
        ),
    }


def _event_entailment(item):
    text = _normalize_oligomer_text(_quoted_text(item, sufficient_only=True))
    counts = item.get("from_target_subunit_counts", []) + item.get(
        "to_target_subunit_counts", []
    )
    positive_counts = {int(value) for value in counts if int(value) > 0}
    endpoints_supported = all(_count_supported(count, text) for count in positive_counts)
    change_supported = any(re.search(pattern, text) for pattern in EVENT_CHANGE_PATTERNS)
    unknown_only = bool(counts) and not positive_counts
    explicit_unknown_endpoint = bool(
        0 in counts
        and positive_counts
        and re.search(
            r"\b(?:unknown|unresolved|undetermined|heterogeneous)\b|"
            r"\boligomeric\s+(?:state|species)\b",
            text,
        )
    )
    unknown_association = bool(
        unknown_only
        and ASSOCIATION_CHANGE_PATTERN.search(text)
        and not (
            CONFORMATION_ONLY_PATTERN.search(text)
            and not ASSOCIATION_CHANGE_PATTERN.search(text)
        )
    )
    stable_dimer_association = bool(
        "irreversible_assembly_or_disassembly" in item.get("event_types", [])
        and 2 in positive_counts
        and _count_supported(2, text)
        and STABLE_DIMER_ASSOCIATION_PATTERN.search(text)
    )
    return {
        "claim_entailment_supported": bool(
            (
                positive_counts
                and endpoints_supported
                and change_supported
                and (0 not in counts or explicit_unknown_endpoint)
            )
            or (explicit_unknown_endpoint and endpoints_supported and change_supported)
            or unknown_association
            or stable_dimer_association
        ),
        "unknown_stoichiometry_association_supported": unknown_association,
        "explicit_unknown_known_endpoint_supported": explicit_unknown_endpoint,
        "stable_dimer_irreversible_association_supported": stable_dimer_association,
    }


def _contradictory_evidence(item, field):
    claimed_counts = {
        int(value)
        for key in (
            ("subunit_count",) if field == "state_observations" else (
                "from_target_subunit_counts",
                "to_target_subunit_counts",
            )
        )
        for value in (
            [item.get(key, 0)] if key == "subunit_count" else item.get(key, [])
        )
        if int(value or 0) > 0
    }
    if not claimed_counts:
        return False
    for evidence in item.get("evidence", []):
        if evidence.get("evidence_origin") not in DIRECT_ORIGINS:
            continue
        verification = evidence.get("verification", {})
        if not (
            (
                verification.get("exact_match")
                or verification.get("normalized_exact_match")
            )
            and (
                verification.get("effective_source_exact")
                or verification.get("claimed_source_normalized_exact")
            )
        ):
            continue
        text = _normalize_oligomer_text(evidence.get("quote", ""))
        for count in claimed_counts:
            names = [name for name, value in OLIGOMER_WORD_COUNTS.items() if value == count]
            state_pattern = r"(?:homo[- ]?)?(?:%s)" % "|".join(names)
            if re.search(
                r"\b(?:did|does|do|was|were|is|are|could|can)\s+not\b.{0,45}\b%s\b|"
                r"\bno\s+(?:detectable\s+)?%s\b"
                % (state_pattern, state_pattern),
                text,
            ):
                return True
    return False


def _event_scope_gate(item):
    text = _quoted_text(item, sufficient_only=True)
    excluded = item.get("trigger_type") in {
        "membrane_or_lipid",
        "nucleic_acid",
        "binding_partner",
    } or any(re.search(pattern, text, re.I) for pattern in EXCLUDED_EVENT_SCOPE_PATTERNS)
    separately_measured_free_target = bool(
        FREE_TARGET_PATTERN.search(text)
        and DIRECT_POPULATION_MEASUREMENT_PATTERN.search(text)
    )
    return {
        "excluded_assembly_scope_detected": excluded,
        "separately_measured_free_target_redistribution": separately_measured_free_target,
        "event_assembly_scope_supported": not excluded or separately_measured_free_target,
    }


def _event_speculation_gate(item):
    text = _quoted_text(item, sufficient_only=True)
    speculative = bool(SPECULATIVE_EVENT_PATTERN.search(text))
    directly_measured = bool(DIRECT_POPULATION_MEASUREMENT_PATTERN.search(text))
    return {
        "speculative_event_claim": speculative,
        "direct_population_measurement_support": directly_measured,
        "non_speculative_or_directly_measured": not speculative or directly_measured,
    }


def _combines_distinct_variant_responses(item):
    text = _quoted_text(item)
    variant_tokens = set(
        re.findall(
            r"\b(?:wild[- ]?type|wt|[A-Z][0-9]{1,5}[A-Z]|"
            r"[A-Za-z0-9-]+(?:mutant|variant|construct|truncation))\b",
            " ".join(str(e.get("quote") or "") for e in item.get("evidence", [])),
            re.I,
        )
    )
    contrast = bool(
        re.search(
            r"\b(?:whereas|while|but|in contrast|respectively|unlike)\b",
            text,
        )
    )
    distinct_responses = bool(
        re.search(
            r"\b(?:increase|decrease|enhance|reduce|promote|inhibit|stabiliz|disrupt)\w*\b"
            r".{0,140}\b(?:whereas|while|but|in contrast|respectively|unlike)\b"
            r".{0,140}\b(?:increase|decrease|enhance|reduce|promote|inhibit|stabiliz|disrupt|"
            r"unchanged|no effect)\w*\b",
            text,
        )
    )
    construct = normalized_text(item.get("construct", ""))
    combined_construct = bool(re.search(r"\b(?:and|versus|vs\.?|/)\b", construct))
    return len({normalized_text(value) for value in variant_tokens}) >= 2 and contrast and (
        distinct_responses or combined_construct
    )


def _concentration_warnings(item):
    warnings = []
    for value in item.get("quantitative_values", []):
        unit = normalized_text(value.get("unit", ""))
        if unit != "mm":
            continue
        raw = value.get("numeric_value") or value.get("value_text")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw or "").replace(",", ""))
        if match and float(match.group()) >= 100:
            warnings.append(
                "suspicious_concentration_unit: preserved extracted mM value %s"
                % value.get("value_text", raw)
            )
    return warnings


def _asu_only(item):
    evidence = " ".join(
        str(value.get("quote") or "") for value in item.get("evidence", [])
    )
    combined = normalized_text(
        " ".join(
            [
                str(item.get("state_label") or ""),
                str(item.get("method") or ""),
                str(item.get("population_text") or ""),
                evidence,
            ]
        )
    )
    return "asymmetric unit" in combined and not any(
        phrase in combined
        for phrase in (
            "in solution",
            "solution state",
            "biological assembly",
            "physiological",
            "sedimentation",
            "sec-mals",
            "light scattering",
        )
    )


def _construct_relation_gate(item):
    """Require the v3 construct declaration to match deterministic graph paths."""
    relation = item.get("construct_relation")
    if relation is None:
        return {
            "construct_relation_declared": False,
            "construct_parent_supported": True,
        }
    resolution = item.get("identity_resolution") or {}
    if resolution.get("status") == "legacy_identity_grounding":
        return {
            "construct_relation_declared": True,
            "construct_relation": relation,
            "deterministic_construct_relations": [],
            "construct_parent_supported": True,
            "legacy_construct_grounding": True,
        }
    observed = {
        str(step.get("relation_type") or "")
        for step in resolution.get("anchor_path") or []
    }
    expected = {
        "exact_variant": {"variant_of"},
        "exact_domain": {"domain_of"},
        "truncation": {"truncation_of"},
        "fragment": {"fragment_of"},
        "processed_product": {"processed_product_of"},
        "other_construct": {"construct_of"},
    }
    construct = str(item.get("construct") or "")
    evidence_text = " ".join(
        str(evidence.get("quote") or "") for evidence in item.get("evidence", [])
    )
    mutation_or_subprotein_signal = bool(
        re.search(
            r"\b(?:mutant|mutation|variant|domain|fragment|truncat\w*|"
            r"delet\w*|residues?|amino acids?)\b|"
            r"(?:\b[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\b|\b[A-Z]\d+[A-Z]\b|Δ\d+)",
            construct + " " + evidence_text,
        )
    )
    exact_alias_steps = [
        step
        for step in resolution.get("anchor_path") or []
        if step.get("relation_type") == "alias_of"
        and normalized_text(step.get("mention", "")) == normalized_text(construct)
        and step.get("anchor_source_id")
        and step.get("anchor_quote")
    ]
    corrected_exact_alias_variant = bool(
        relation == "exact_variant"
        and resolution.get("status") == "exact"
        and normalized_text(construct)
        and exact_alias_steps
        and not mutation_or_subprotein_signal
    )
    if relation == "full_length":
        supported = resolution.get("status") == "exact" and not (
            observed
            & {
                "construct_of",
                "variant_of",
                "domain_of",
                "truncation_of",
                "fragment_of",
                "processed_product_of",
            }
        )
    elif relation == "unresolved":
        supported = False
    else:
        supported = bool(
            (
                resolution.get("status") == "construct_of_exact"
                or (
                    resolution.get("status") == "exact"
                    and resolution.get("construct_scoped_claim")
                )
            )
            and resolution.get("construct_scoped_claim")
            and observed & expected.get(relation, set())
        ) or corrected_exact_alias_variant
    result = {
        "construct_relation_declared": True,
        "construct_relation": relation,
        "deterministic_construct_relations": sorted(observed),
        "construct_parent_supported": supported,
    }
    if corrected_exact_alias_variant:
        result.update(
            {
                "construct_relation_corrected": True,
                "construct_relation_effective": "full_length_exact_alias",
                "construct_relation_correction_warning": (
                    "exact_variant corrected to exact alias: no mutation, domain, "
                    "fragment, or truncation signal"
                ),
            }
        )
    return result


def _rejection_reasons(field, item, gate):
    reasons = []
    identity = item.get("identity_resolution") or {}
    identity_status = identity.get("status")
    identity_reason = identity.get("reason", "")
    if identity_status == "ambiguous":
        reasons.append("alias_ambiguous_multiple_targets")
    elif identity_status == "unlinked":
        reasons.append(
            "species_unresolved"
            if "species" in identity_reason
            else "target_alias_unlinked"
        )
    elif identity_status == "mismatch":
        reasons.append(
            "species_mismatch"
            if "species" in identity_reason
            else "related_protein_only"
        )
    if gate.get("wrong_target_evidence_present"):
        reasons.append("related_protein_only")
    if gate.get("layout_abstention_present"):
        reasons.append("quote_noncontiguous/layout_abstention")
    if not gate.get("model_identity_anchors_verified", True):
        anchor_records = [
            evidence.get("model_identity_anchor_validation") or {}
            for evidence in item.get("evidence", [])
        ]
        if any(
            any(
                mapping.get("known")
                and mapping.get("target_id")
                and not mapping.get("maps_to_focus_target")
                for mapping in record.get("target_mappings") or []
            )
            for record in anchor_records
        ):
            reasons.append("related_protein_only")
        else:
            reasons.append("target_alias_unlinked")
    if not gate.get("construct_parent_supported", True):
        reasons.append("construct_parent_unproven")
    if (
        field in {"state_observations", "dynamic_events"}
        and not gate.get("single_evidence_method_result_link_supported", True)
    ):
        reasons.append("method_result_link_unproven")
    for flag, reason in (
        ("likely_prior_citation_only", "prior_citation_only"),
        ("purification_fraction_only", "purification_only"),
        ("asymmetric_unit_only", "asymmetric_unit_only"),
        ("contradictory_evidence_present", "contradictory_evidence"),
        ("speculative_event_claim", "speculative"),
        ("excluded_assembly_scope_detected", "event_scope_excluded"),
        ("combines_distinct_variant_responses", "construct_responses_combined"),
    ):
        if gate.get(flag):
            reasons.append(reason)
    if not gate.get("claim_entailment_supported", True):
        reasons.append("claim_not_entailed")
    if gate.get("duplicate_of_event_index") is not None:
        reasons.append("duplicate_event")
    if not gate.get("pathway_consistent", True):
        reasons.append("pathway_inconsistent")
    return sorted(set(reasons))


def _signature(event):
    return (
        tuple(sorted(event.get("event_types", []))),
        tuple(event.get("from_target_subunit_counts", [])),
        tuple(event.get("to_target_subunit_counts", [])),
        tuple(normalized_text(value) for value in event.get("from_states", [])),
        tuple(normalized_text(value) for value in event.get("to_states", [])),
        normalized_text(event.get("trigger_type", "")),
        normalized_text(event.get("trigger_name", "")),
        normalized_text(event.get("construct", "")),
        normalized_text(event.get("method", "")),
        tuple(
            sorted(normalized_text(value) for value in event.get("compared_conditions", []))
        ),
    )


def _pathway_reports(events):
    members = {}
    for index, event in enumerate(events):
        pathway_id = event.get("pathway_id", "")
        if pathway_id:
            members.setdefault(pathway_id, []).append(index)
    reports = []
    for pathway_id in sorted(members):
        indices = sorted(
            members[pathway_id],
            key=lambda index: (events[index].get("step_index", 0), index),
        )
        steps = [events[index].get("step_index", 0) for index in indices]
        consecutive = steps == list(range(1, len(indices) + 1))
        edges = []
        for left_index, right_index in zip(indices, indices[1:]):
            left = events[left_index]
            right = events[right_index]
            count_connection = bool(
                set(left.get("to_target_subunit_counts", []))
                & set(right.get("from_target_subunit_counts", []))
            )
            state_connection = bool(
                {normalized_text(value) for value in left.get("to_states", [])}
                & {normalized_text(value) for value in right.get("from_states", [])}
            )
            edges.append(count_connection and state_connection)
        consistent = consecutive and all(edges)
        reports.append(
            {
                "pathway_id": pathway_id,
                "event_indices": indices,
                "step_indices": steps,
                "steps_consecutive_from_one": consecutive,
                "adjacent_endpoints_connect": all(edges),
                "consistent": consistent,
            }
        )
    return reports


def curate_result(verified):
    """Return a curated copy with deterministic acceptance decisions."""
    curated = copy.deepcopy(verified)
    for field, _, item in _iter_items(curated):
        gate = _evidence_gate(item.get("evidence", []))
        identity_resolution = item.get("identity_resolution") or {}
        identity_status = identity_resolution.get(
            "status", "legacy_identity_grounding"
        )
        legacy_identity = identity_status == "legacy_identity_grounding"
        gate["identity_grounding_status"] = identity_status
        gate["legacy_identity_grounding"] = legacy_identity
        gate["typed_identity_grounding_sufficient"] = bool(
            legacy_identity or _typed_identity_sufficient(identity_resolution)
        )
        gate["asymmetric_unit_only"] = _asu_only(item) if field in {
            "state_observations",
            "dynamic_events",
        } else False
        if field == "state_observations":
            gate["claim_entailment_supported"] = _state_entailment(item)
            gate["single_evidence_method_result_link_supported"] = any(
                _state_entailment(dict(item, evidence=[evidence]))
                for evidence in item.get("evidence", [])
                if _sufficient_exact_direct_evidence(evidence)
            )
            gate.update(_state_purification_gate(item))
        elif field == "dynamic_events":
            gate.update(_event_entailment(item))
            gate["single_evidence_method_result_link_supported"] = any(
                _event_entailment(dict(item, evidence=[evidence]))[
                    "claim_entailment_supported"
                ]
                for evidence in item.get("evidence", [])
                if _sufficient_exact_direct_evidence(evidence)
            )
            gate.update(_event_scope_gate(item))
            gate.update(_event_speculation_gate(item))
            gate["combines_distinct_variant_responses"] = (
                _combines_distinct_variant_responses(item)
            )
        else:
            gate["claim_entailment_supported"] = True
        if field in {"state_observations", "dynamic_events"}:
            gate.update(_construct_relation_gate(item))
        gate["curation_warnings"] = _concentration_warnings(item)
        if gate.get("construct_relation_correction_warning"):
            gate["curation_warnings"].append(
                gate["construct_relation_correction_warning"]
            )
        gate["contradictory_evidence_present"] = _contradictory_evidence(item, field)
        gate["accepted_for_direct_analysis"] = bool(
            gate["sufficient_exact_direct_evidence_present"]
            and gate["typed_identity_grounding_sufficient"]
            and not gate["likely_prior_citation_only"]
            and not gate["explicit_target_mismatch"]
            and not gate["wrong_target_evidence_present"]
            and not gate["contradictory_evidence_present"]
            and gate["claim_entailment_supported"]
            and gate.get("event_assembly_scope_supported", True)
            and gate.get("non_speculative_or_directly_measured", True)
            and not gate.get("purification_fraction_only", False)
            and not gate.get("combines_distinct_variant_responses", False)
            and not gate.get("layout_abstention_present", False)
            and gate.get("model_identity_anchors_verified", True)
            and gate.get("construct_parent_supported", True)
            and gate.get("single_evidence_method_result_link_supported", True)
            and not gate["asymmetric_unit_only"]
            and field != "excluded_claims"
        )
        item["curation_validation"] = gate

    events = curated.get("dynamic_events", [])
    seen = {}
    for index, event in enumerate(events):
        counts = event.get("from_target_subunit_counts", []) + event.get(
            "to_target_subunit_counts", []
        )
        positives = {int(value) for value in counts if int(value) > 0}
        states_differ = {
            normalized_text(value) for value in event.get("from_states", [])
        } != {normalized_text(value) for value in event.get("to_states", [])}
        unknown_endpoint = (
            0 in counts and any(value > 1 for value in positives) and states_differ
        )
        stoichiometry = len(positives) >= 2 or unknown_endpoint
        reversible = not (
            "reversible_equilibrium" in event.get("event_types", [])
            and event.get("reversible") != "yes"
        )
        signature = _signature(event)
        duplicate_of = seen.get(signature)
        if duplicate_of is None:
            seen[signature] = index
        validation = event["curation_validation"]
        unknown_association = validation.get(
            "unknown_stoichiometry_association_supported", False
        )
        stoichiometry = stoichiometry or unknown_association
        validation.update(
            {
                "target_homo_stoichiometry_redistribution_supported": stoichiometry,
                "reversible_equilibrium_consistent": reversible,
                "duplicate_of_event_index": duplicate_of,
            }
        )
        validation["accepted_for_direct_analysis"] = bool(
            validation["accepted_for_direct_analysis"]
            and stoichiometry
            and reversible
            and duplicate_of is None
        )

    reports = _pathway_reports(events)
    for report in reports:
        if not report["consistent"]:
            for index in report["event_indices"]:
                events[index]["curation_validation"]["pathway_consistent"] = False
                events[index]["curation_validation"][
                    "accepted_for_direct_analysis"
                ] = False
        else:
            for index in report["event_indices"]:
                events[index]["curation_validation"]["pathway_consistent"] = True
    for event in events:
        if not event.get("pathway_id"):
            standalone = event.get("step_index") == 0
            event["curation_validation"]["pathway_consistent"] = standalone
            if not standalone:
                event["curation_validation"]["accepted_for_direct_analysis"] = False

    curated["event_graph_validation"] = {
        "n_model_events": len(events),
        "n_accepted_unique_direct_events": sum(
            item["curation_validation"]["accepted_for_direct_analysis"]
            for item in events
        ),
        "duplicate_event_indices": [
            index
            for index, item in enumerate(events)
            if item["curation_validation"]["duplicate_of_event_index"] is not None
        ],
        "pathways": reports,
    }
    curated["direct_target_state_evidence"] = any(
        item.get("curation_validation", {}).get("accepted_for_direct_analysis")
        for field in ("state_observations", "dynamic_events")
        for item in curated.get(field, [])
    )
    for field, _, item in _iter_items(curated):
        gate = item.get("curation_validation") or {}
        reasons = _rejection_reasons(field, item, gate)
        gate["rejection_reasons"] = reasons
        gate["abstention_reasons"] = [
            reason
            for reason in reasons
            if reason
            in {
                "target_alias_unlinked",
                "alias_ambiguous_multiple_targets",
                "species_unresolved",
                "construct_parent_unproven",
                "method_result_link_unproven",
                "quote_noncontiguous/layout_abstention",
            }
        ]
    return curated


def build_layers(
    model_output,
    passages,
    group,
    identity_graph=None,
    schema_version=SCHEMA_VERSION,
    schema_sha256_value=None,
):
    """Build separate parsed, verified, and curated layers."""
    parsed = json.loads(model_output) if isinstance(model_output, str) else copy.deepcopy(
        model_output
    )
    if not isinstance(parsed, dict):
        raise ValueError("model output must be a JSON object")
    validate_parsed_output(
        parsed,
        schema_version=schema_version,
        expected_schema_sha256=schema_sha256_value,
    )
    verified = verify_result(
        parsed,
        passages,
        group,
        identity_graph=identity_graph,
    )
    curated = curate_result(verified)
    return {"parsed": parsed, "verified": verified, "curated": curated}
