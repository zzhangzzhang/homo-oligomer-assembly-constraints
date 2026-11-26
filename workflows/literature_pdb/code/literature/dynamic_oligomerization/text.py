"""Original-paper text extraction and target-prioritized block retrieval.

Offsets in this module are zero-based, end-exclusive offsets into a normalized
source block.  Retrieval windows are always contiguous substrings of exactly
one block; adjacent blocks are returned separately so a downstream extractor
can never mistake reconstructed reading order for verbatim source text.
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from .aliases import (
    MAX_COMPETITOR_TERMS,
    MAX_MATCH_LABEL_CHARS,
    MAX_RETRIEVAL_TERMS,
    bounded_match_labels,
)
from .prompt import target_terms_with_audit


MAX_BLOCK_CHARS = 4000
MAX_BLOCKS_PER_FAMILY = 6
MAX_SELECTED_BLOCKS = 24
MAX_CONTEXT_SENTENCES = 2
MAX_NEIGHBOR_BLOCKS = 1

RETRIEVAL_FAMILIES = {
    "state_dynamics": (
        "monomer", "dimer", "trimer", "tetramer", "oligomer", "multimer",
        "equilibrium", "coexist", "mixture", "population", "fraction",
        "associate", "dissociate", "assembly", "stoichiometry",
    ),
    "methods_quantitative": (
        "sec-mals", "size-exclusion", "gel filtration", "native ms",
        "mass spectrometry", "ultracentrifug", "sedimentation", "saxs",
        "light scattering", "cross-link", "molecular weight",
        "dissociation constant", "concentration",
    ),
    "conditions": (
        "ligand", "substrate", "inhibitor", "cofactor", "ph ", "acidic",
        "alkaline", "salt", "metal", "redox", "mutation", "temperature",
        "membrane", "lipid", "dna", "rna", "binding partner",
    ),
    "function_disease_therapy": (
        "activity", "regulat", "alloster", "signal", "disease", "pathogenic",
        "cancer", "phenotype", "therapeut", "inhibitor", "treatment",
    ),
}
IDENTITY_CONSTRUCT_TERMS = (
    "construct", "full-length", "domain", "truncat", "mutant", "mutation",
    "variant", "isoform", "tagged", "residues", "expressed", "purified",
)
STRONG_PHRASES = (
    "equilibrium between", "dynamic equilibrium", "coexist", "mixture of",
    "small fraction", "predominantly monomer", "dimer in solution",
    "monomeric in solution", "dissociate into", "concentration-dependent",
    "shifted the equilibrium", "switch between",
)
CONTEXT_RESULT_TERMS = RETRIEVAL_FAMILIES["state_dynamics"] + (
    "observed", "showed", "revealed", "result", "corresponded", "consistent with",
)
# These are ranking penalties, not filters.  Candidate text and provenance are
# retained unchanged even when a competing protein is mentioned.
NEGATIVE_COMPETITOR_TERMS = (
    "unrelated protein", "other protein", "binding partner", "heterolog",
    "homolog", "ortholog", "competitor", "control protein",
)


def normalize_text(text):
    for old, new in {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u00ad": "",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentence_spans(text):
    """Return deterministic, one-based sentence spans without changing text."""
    spans = []
    for match in re.finditer(r"\S.*?(?:[.!?]+(?=\s|$)|\Z)", text, re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans or ([(0, len(text))] if text else [])


def _provenance(page, block, sentence_start, sentence_end, char_start, char_end):
    return {
        "page": str(page),
        "block": int(block),
        "sentence_start": int(sentence_start),
        "sentence_end": int(sentence_end),
        "char_start": int(char_start),
        "char_end": int(char_end),
        "offset_basis": "normalized_block_text",
    }


def _layout_diagnostics(blocks, page_width):
    """Describe block geometry without rejecting ordinary two-column pages."""
    geometries = []
    midpoint = float(page_width or 0.0) / 2.0
    for index, block in enumerate(blocks):
        bbox = tuple(float(value) for value in block[:4])
        x0, y0, x1, y1 = bbox
        if midpoint and x1 <= midpoint * 1.12:
            column = "left"
        elif midpoint and x0 >= midpoint * 0.88:
            column = "right"
        else:
            column = "spanning_or_unknown"
        geometries.append((index, bbox, column, y0, y1))

    left = [item for item in geometries if item[2] == "left"]
    right = [item for item in geometries if item[2] == "right"]
    two_column = any(
        max(left_item[3], right_item[3]) < min(left_item[4], right_item[4])
        for left_item in left
        for right_item in right
    )
    diagnostics = []
    for index, bbox, column, _, _ in geometries:
        diagnostics.append(
            {
                "bbox": [round(value, 3) for value in bbox],
                "column": column,
                "extraction_index": index + 1,
                "page_width": round(float(page_width or 0.0), 3),
                "page_has_separate_columns": two_column,
                "reasons": [],
                "requires_abstention": False,
            }
        )
    return diagnostics


def _dict_line(line, index):
    spans = line.get("spans") or []
    text = normalize_text("".join(str(span.get("text") or "") for span in spans))
    bbox = line.get("bbox")
    if not bbox and spans:
        boxes = [span.get("bbox") for span in spans if span.get("bbox")]
        if boxes:
            bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
    if not text or not bbox or len(bbox) < 4:
        return None
    return {
        "text": text,
        "bbox": tuple(float(value) for value in bbox[:4]),
        "line_index": int(index),
    }


def _coordinate_column_groups(block, page_width):
    """Return left/right line groups when one block contains both columns."""
    lines = [
        record
        for index, line in enumerate(block.get("lines") or [])
        for record in [_dict_line(line, index)]
        if record is not None
    ]
    if len(lines) < 4 or not page_width:
        return None
    block_bbox = block.get("bbox") or ()
    midpoint = float(page_width) / 2.0
    if len(block_bbox) < 4 or not (
        float(block_bbox[0]) < midpoint * 0.8
        and float(block_bbox[2]) > midpoint * 1.2
    ):
        return None

    starts = sorted({line["bbox"][0] for line in lines})
    if len(starts) < 2:
        return None
    gaps = [
        (right - left, left, right)
        for left, right in zip(starts, starts[1:])
    ]
    gap, left_edge, right_edge = max(gaps, key=lambda item: (item[0], -item[1]))
    if gap < max(18.0, float(page_width) * 0.12):
        return None
    split = (left_edge + right_edge) / 2.0
    left = [line for line in lines if line["bbox"][0] <= split]
    right = [line for line in lines if line["bbox"][0] > split]
    if min(len(left), len(right)) < 2:
        return None

    # A line crossing deeply into the other column cannot be assigned without
    # inventing reading order. Preserve the block and require abstention.
    gutter = midpoint
    ambiguous = [
        line
        for line in lines
        if line["bbox"][0] < gutter - 4 and line["bbox"][2] > gutter + 4
    ]
    return {
        "left": sorted(left, key=lambda line: (line["bbox"][1], line["bbox"][0], line["line_index"])),
        "right": sorted(right, key=lambda line: (line["bbox"][1], line["bbox"][0], line["line_index"])),
        "ambiguous": ambiguous,
        "split_x": split,
    }


def _append_pdf_passage(
    passages,
    text,
    page_number,
    block_number,
    source_id,
    base_source_id,
    diagnostics,
    extra_provenance=None,
    layout_transform=None,
    requires_abstention=False,
):
    text = normalize_text(text)
    if len(text) < 30:
        return
    sentences = _sentence_spans(text)
    provenance = _provenance(
        page_number, block_number, 1, len(sentences), 0, len(text)
    )
    provenance["base_source_id"] = base_source_id
    if extra_provenance:
        provenance.update(extra_provenance)
    passage = {
        "source_id": source_id,
        "base_source_id": base_source_id,
        "section": "unknown",
        "page": str(page_number),
        "block": block_number,
        "block_index": block_number,
        "sentence_start": 1,
        "sentence_end": len(sentences),
        "char_start": 0,
        "char_end": len(text),
        "provenance": provenance,
        "layout_diagnostics": diagnostics,
        "requires_abstention": bool(requires_abstention),
        "text": text,
    }
    if layout_transform:
        passage["layout_transform"] = layout_transform
        passage["provenance"]["layout_transform"] = layout_transform
    passages.append(passage)


def extract_pdf(path):
    """Extract stable PDF blocks; PyMuPDF remains an optional dependency."""
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires PyMuPDF") from exc
    passages = []
    with fitz.open(Path(path)) as document:
        for page_index, page in enumerate(document):
            page_width = getattr(getattr(page, "rect", None), "width", 0.0)
            try:
                page_dict = page.get_text("dict")
            except Exception:
                page_dict = None
            dict_blocks = (
                list(page_dict.get("blocks") or [])
                if isinstance(page_dict, dict)
                else []
            )
            if dict_blocks:
                page_number = page_index + 1
                for block_index, block in enumerate(dict_blocks):
                    if block.get("type", 0) != 0:
                        continue
                    block_number = block_index + 1
                    base_source_id = "page%d.block%d" % (
                        page_number, block_number
                    )
                    groups = _coordinate_column_groups(block, page_width)
                    block_bbox = [
                        round(float(value), 3)
                        for value in (block.get("bbox") or (0, 0, 0, 0))[:4]
                    ]
                    if groups and not groups["ambiguous"]:
                        for suffix, column, lines in (
                            ("colL", "left", groups["left"]),
                            ("colR", "right", groups["right"]),
                        ):
                            source_id = "%s.%s" % (base_source_id, suffix)
                            line_boxes = [
                                [round(value, 3) for value in line["bbox"]]
                                for line in lines
                            ]
                            diagnostic = {
                                "bbox": block_bbox,
                                "column": column,
                                "extraction_index": block_number,
                                "page_width": round(float(page_width or 0.0), 3),
                                "reasons": [],
                                "requires_abstention": False,
                                "layout_transform": "coordinate_column_split",
                                "source_line_indices": [
                                    line["line_index"] + 1 for line in lines
                                ],
                                "source_line_bboxes": line_boxes,
                            }
                            _append_pdf_passage(
                                passages,
                                " ".join(line["text"] for line in lines),
                                page_number,
                                block_number,
                                source_id,
                                base_source_id,
                                diagnostic,
                                extra_provenance={
                                    "original_block_source_id": base_source_id,
                                    "source_line_indices": diagnostic[
                                        "source_line_indices"
                                    ],
                                    "source_line_bboxes": line_boxes,
                                    "offset_basis": "normalized_coordinate_column_text",
                                },
                                layout_transform="coordinate_column_split",
                            )
                        continue

                    lines = [
                        record
                        for index, line in enumerate(block.get("lines") or [])
                        for record in [_dict_line(line, index)]
                        if record is not None
                    ]
                    text = " ".join(line["text"] for line in lines)
                    unresolved = bool(groups and groups["ambiguous"])
                    reasons = (
                        ["spanning_block_unresolved_column_interleaving"]
                        if unresolved
                        else []
                    )
                    diagnostic = {
                        "bbox": block_bbox,
                        "column": "spanning_or_unknown",
                        "extraction_index": block_number,
                        "page_width": round(float(page_width or 0.0), 3),
                        "reasons": reasons,
                        "requires_abstention": unresolved,
                    }
                    _append_pdf_passage(
                        passages,
                        text,
                        page_number,
                        block_number,
                        base_source_id,
                        base_source_id,
                        diagnostic,
                        requires_abstention=unresolved,
                    )
                continue

            blocks = list(page.get_text("blocks"))
            diagnostics = _layout_diagnostics(blocks, page_width)
            for block_index, block in enumerate(blocks):
                page_number = page_index + 1
                block_number = block_index + 1
                source_id = "page%d.block%d" % (page_number, block_number)
                _append_pdf_passage(
                    passages,
                    str(block[4]),
                    page_number,
                    block_number,
                    source_id,
                    source_id,
                    diagnostics[block_index],
                )
    return passages


def _local_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def _xml_page(element, inherited):
    for key in ("page", "page-number", "fpage"):
        if element.get(key):
            return str(element.get(key))
    return inherited


def extract_xml(path):
    """Extract JATS-like XML while retaining section and element provenance."""
    root = ET.parse(Path(path)).getroot()
    passages = []
    counters = {}

    def append_element(element, tag, section, page, element_path):
        text = normalize_text(" ".join(element.itertext()))
        if len(text) < 30:
            return
        counters[tag] = counters.get(tag, 0) + 1
        block_number = len(passages) + 1
        sentences = _sentence_spans(text)
        provenance = _provenance(
            page, block_number, 1, len(sentences), 0, len(text)
        )
        provenance.update(
            {
                "element_path": element_path,
                "element_tag": tag,
            }
        )
        source_id = "xml.%s%d" % (tag, counters[tag])
        provenance["base_source_id"] = source_id
        passages.append(
            {
                "source_id": source_id,
                "base_source_id": source_id,
                "section": section,
                "page": str(page),
                "block": block_number,
                "block_index": block_number,
                "sentence_start": 1,
                "sentence_end": len(sentences),
                "char_start": 0,
                "char_end": len(text),
                "provenance": provenance,
                "element_path": element_path,
                "requires_abstention": False,
                "text": text,
            }
        )

    def walk(element, section, page, element_path):
        tag = _local_name(element.tag)
        page = _xml_page(element, page)
        if tag == "abstract":
            section = "abstract"
            if not any(_local_name(child.tag) == "p" for child in element.iter()):
                append_element(element, tag, section, page, element_path)
                return
        elif tag == "sec":
            title = next(
                (
                    normalize_text(" ".join(child.itertext()))
                    for child in element
                    if _local_name(child.tag) == "title"
                ),
                "",
            )
            if title:
                section = title
        elif tag == "caption":
            append_element(
                element, tag, "figure_or_table_caption", page, element_path
            )
            return
        elif tag == "td":
            append_element(element, tag, "table", page, element_path)
            return
        elif tag == "p":
            append_element(element, tag, section, page, element_path)
            return

        child_counts = {}
        for child in element:
            child_tag = _local_name(child.tag)
            child_counts[child_tag] = child_counts.get(child_tag, 0) + 1
            child_path = "%s/%s[%d]" % (
                element_path,
                child_tag,
                child_counts[child_tag],
            )
            walk(child, section, page, child_path)

    walk(root, "unknown", "unknown", "/%s[1]" % _local_name(root.tag))
    return passages


def extract_text(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".xml", ".nxml"}:
        return extract_xml(path)
    raise ValueError("unsupported paper type: %s" % suffix)


def _literal_spans(text, phrases):
    lowered = text.lower()
    spans = []
    for phrase in phrases:
        start = 0
        phrase = str(phrase).lower()
        if not phrase:
            continue
        while True:
            position = lowered.find(phrase, start)
            if position < 0:
                break
            spans.append((position, position + len(phrase)))
            start = position + max(1, len(phrase))
    return sorted(set(spans))


@lru_cache(maxsize=4096)
def _target_pattern(term):
    """Cache one bounded, literal token-boundary pattern."""
    value = str(term).strip()
    if not (2 <= len(value) <= MAX_MATCH_LABEL_CHARS):
        return re.compile(r"(?!x)x")
    return re.compile(
        r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(value),
        re.IGNORECASE,
    )


def _target_spans(text, terms):
    """Match target aliases only as complete alphanumeric tokens/phrases."""
    spans = []
    bounded, _ = bounded_match_labels(terms, MAX_RETRIEVAL_TERMS)
    for term in bounded:
        spans.extend(
            (match.start(), match.end())
            for match in _target_pattern(term).finditer(text)
        )
    return sorted(set(spans))


class _TargetMatcher:
    """Reuse bounded patterns and passage scans throughout one retrieval."""

    def __init__(self, terms, limit):
        self.terms, self.audit = bounded_match_labels(terms, limit)
        self.patterns = tuple(_target_pattern(term) for term in self.terms)
        self._cache = {}

    def spans(self, text):
        text = str(text or "")
        if text not in self._cache:
            spans = [
                (match.start(), match.end())
                for pattern in self.patterns
                for match in pattern.finditer(text)
            ]
            self._cache[text] = sorted(set(spans))
        return self._cache[text]


def _competitor_penalty(text, competitor_terms=None, competitor_matcher=None):
    lowered = text.lower()
    generic = sum(lowered.count(term) for term in NEGATIVE_COMPETITOR_TERMS) * 8
    matcher = competitor_matcher or _TargetMatcher(
        competitor_terms or (), MAX_COMPETITOR_TERMS
    )
    explicit = len(matcher.spans(text)) * 16
    return generic + explicit


def _score(
    passage,
    keywords,
    terms,
    competitor_terms=None,
    target_matcher=None,
    competitor_matcher=None,
):
    """Compatibility scoring helper with boundary-safe target matching."""
    text = str(passage.get("text") or "")
    keyword_spans = _literal_spans(text, keywords)
    target_matcher = target_matcher or _TargetMatcher(terms, MAX_RETRIEVAL_TERMS)
    targets = target_matcher.spans(text)
    strong = _literal_spans(text, STRONG_PHRASES)
    return (
        len(keyword_spans) * 4
        + min(len(targets), 3) * 3
        + len(strong) * 25
        - _competitor_penalty(text, competitor_terms, competitor_matcher)
    )


def _nearest_span(primary, secondary):
    if not primary:
        return []
    if not secondary:
        return [primary[0]]
    pair = min(
        (
            (abs((left[0] + left[1]) - (right[0] + right[1])), left, right)
            for left in primary
            for right in secondary
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    return [pair[1], pair[2]]


def _base_source_id(passage):
    return str(passage.get("base_source_id") or passage.get("source_id") or "")


def _window_source_id(
    passage, char_start, char_end, sentence_start, sentence_end, full_passage
):
    if full_passage:
        return str(passage["source_id"])
    return "%s.window.c%d-%d.s%d-%d" % (
        passage["source_id"],
        char_start,
        char_end,
        sentence_start,
        sentence_end,
    )


def _bounded_window(text, evidence_spans):
    sentence_spans = _sentence_spans(text)
    if not evidence_spans:
        local_start = 0
        local_end = min(len(text), MAX_BLOCK_CHARS)
    else:
        evidence_spans = sorted(evidence_spans)
        covered = [
            index
            for index, (start, end) in enumerate(sentence_spans)
            if any(left < end and right > start for left, right in evidence_spans)
        ]
        anchor_index = covered[0] if covered else 0
        first = max(0, anchor_index - MAX_CONTEXT_SENTENCES)
        last = min(
            len(sentence_spans) - 1,
            (covered[-1] if covered else anchor_index) + MAX_CONTEXT_SENTENCES,
        )
        local_start = sentence_spans[first][0]
        local_end = sentence_spans[last][1]
        if local_end - local_start > MAX_BLOCK_CHARS:
            evidence_start = min(span[0] for span in evidence_spans)
            evidence_end = max(span[1] for span in evidence_spans)
            if evidence_end - evidence_start >= MAX_BLOCK_CHARS:
                local_start = evidence_start
                local_end = evidence_end
            else:
                padding = MAX_BLOCK_CHARS - (evidence_end - evidence_start)
                local_start = max(0, evidence_start - padding // 2)
                local_end = min(len(text), local_start + MAX_BLOCK_CHARS)
                local_start = max(0, local_end - MAX_BLOCK_CHARS)
                if local_start > evidence_start:
                    local_start = evidence_start
                if local_end < evidence_end:
                    local_end = evidence_end
    covered_sentences = [
        index
        for index, (start, end) in enumerate(sentence_spans)
        if start < local_end and end > local_start
    ]
    first_sentence = covered_sentences[0] if covered_sentences else 0
    last_sentence = covered_sentences[-1] if covered_sentences else 0
    return local_start, local_end, first_sentence, last_sentence


def _evidence_clusters(text, primary_spans, secondary_spans=None):
    """Cluster far-apart hits into separate, non-overlapping windows."""
    candidates = []
    secondary_spans = list(secondary_spans or [])
    for primary in sorted(set(primary_spans)):
        evidence = [primary]
        if secondary_spans:
            evidence.extend(_nearest_span(secondary_spans, [primary])[:1])
        evidence = sorted(set(evidence))
        start, end, _, _ = _bounded_window(text, evidence)
        candidates.append({"start": start, "end": end, "evidence": evidence})

    clusters = []
    for candidate in candidates:
        if (
            clusters
            and candidate["start"] < clusters[-1]["end"]
            and max(candidate["end"], clusters[-1]["end"])
            - min(candidate["start"], clusters[-1]["start"])
            <= MAX_BLOCK_CHARS
        ):
            clusters[-1]["start"] = min(
                clusters[-1]["start"], candidate["start"]
            )
            clusters[-1]["end"] = max(clusters[-1]["end"], candidate["end"])
            clusters[-1]["evidence"] = sorted(
                set(clusters[-1]["evidence"] + candidate["evidence"])
            )
        else:
            clusters.append(candidate)
    return [cluster["evidence"] for cluster in clusters]


def _window(passage, family, evidence_spans=None):
    """Return one contiguous, evidence-centered, source-auditable window."""
    text = str(passage.get("text") or "")
    evidence_spans = list(evidence_spans or [])
    local_start, local_end, first_sentence, last_sentence = _bounded_window(
        text, evidence_spans
    )

    base_char = int(passage.get("char_start", 0))
    char_start = base_char + local_start
    char_end = base_char + local_end
    source_sentence = int(passage.get("sentence_start", 1))
    sentence_start = source_sentence + first_sentence
    sentence_end = source_sentence + last_sentence
    block = int(passage.get("block", passage.get("block_index", 1)))
    page = str(passage.get("page", "unknown"))
    provenance = _provenance(
        page, block, sentence_start, sentence_end, char_start, char_end
    )
    original_provenance = passage.get("provenance") or {}
    base_source_id = _base_source_id(passage)
    provenance["base_source_id"] = base_source_id
    for key in (
        "element_path",
        "element_tag",
        "original_block_source_id",
        "source_line_indices",
        "source_line_bboxes",
        "layout_transform",
    ):
        if key in original_provenance:
            provenance[key] = original_provenance[key]
    full_passage = local_start == 0 and local_end == len(text)
    source_id = _window_source_id(
        passage,
        char_start,
        char_end,
        sentence_start,
        sentence_end,
        full_passage,
    )

    result = {
        "source_id": source_id,
        "base_source_id": base_source_id,
        "source_passage_id": str(passage["source_id"]),
        "section": passage.get("section", "unknown"),
        "page": page,
        "block": block,
        "block_index": block,
        "sentence_start": sentence_start,
        "sentence_end": sentence_end,
        "char_start": char_start,
        "char_end": char_end,
        "provenance": provenance,
        "retrieval_family": family,
        "retrieval_families": [family],
        "requires_abstention": bool(passage.get("requires_abstention", False)),
        "text": text[local_start:local_end],
    }
    for key in ("layout_diagnostics", "element_path", "layout_transform"):
        if key in passage:
            result[key] = passage[key]
    return result


def _rank_family(
    passages,
    keywords,
    terms,
    identity=False,
    competitor_terms=None,
    target_matcher=None,
    competitor_matcher=None,
):
    target_matcher = target_matcher or _TargetMatcher(
        terms, MAX_RETRIEVAL_TERMS
    )
    ranked = []
    for index, passage in enumerate(passages):
        text = str(passage.get("text") or "")
        targets = target_matcher.spans(text)
        keywords_found = _literal_spans(text, keywords)
        if identity:
            if not targets:
                continue
            score = len(targets) * 12 + len(keywords_found) * 3
            clusters = _evidence_clusters(text, targets, keywords_found)
        else:
            if not keywords_found:
                continue
            score = _score(
                passage,
                keywords,
                terms,
                competitor_terms,
                target_matcher,
                competitor_matcher,
            )
            clusters = _evidence_clusters(text, keywords_found, targets)
        if identity:
            score -= _competitor_penalty(
                text, competitor_terms, competitor_matcher
            )
        for evidence in clusters:
            ranked.append((score, index, evidence))
    return sorted(ranked, key=lambda item: (-item[0], item[1], item[2]))


def _identity_graph_terms(identity_graph):
    if not isinstance(identity_graph, dict):
        return [], []
    focus_id = identity_graph.get("focus_target_id")
    focus_terms = set()
    competitor_terms = set()
    edge_targets = {}
    for edge in identity_graph.get("edges") or []:
        edge_targets.setdefault(edge.get("source_node_id"), set()).add(
            (edge.get("target_identity") or {}).get("target_id")
        )
    for node in identity_graph.get("nodes") or []:
        labels = node.get("identity_labels") or [node.get("label", "")]
        target_ids = set(edge_targets.get(node.get("node_id"), set()))
        node_target = (node.get("target_identity") or {}).get("target_id")
        if node_target:
            target_ids.add(node_target)
        destination = (
            focus_terms
            if focus_id and target_ids == {focus_id}
            else competitor_terms
            if focus_id and target_ids and focus_id not in target_ids
            else None
        )
        if destination is not None:
            destination.update(str(label) for label in labels if str(label).strip())
    for target in identity_graph.get("targets") or []:
        destination = (
            focus_terms
            if target.get("target_id") == focus_id
            else competitor_terms
        )
        destination.update(target.get("species") or [])
    focus, _ = bounded_match_labels(focus_terms, MAX_RETRIEVAL_TERMS)
    competitors, _ = bounded_match_labels(
        competitor_terms, MAX_COMPETITOR_TERMS
    )
    return focus, competitors


def _identity_anchor_ranked(
    passages, identity_graph, competitor_terms, competitor_matcher=None
):
    anchors = {}
    for edge in (identity_graph or {}).get("edges") or []:
        source_id = str(edge.get("source_id") or "")
        quote = str(edge.get("anchor_quote") or "")
        if source_id and quote and not source_id.startswith("manifest:"):
            anchors.setdefault(source_id, set()).add(quote)
    ranked = []
    for index, passage in enumerate(passages):
        source_ids = {
            str(passage.get("source_id") or ""),
            _base_source_id(passage),
        }
        quotes = {
            quote
            for source_id in source_ids
            for quote in anchors.get(source_id, set())
        }
        if not quotes:
            continue
        text = str(passage.get("text") or "")
        spans = []
        for quote in sorted(quotes):
            start = text.find(quote)
            if start >= 0:
                spans.append((start, start + len(quote)))
        if spans:
            score = len(spans) * 100 - _competitor_penalty(
                text, competitor_terms, competitor_matcher
            )
            for evidence in _evidence_clusters(text, spans):
                ranked.append((score, index, evidence))
    return sorted(ranked, key=lambda item: (-item[0], item[1], item[2]))


def _context_family(passages, method_ranked, terms, target_matcher=None):
    target_matcher = target_matcher or _TargetMatcher(
        terms, MAX_RETRIEVAL_TERMS
    )
    candidates = {}
    for method_score, center_index, _ in method_ranked:
        for distance in range(1, MAX_NEIGHBOR_BLOCKS + 1):
            for neighbor_index in (center_index - distance, center_index + distance):
                if not 0 <= neighbor_index < len(passages):
                    continue
                text = str(passages[neighbor_index].get("text") or "")
                result_spans = _literal_spans(text, CONTEXT_RESULT_TERMS)
                if not result_spans:
                    continue
                target_matches = target_matcher.spans(text)
                evidence = _nearest_span(result_spans, target_matches)
                score = method_score + len(result_spans) * 5 - distance
                candidate = (score, neighbor_index, evidence)
                previous = candidates.get(neighbor_index)
                if previous is None or candidate > previous:
                    candidates[neighbor_index] = candidate
    return sorted(
        candidates.values(), key=lambda item: (-item[0], item[1], item[2])
    )


def _priority_bounded_terms(primary, secondary, limit):
    selected = []
    seen = set()
    valid_total = 0
    audits = []
    for values in (primary, secondary):
        bounded, audit = bounded_match_labels(values, limit)
        audits.append(audit)
        for value in bounded:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            valid_total += 1
            if len(selected) < limit:
                selected.append(value)
    truncated = max(0, valid_total - len(selected))
    return selected, {
        "input_label_count": sum(item["input_label_count"] for item in audits),
        "valid_unique_label_count": valid_total,
        "returned_label_count": len(selected),
        "truncated_label_count": truncated
        + sum(item["truncated_label_count"] for item in audits),
        "truncated": bool(truncated)
        or any(item["truncated"] for item in audits),
        "complete": not (
            truncated or any(item["truncated"] for item in audits)
        ),
        "limit": limit,
        "max_label_chars": MAX_MATCH_LABEL_CHARS,
    }


def retrieve_source_blocks(
    passages,
    group,
    identity_graph=None,
    competitor_terms=None,
    return_audit=False,
):
    """Select deterministic, provenance-preserving blocks with family coverage."""
    graph_terms, graph_competitors = _identity_graph_terms(identity_graph)
    manifest_terms, manifest_term_audit = target_terms_with_audit(group)
    terms, combined_term_audit = _priority_bounded_terms(
        manifest_terms, graph_terms, MAX_RETRIEVAL_TERMS
    )
    competitor_terms, competitor_term_audit = _priority_bounded_terms(
        competitor_terms or (),
        graph_competitors,
        MAX_COMPETITOR_TERMS,
    )
    target_matcher = _TargetMatcher(terms, MAX_RETRIEVAL_TERMS)
    competitor_matcher = _TargetMatcher(
        competitor_terms, MAX_COMPETITOR_TERMS
    )
    selected = []

    def add(window):
        overlapping = next(
            (
                existing
                for existing in selected
                if existing.get("source_passage_id", existing["source_id"])
                == window.get("source_passage_id", window["source_id"])
                and int(existing.get("char_start", 0))
                < int(window.get("char_end", 0))
                and int(window.get("char_start", 0))
                < int(existing.get("char_end", 0))
            ),
            None,
        )
        if overlapping is not None:
            existing = overlapping
            family = window["retrieval_family"]
            if family not in existing["retrieval_families"]:
                existing["retrieval_families"].append(family)
            # An overlapping evidence-centered window may replace a generic
            # front prefix, but ranges are never unioned: that would fabricate
            # a quote spanning text that was not selected contiguously.
            if (
                (
                    existing["retrieval_family"] == "front"
                    and family != "front"
                )
                or family == "identity_anchor"
            ):
                primary_family = existing["retrieval_family"]
                families = existing["retrieval_families"]
                existing.clear()
                existing.update(window)
                existing["retrieval_family"] = primary_family
                existing["retrieval_families"] = families
            return False
        selected.append(window)
        return True

    for passage in passages[:3]:
        add(_window(passage, "front"))

    state_method = RETRIEVAL_FAMILIES["state_dynamics"] + RETRIEVAL_FAMILIES[
        "methods_quantitative"
    ]
    ranked = {
        "identity_anchor": _identity_anchor_ranked(
            passages,
            identity_graph,
            competitor_terms,
            competitor_matcher=competitor_matcher,
        ),
        "identity_construct_anchor": _rank_family(
            passages,
            IDENTITY_CONSTRUCT_TERMS,
            terms,
            identity=True,
            competitor_terms=competitor_terms,
            target_matcher=target_matcher,
            competitor_matcher=competitor_matcher,
        ),
        "oligomerization_evidence": _rank_family(
            passages,
            state_method,
            terms,
            competitor_terms=competitor_terms,
            target_matcher=target_matcher,
            competitor_matcher=competitor_matcher,
        ),
    }
    for family, keywords in RETRIEVAL_FAMILIES.items():
        ranked[family] = _rank_family(
            passages,
            keywords,
            terms,
            competitor_terms=competitor_terms,
            target_matcher=target_matcher,
            competitor_matcher=competitor_matcher,
        )
    ranked["method_result_context"] = _context_family(
        passages,
        ranked["methods_quantitative"],
        terms,
        target_matcher=target_matcher,
    )

    # Round-robin selection prevents early broad families from consuming the
    # global budget.  A duplicate annotates the existing record, then the
    # cursor advances until the family quota is backfilled by a unique source.
    cursors = {family: 0 for family in ranked}
    family_counts = {family: 0 for family in ranked}
    while len(selected) < MAX_SELECTED_BLOCKS:
        progress = False
        for family, candidates in ranked.items():
            if len(selected) >= MAX_SELECTED_BLOCKS:
                break
            if family_counts[family] >= MAX_BLOCKS_PER_FAMILY:
                continue
            while cursors[family] < len(candidates):
                _, index, evidence = candidates[cursors[family]]
                cursors[family] += 1
                window = _window(passages[index], family, evidence)
                if add(window):
                    family_counts[family] += 1
                    progress = True
                    break
                progress = True
            # Exhaustion is normal for narrowly focused papers.
        if not progress:
            break
    # Identity anchors are audit prerequisites, not merely high-ranking
    # candidates. They may extend the normal retrieval budget, but remain
    # provenance-preserving source windows and are never selected by filtering
    # out competitor-bearing passages.
    for _, index, evidence in ranked["identity_anchor"]:
        add(_window(passages[index], "identity_anchor", evidence))
    candidate_windows = {
        family: [
            _window(passages[index], family, evidence)
            for _, index, evidence in candidates
        ]
        for family, candidates in ranked.items()
    }
    for window in selected:
        for family, candidates in candidate_windows.items():
            overlaps_candidate = any(
                candidate.get("source_passage_id", candidate["source_id"])
                == window.get("source_passage_id", window["source_id"])
                and int(candidate.get("char_start", 0))
                < int(window.get("char_end", 0))
                and int(window.get("char_start", 0))
                < int(candidate.get("char_end", 0))
                for candidate in candidates
            )
            if overlaps_candidate and family not in window["retrieval_families"]:
                window["retrieval_families"].append(family)
    audit = {
        "target_terms": {
            "manifest": manifest_term_audit,
            "combined": combined_term_audit,
            "matcher": target_matcher.audit,
            "truncated": bool(
                manifest_term_audit.get("truncated")
                or combined_term_audit.get("truncated")
                or target_matcher.audit.get("truncated")
            ),
        },
        "competitor_terms": {
            "combined": competitor_term_audit,
            "matcher": competitor_matcher.audit,
        },
        "selected_passage_count": len(selected),
    }
    audit["target_terms"]["complete"] = not audit["target_terms"]["truncated"]
    return (selected, audit) if return_audit else selected
