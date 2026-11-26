"""Retryable, resumable RCSB entity metadata retrieval."""

import json
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .aliases import normalize_gene_aliases


RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}

# Identity and concise construct fields only. In particular, this query does
# not request ligands, interfaces, or biological assemblies.
ENTITY_QUERY = """
query EntityTargets($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    polymer_entities {
      rcsb_id
      entity_poly {
        pdbx_strand_id
        type
        rcsb_sample_sequence_length
        rcsb_mutation_count
      }
      rcsb_polymer_entity {
        pdbx_description
        pdbx_mutation
      }
      rcsb_polymer_entity_container_identifiers {
        entity_id
        reference_sequence_identifiers {
          database_accession
          database_name
        }
      }
      entity_src_gen {
        pdbx_gene_src_scientific_name
        pdbx_gene_src_gene
      }
      entity_src_nat { pdbx_organism_scientific }
    }
  }
}
""".strip()


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _source(entity):
    generated = entity.get("entity_src_gen") or []
    natural = entity.get("entity_src_nat") or []
    organisms = set()
    raw_genes = []
    for item in generated:
        value = _text(item.get("pdbx_gene_src_scientific_name"))
        if value:
            organisms.add(value)
        value = _text(item.get("pdbx_gene_src_gene"))
        if value:
            raw_genes.append(value)
    for item in natural:
        value = _text(item.get("pdbx_organism_scientific"))
        if value:
            organisms.add(value)
    genes, gene_audit = normalize_gene_aliases(raw_genes)
    return sorted(organisms), genes, gene_audit


def parse_entry(entry):
    """Convert an RCSB GraphQL entry into stable, minimal entity records."""
    pdb_id = str(entry.get("rcsb_id") or "").upper()
    entities = []
    for entity in entry.get("polymer_entities") or []:
        ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        refs = ids.get("reference_sequence_identifiers") or []
        uniprots = sorted(
            {
                str(ref.get("database_accession")).strip().upper()
                for ref in refs
                if str(ref.get("database_name") or "").lower() == "uniprot"
                and _text(ref.get("database_accession"))
            }
        )
        polymer = entity.get("entity_poly") or {}
        identity = entity.get("rcsb_polymer_entity") or {}
        organisms, genes, gene_audit = _source(entity)
        chains = sorted(
            set(
                part.strip()
                for part in str(polymer.get("pdbx_strand_id") or "").split(",")
                if part.strip()
            )
        )
        entities.append(
            {
                "entity_id": _text(ids.get("entity_id")),
                "rcsb_entity_id": _text(entity.get("rcsb_id")),
                "polymer_type": _text(polymer.get("type")),
                "chains": chains,
                "description": _text(identity.get("pdbx_description")),
                "mutation_description": _text(identity.get("pdbx_mutation")),
                "sequence_length": polymer.get("rcsb_sample_sequence_length"),
                "mutation_count": polymer.get("rcsb_mutation_count"),
                "source_organisms": organisms,
                "source_genes": genes,
                "source_gene_alias_audit": gene_audit,
                "uniprot_accessions": uniprots,
            }
        )
    entities.sort(key=lambda item: (item["entity_id"] or "", item["rcsb_entity_id"] or ""))
    return {
        "pdb_id": pdb_id,
        "entry_title": _text((entry.get("struct") or {}).get("title")),
        "entry_polymer_entity_count": len(entities),
        "entry_distinct_uniprot_accessions": sorted(
            {
                accession
                for entity in entities
                for accession in entity["uniprot_accessions"]
            }
        ),
        "entities": entities,
    }


def post_graphql(url, query, variables, timeout):
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "dynamic-oligomerization-manifest/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, {"errors": [{"message": str(exc)}]}


def read_cache(path):
    records = {}
    cache = Path(path)
    if not cache.exists():
        return records
    with cache.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pdb_id = str(row.get("pdb_id") or "").upper()
            if not pdb_id:
                raise ValueError("cache line %d lacks pdb_id" % line_number)
            records[pdb_id] = row
    return records


def write_cache(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for pdb_id in sorted(records):
            handle.write(json.dumps(records[pdb_id], sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(destination)


def _request_batch(pdb_ids, url, timeout, retries, backoff, transport):
    last_error = None
    last_http = None
    for attempt in range(1, retries + 2):
        try:
            status, payload = transport(url, ENTITY_QUERY, {"ids": pdb_ids}, timeout)
            last_http = status
            errors = payload.get("errors") or []
            if status == 200 and not errors:
                return attempt, status, payload, None
            last_error = "; ".join(str(error.get("message") or error) for error in errors)
            # GraphQL commonly reports transient backend errors in a 200
            # response, so retry 200-with-errors as well as retryable HTTP.
            if status != 200 and status not in RETRYABLE_HTTP:
                break
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)
        if attempt <= retries:
            # Jitter avoids synchronized retries in parallel callers, while
            # output remains deterministic because delays are never persisted.
            time.sleep(backoff * (2 ** (attempt - 1)) * (0.9 + random.random() * 0.2))
    return attempt, last_http, None, last_error or "RCSB request failed"


def resolve_pdb_entities(
    pdb_ids,
    cache_path,
    batch_size=100,
    retries=3,
    timeout=30,
    backoff=1.0,
    url=RCSB_GRAPHQL_URL,
    transport=post_graphql,
    refresh_missing_identity=False,
):
    """Resolve PDB polymer entities, preserving terminal and retry statuses."""
    if batch_size < 1 or retries < 0:
        raise ValueError("batch_size must be positive and retries nonnegative")
    requested = sorted({str(pdb_id).strip().upper() for pdb_id in pdb_ids if str(pdb_id).strip()})
    records = read_cache(cache_path)
    pending = [
        pdb_id
        for pdb_id in requested
        if records.get(pdb_id, {}).get("status") not in {"ok", "not_found"}
        or (
            refresh_missing_identity
            and records.get(pdb_id, {}).get("status") == "ok"
            and any(
                field not in records[pdb_id]
                for field in (
                    "entry_title",
                    "entry_polymer_entity_count",
                    "entry_distinct_uniprot_accessions",
                )
            )
        )
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        attempts, http_status, payload, error = _request_batch(
            batch, url, timeout, retries, backoff, transport
        )
        if payload is None:
            for pdb_id in batch:
                records[pdb_id] = {
                    "pdb_id": pdb_id,
                    "status": "retry_exhausted",
                    "attempts": attempts,
                    "http_status": http_status,
                    "error": error,
                }
        else:
            returned = {}
            for entry in (payload.get("data") or {}).get("entries") or []:
                parsed = parse_entry(entry)
                returned[parsed["pdb_id"]] = parsed
            for pdb_id in batch:
                parsed = returned.get(pdb_id)
                if parsed is None:
                    records[pdb_id] = {
                        "pdb_id": pdb_id,
                        "status": "not_found",
                        "attempts": attempts,
                        "http_status": http_status,
                        "entities": [],
                    }
                else:
                    parsed.update(
                        {"status": "ok", "attempts": attempts, "http_status": http_status}
                    )
                    records[pdb_id] = parsed
        write_cache(cache_path, records)
    return {pdb_id: records[pdb_id] for pdb_id in requested if pdb_id in records}
