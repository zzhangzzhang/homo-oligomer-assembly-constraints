# Literature evidence and structural recomputation

## Overall workflow

```text
Literature branch
PDB/DOI/UniProt links
→ Mini target-card retrieval
→ DOI-complete Terra dossiers
→ exact/normalized quote verification
→ quote-only claim tiers
→ DOI×group target classifications
                         │
                         ├──────────────────────────────────────┐
Structural branch        │                                      │
PDB/UniProt mapping      │                                      │
→ eligible UniProt list │                                      │
→ AlphaFoldDB MSA       │                                      │
→ qid35 Pairformer      │                                      │
→ final manifests       │                                      │
   ├─ deposited-homo branch                                   │
   │  → assembly-1 ConFind                                    │
   │  → Pairformer contact recovery                           │
   │  → assembly-1 PISA                                      │
   │  → literature join → Panel E                            │
   │                                                         │
   └─ deposited-monomer branch                               │
      → three-class literature cohort                         │
      → ASU/lattice dimer enumeration                         │
      → candidate ConFind + Pairformer scoring                 │
      → select maximum-recovery candidate                     │
      → PISA on that exact selected dimer                      │
      → literature join → Panel D                              │
                                                                │
Literature funnel/composition → Panels B/C ◀────────────────────┘
Panels B–E tables → quantitative plot
Quantitative source + structures → manual Figure 6 composition
```



## 1. Literature branch



### 1.1 Canonical DOI×target manifest

```text
PDB→DOI mappings + RCSB entity metadata + article-download status
→ DOI normalization and merge audit
→ exact PDB-entity→UniProt grouping
→ one canonical target card per DOI×group
```

Core implementation:

- `doi.py`: canonical DOI syntax and merge handling.
- `rcsb.py`: parses cached RCSB entry/entity metadata.
- `aliases.py`: bounded gene/protein aliases used only for retrieval.
- `identity.py`: target/construct identity graph and deterministic hashes.
- `manifest.py`: groups PDB entities into DOI×UniProt target cards, applies
readable-paper gates, and writes exclusions/audits.



### 1.2 Mini candidate retrieval

```text
canonical target card + article
→ provenance-preserving text blocks
→ targeted high-recall prompt
→ strict candidate state/event response
```

Core implementation:

- `text.py`: extracts article blocks, assigns stable source IDs/pages/offsets,
and selects at most the configured bounded evidence context.
- `prompt.py`: Mini system/user instructions and target-specific terms.
- `schema.py`: strict candidate state/event JSON schema.
- `extractor.py`: builds one deterministic request and provenance record.
- `lifecycle.py`: immutable request shards, source verification, optional
batch submission/status/collection, and parsed output layers.
- `analysis.py`: candidate/result summaries.


Mini is a retrieval stage: its candidates are hints, not accepted biological
claims or trusted target attribution.

### 1.3 DOI-complete Terra dossiers

```text
Mini candidates + all canonical sibling targets from the same DOI
→ predictor-blind context index
→ DOI-level dossier with full/selected article blocks
```

`contextual.py` performs this deterministic preparation. It:

- joins Mini candidates back to the immutable target manifest;
- includes every canonical target from the same paper;
- treats candidate origin only as a retrieval pointer;
- uses full non-bibliography text below the token cap;
- selects required candidate/identity/context blocks for oversized papers;
- preserves block IDs, pages, sentence/character offsets, layout diagnostics,
article hashes, and token audit;

### 1.4 Terra paper-level extraction

```text
DOI dossier
→ paper-level prompt + strict schema
→ deterministic API request
→ raw structured response
```

Core implementation:

- `contextual_prompt.py`: instructions for target attribution, coreference,
constructs, state/event semantics, method/result linkage, citation scope,
exclusions, and contiguous verbatim evidence.
- `contextual_schema.py`: strict response objects for paper identities,
constructs, state claims, transition claims, exclusions, and unassigned
observations.
- `contextual_extractor.py`: stable custom IDs, exact model/settings,
JSON-schema response format, and prompt/schema/request hashes.


### 1.5 Response parsing and deterministic validation


Core implementation:

- `contextual_pilot.py`: despite its historical name, performs both bounded
synchronous pilots and the production offline response rebuild.
- `contextual_validation.py`: strict schema checks, candidate accounting,
component-level diagnostics, target/construct/evidence validation, and
tolerant technical-error handling.
- `contextual_quote_audit.py`: resolves evidence references to source blocks
and records literal/normalized/invalid matches.
- `validation.py`: deterministic PDF/OCR normalization used for exact quote
comparison.

Outputs include parsed results, verified results, tiered results, validation
errors, quote audit, and a complete rebuild report. The model does not make
the final acceptance decision.

### 1.6 Final quote-only policy and target classes

```text
parsed Terra semantics + DOI dossier blocks
→ require every accepted result quote to match exactly or after allowlisted
  PDF/OCR normalization
→ apply policy exclusions
→ accepted / excluded / ambiguous tiers
→ Monomer-only / Multiple-states / Oligomer-only / Unknown target classes
```

Core implementation:

- `contextual_quote_only.py`: final evidence gate, policy tiers, and target
classifications.
- `contextual_human_export.py`: compact accepted state/event evidence rows,
display-level quote collapsing, and human-readable classifications.



Final recorded totals:

- 20,812 Terra requests;
- 20,746 schema-valid papers;
- 72,811 claim/disposition objects;
- 31,120 accepted claims;
- 12,263 classified DOI×group targets.

Exact quotation proves traceability, not independent correctness of Terra's
target attribution, construct interpretation, biological context, assembly
class, or method/result semantics.

### 1.7 Shared production helpers

Small live helpers are separated from historical validation/reporting code:

- `api_credentials.py`: optional online-stage API-key loading.
- `hashing.py`: streaming SHA-256 file identity.
- `quote_normalization.py`: deterministic PDF/OCR normalization used by Mini,
  Terra validation, quote policy, and audit tooling.
- `state_classification.py`: accepted-claim signals to target-level
  Monomer-only / Multiple-states / Oligomer-only / Unknown classes.


## 2. Pairformer preparation

1. `code/structural/screen_uniprots.py`
  - Screens SIFTS-linked UniProt IDs for available AlphaFoldDB MSAs and
   length ≤500.

1. `code/structural/run_pairformer.py`
  - Applies HHfilter ID90/QID35/`-diff 512`.
  - Runs MSA-Pairformer without query bias.
  - Writes qid35 score matrices.


1. `code/structural/screen_full_manifest.py`
  - Selects the final PDB×UniProt records with SIFTS mappings, qid35
   matrices, length ≤500, and required coordinate files.



## 3. Deposited-homo branch — Panel E

```text
manifest_homo.csv
→ confind_bioassembly_screen.py
→ score_confind_pairformer.py
→ run_pisa_on_bioassembly.py
→ build_quote_only_homo_join.py
→ Panel E rows
```

- `confind_bioassembly_screen.py`
  - Renumbers target chains to UniProt coordinates.
  - Runs ConFind on deposited assembly 1.
  - Separates intrachain and biological-interface contacts.
- `score_confind_pairformer.py`
  - Symmetrizes qid35 matrices and calculates top-X recovery.
- `run_pisa_on_bioassembly.py`
  - Runs protein-only PISA `--as-is` on target assembly 1.
- `build_quote_only_homo_join.py`
  - Joins structural scores, PISA, and final DOI×UniProt classes.



## 4. Deposited-monomer candidate branch — Panel D

```text
final literature classes
→ audit_monomer_three_class_confind_gate.py
→ dimer_confind_qid35_pipeline.py enumerate
→ dimer_confind_qid35_pipeline.py run/finalize
→ dimer_pisa_pipeline.py run/finalize
→ build_quote_only_monomer_pdb_table.py
→ Panel D rows
```

- `reverse_assembly_helpers.py`
  - Shared crystallographic transforms, coordinate/SIFTS handling, and PISA
  XML parsing.
- `dimer_confind_qid35_pipeline.py`
  - Enumerates ASU and lattice dimers without Pairformer selection.
  - Builds full-atom P1 dimers, runs ConFind, and scores qid35 recovery.
  - Selects the maximum-recovery candidate after quality gates.
- `dimer_pisa_pipeline.py`
  - Runs PISA on the same saved selected candidate dimer.
- `build_quote_only_monomer_pdb_table.py`
  - Joins the final candidate, PISA result, and literature class.


## 5. Final literature/panel tables

- `audit_mini_mono_homo_funnel.py`: Mini retrieval denominators.
- `build_quote_only_branch_tables.py`: final branch compositions.
- `build_quote_only_structural_funnel.py`: structural-stage counts.
- `build_quote_only_monomer_pdb_table.py`: Panel D.
- `build_quote_only_homo_join.py`: Panel E.


## External data and tools

Required for full recomputation:

- source articles obtained lawfully;
- paid Mini/Terra model access;
- AlphaFoldDB MSAs;
- MSA-Pairformer source and weights;
- PDB/mmCIF and SIFTS files;
- ConFind executable and DB-2010 rotamer library;
- Docker/CCP4 PISA.
