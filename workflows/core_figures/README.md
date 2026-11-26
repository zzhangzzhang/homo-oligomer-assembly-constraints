# Core figures

## Workflow map

```text
Reference structures/contacts
├─ language-model contact generation
│  ├─ official ESM2 contact head
│  ├─ overlay-head fitting/prediction
│  └─ ESM2 / ESM3 / ESM-C categorical Jacobians
│       → EPPIC contact-recovery evaluation
│
├─ same-MSA homolog sampling
│  → ESM2 / ESM-C categorical Jacobians per homolog
│  → RMS-filtered Spearman scores
│
├─ saved ESM2/Pairformer score matrices
│  → resolved-pair filtering and deterministic top-X selection
│
├─ Pairformer QID evaluation table
│  → complete eight-QID cohort and retrospective maxima
│
└─ AF2-Multimer perturbation branches
   ├─ QID × query-bias conditions
   └─ covariance-mask conditions
        → DockQ / US-align scoring
        → balanced processed tables
```



## 1. EPPIC and categorical-Jacobian contacts



### Inference/model-generation scripts

- `scripts/model_generation/catjac/_catjac_contact_common.py`
  - Shared amino-acid ordering, centering, symmetrization, Frobenius norm,
  diagonal removal, and APC conversion.
- `generate_esm2_catjac_contacts.py`
  - ESM2 categorical-Jacobian generation.
- `generate_esm3_catjac_contacts.py`
  - ESM3 categorical-Jacobian generation.
- `generate_esmc_catjac_contacts.py`
  - ESM-C 300M/600M/6B categorical-Jacobian generation.
- `fit_predict_esm2_650m_overlay_head.py`
  - Fits the 17-protein ESM2 overlay head and predicts evaluation maps.
- `fit_predict_esmc_overlay_heads.py`
  - Fits/applies ESM-C overlay heads.

The released ESM2 contact head was trained on 20 structures. The overlay
refits reported here intentionally use the 17 monomer/homo-oligomer
structures in that set; the three hetero-oligomer structures are excluded.

### Evaluation scripts

- `evaluate_eppic_contact_recovery.py`
  - Evaluates model scaling and Pairformer QID predictions against the
  biological/crystal reference sets.
- `evaluate_biological_interface_scatters.py`
  - Produces the 40-target comparison table



## 2. Same-MSA categorical-Jacobian analysis

```text
11 source A3Ms
→ deterministic identity-bin sampling 
→ ESM2 and ESM-C Jacobian generation
→ query-selected top-L / RMS-filtered Spearman scores
→ per-family and supplementary plots
```

Core scripts:

- `sample_same_msa_sequences.py`
- `generate_same_msa_spearman_scores.py`
- `generate_same_msa_esmc_spearman_scores.py`
- `plot_same_msa_jacobian_esm2_esmc_per_protein.py`



## 3. Contact-map source panels

`workflows/contact_maps/scripts/plot_contact_maps.py` loads the bundled reference
contacts, resolved masks, and model score matrices; symmetrizes predictions;
applies the sequence-separation mask; performs deterministic top-X selection;
and writes processed NPZ/metrics plus contact-map figures.

## 4. Pairformer QID scan

`workflows/pairformer_qid_scan/scripts/prepare_and_plot_qid_scan.py`:

- requires QIDs 15–50 for each assembly;
- creates the 537-assembly panel table;
- computes retrospective maxima;



## 5. AF2 query bias and covariance masking

```text
109-target cohort + MSAs + native contacts
├─ run_af2_query_bias_qid_scan.py
└─ masking wrappers → _run_masking_strategy.py
      → _af2_multimer_common.py
      → prediction PDB/log trees
      → scripts/scoring/
      → prepare_af2_query_bias_masking_tables.py
      → plot_af2_query_bias_masking_violins.py
```

Inference scripts:

- `run_af2_query_bias_qid_scan.py`
- `_run_masking_strategy.py`
- four strategy wrappers
- `_af2_multimer_common.py`

Scoring/preparation:

- `scripts/scoring/run_dockq_tmalign.py`
- `scripts/scoring/rebuild_combined_df.py`
- `scripts/scoring/lib/` helpers
- `scripts/analysis/prepare_af2_query_bias_masking_tables.py`

