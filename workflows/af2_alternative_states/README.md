# Alternative-conformation covariance masking

## Scientific workflow

```text
alternative-state structure pairs
→ 01 select pairs and canonical sequences
→ 02 calculate ConFind contacts
→ 03 convert contacts to sequence-index matrices
→ 04 align the two conformations
→ 05 identify state-exclusive contacts and build masks
→ 06 run canonical AF2 conditions
     ├─ no_mask
     ├─ mask_excl1 (mask state-1-exclusive contacts)
     └─ optional mask_excl2
→ 07 score predictions against both experimental states with US-align


## Core scripts

1. `scripts/01_prepare_dataset.py`
   - Selects alternative-conformation pairs and extracts canonical sequences.
   - Inputs: source pair table and assembly mmCIF files.
   - Outputs: filtered pair table and sequence dictionary.

2. `scripts/02_run_confind.py`
   - Runs ConFind on experimental structures.
   - Outputs raw contact records.

3. `scripts/03_build_contact_maps.py`
   - Converts ConFind records into binary sequence-index contact matrices.

4. `scripts/04_align_conformations.py`
   - Uses HHalign global alignment to map the two conformations.

5. `scripts/05_build_exclusive_masks.py`
   - Restricts contacts to mutually resolved aligned positions.
   - Produces `excl1`, `excl2`, and alignment-index arrays.

6. `scripts/06_predict_conditions.py`
   - Canonical public prediction runner.
   - Runs the final no-mask and full state-exclusive-mask conditions.
   - Resumes independently by target/pair/condition.

7. `scripts/07_score_predictions.py`
   - Scores each saved prediction against both experimental references using
     the reference-normalized US-align TM-score.
