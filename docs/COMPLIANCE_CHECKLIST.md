# Competition Submission Compliance Checklist
**Amazon ML Challenge 2026 — Business Entity Resolution**

Before submitting to the portal or preparing the final zip archive, verify each item in this checklist.

---

## 1. Output Files & Directory Structure

- [ ] Output directory is named `output/`
- [ ] `output/matching_results.tsv` exists and is tab-separated (`\t`)
- [ ] `output/candidate_pairs.tsv` exists and is tab-separated (`\t`)
- [ ] Header for `matching_results.tsv` is exactly: `source1_entity_id\tmatched_entity_ids`
- [ ] Header for `candidate_pairs.tsv` is exactly: `source1_entity_id\tcandidate_entity_ids`

---

## 2. Entity Completeness & Validity

- [ ] **Exact Entity Count**: Exactly **1,732,544 rows** (plus 1 header row) in both TSV files, matching `dataset/test/test_source1.tsv` 1:1.
- [ ] **No Missing S1 Entities**: Every test S1 entity is present, including singletons and all **259,452 French entities**.
- [ ] **No Duplicate S1 Rows**: No `source1_entity_id` is repeated.
- [ ] **Singleton Representation**: Entities with 0 matches have a completely empty second column (i.e. `S1-xxxxx\t\n` or `S1-xxxxx\n`). No `"None"`, `"NaN"`, `"null"`, `"[]"`, or missing rows.
- [ ] **Valid Target IDs**: All predicted IDs belong to Source 2 (`S2-`) or Source 3 (`S3-`).
- [ ] **No S1 Self-Matches**: No `S1-` IDs appear in `matched_entity_ids` or `candidate_entity_ids`.
- [ ] **No Duplicate IDs**: No repeated IDs within any entity's comma-separated list (`S2-001,S2-001` is invalid).
- [ ] **Candidate Subset Invariant**: Every matched ID in `matching_results.tsv` must be present in `candidate_pairs.tsv` ($\text{matches} \subseteq \text{candidates}$).

---

## 3. Automated Validation Command

Run the independent pre-flight audit tool:
```bash
python src/audit_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

And cross-verify with the official helper:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Both tools must return **Exit Code 0 (PASS)**.

---

## 4. Models, Licenses & Academic Integrity Rules

- [ ] **Model Parameter Size**: Any deep learning / transformer model used must be **$\le$ 8 Billion parameters**.
- [ ] **Model Licensing**: All pre-trained models must carry **MIT or Apache 2.0 licenses**.
- [ ] **Strictly Zero External Lookups**:
  - No commercial entity resolution APIs.
  - No company register / government database lookups (MCA, OpenCorporates, SEC EDGAR, SIRENE).
  - No geocoding / map APIs (Google Maps, OpenStreetMap, Nominatim).
  - No web scraping or external data augmentation.
- [ ] **Reproducibility**: All models and predictions can be regenerated end-to-end using only the code in `code/business_entity_resolution/` and the competition training data.

---

## 5. Final Zip Archive Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # Scored on leaderboard
│   └── candidate_pairs.tsv        # Audited candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                   # Complete source code
│       ├── README.md              # End-to-end reproduction guide
│       └── requirements.txt       # Pinned dependencies
└── Documentation_template.md      # Filled methodology write-up
```
