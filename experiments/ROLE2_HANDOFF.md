# Role 2 — Search Engine / Candidate Generation Handoff

**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Role:** Person 2 — Search Engine / Blocking / Candidate Generation  
**Date:** September 25, 2026  
**Audited & Handoff Commits:** `102e56a` / `a0f0271` (Branch: `role2/blocking-optimizer`)  
**Status:** **FINALIZED — READY FOR PERSON 1 INTEGRATION**

---

## 1. Objective

Role 2 owns the **candidate-generation / blocking search engine**. The mission of this subsystem is to maximize candidate recall ($\ge 98\%+$) across heterogeneous business databases ($S_1$, $S_2$, $S_3$), while strictly capping candidate volume ($top\_k \le 80$) and maintaining a lean memory footprint compatible with the competition's **16 GB RAM hardware limit**.

Candidate generation acts as the vital retrieval funnel: any true match missed by Role 2 can never be recovered by the downstream matching model.

---

## 2. Final Retrieval Architecture

The candidate generation engine ([`src/search_engine.py`](file:///d:/amazon%20ml/src/search_engine.py)) implements a 7-pass multi-tier inverted index with guaranteed slot reservation:

* **Pass A — Exact Core Name & Compact Name:** Strips legal entity forms (`Inc`, `LLC`, `Corp`, `Pvt Ltd`, `SARL`, `GmbH`), splits letter-digit boundaries (`Titanium4` $\rightarrow$ `Titanium 4`), and hashes both token-normalized and continuous alphanumeric compact strings (`walmart`).
* **Pass B — Exact Physical Address:** Normalizes street abbreviations (`St`, `Rd`, `Ave`, `Blvd`), standardizes punctuation, and hashes physical address strings.
* **Pass C — Character N-Gram Retrieval:** Extracts character 3-grams over normalized names with sub-linear IDF weighting, providing robust resilience against OCR noise, token concatenation, and spelling typos.
* **Pass D — Word Token TF-IDF:** Traditional token-level inverted index weighted by inverse document frequency, penalizing generic terms (`services`, `group`) and elevating distinctive trade names.
* **Pass E — Rare Token Retrieval:** Fast-tracks matches on low-frequency tokens ($DF \le 100$).
* **Pass F — Postal Code & Street Number Index:** Indexes 4-to-7 digit PIN/ZIP codes and street numbers to resolve geographically collocated facilities.
* **Pass G — Country-Conditioned Phonetic Retrieval:** Combines Soundex phonetic encoding with country partitioning to match spoken/phonetic name variants across accents.
* **Guaranteed Slot Reservation:** Dedicates guaranteed candidate slots (`min(top_k // 4, 20)`) to Pass A and Pass B so high-precision identity and address matches are never crowded out by high-frequency fuzzy candidates.

---

## 3. Validation Results

Clean-room verification was conducted from scratch on the 2,000 S1 entity benchmark and the full 10,000 S1 validation set:

| Evaluation Metric | Legacy Single-Token Index | Role 2 Multi-Pass Search Engine | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Candidate Recall (2K Validation)** | 84.98% (1,646 / 1,937) | **100.00%** (1,937 / 1,937) | **+15.02%** (0 missed) |
| **Candidate Recall (10K Validation)** | 84.38% (8,226 / 9,749) | **100.00%** (9,749 / 9,749) | **+15.62%** (0 missed) |
| **Full-Entity Recall (10K Validation)** | 76.54% (4,932 / 6,444) | **100.00%** (6,444 / 6,444) | **+23.46%** |
| **Zero-Recall Entities (10K Validation)** | 988 entities | **0 entities** | **-988 entities** |
| **Candidate Cap ($top\_k$)** | 80 | 80 (capped) | Deterministic |
| **Query Throughput (10K Validation)** | 232 queries / s | 107 queries / s | Production-ready |

---

## 4. Memory Scaling Results

Memory footprint was rigorously profiled at the operating-system process level (Peak RSS via `psutil`), completely avoiding reliance on internal Python heap tracers.

### Actual Empirical Measurements (10K to 1,000,000 Records)
Measured across isolated subprocess executions (see [`experiments/role2_memory_scaling.csv`](file:///d:/amazon%20ml/experiments/role2_memory_scaling.csv)):

| Scale | Indexed Records | Build Time | Base RSS | Peak RSS | Incremental RSS | Bytes / Record |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10K** | 10,000 | 1.50s | 22.1 MB | 38.6 MB | +16.5 MB | 1,724.8 B |
| **25K** | 25,000 | 3.49s | 22.5 MB | 53.0 MB | +30.5 MB | 1,279.1 B |
| **50K** | 50,000 | 7.08s | 22.4 MB | 75.0 MB | +52.5 MB | 1,101.0 B |
| **100K** | 100,000 | 14.42s | 22.5 MB | 111.9 MB | +89.4 MB | 937.1 B |
| **250K** | 250,000 | 33.41s | 22.5 MB | 198.7 MB | +176.2 MB | 739.1 B |
| **500K** | 500,000 | 63.58s | 22.4 MB | 322.6 MB | +300.1 MB | 629.5 B |
| **1M** | 1,000,000 | 132.02s | 22.5 MB | 545.4 MB | +522.9 MB | 548.3 B |

### Full-Scale 10.3M Projection
Linear regression on Incremental Process RSS vs $N$ ($R^2 = 0.993010$, slope = 531.8 B/rec):
* **Projected Incremental Index Memory at 10.3M:** **5.13 GB**
* **Projected Total Process Peak RSS at 10.3M:** **5.15 GB**
* **Conservative Upper-Bound Projection (+15% fragmentation buffer):** **5.92 GB**
* **Projected RAM Headroom under 16.0 GB Budget:** **+10.08 GB FREE RAM**

> **Explicit Measurement Disclosure:**  
> "10.3M was not directly executed locally; the full-scale number is an empirical extrapolation from measured scaling through 1M records."

---

## 5. Anti-Leakage Audit
**Verdict: PASS (100% Leakage-Free)**  
A line-by-line static and dynamic code inspection verified that:
* Neither ground-truth files (`train_ground_truth.tsv`), target labels, nor positive pair identifiers are ingested by `MultiPassSearchEngine`.
* Normalization and indexing rules operate strictly on raw text fields.
* Ground truth is utilized exclusively post-hoc within the evaluation harness ([`src/candidate_evaluator.py`](file:///d:/amazon%20ml/src/candidate_evaluator.py)) to measure recall metrics.

---

## 6. Testing & Quality Assurance
* **Unit Test Suite:** [`tests/test_candidate_generation.py`](file:///d:/amazon%20ml/tests/test_candidate_generation.py) passes **16/16 tests deterministically** in 0.02s.
* **Unseen Country Handling:** Verified 5/5 test cases (France, Germany, Brazil, unknown country codes, Unicode accents) without crashes or dropped entities.
* **Adversarial Stress Testing:** Verified under 250-record distractor flooding; true matches were retained at Rank #0 in all cases.

---

## 7. Integration Instructions for Person 1

The search engine is completely integrated into [`src/baseline_v3.py`](file:///d:/amazon%20ml/src/baseline_v3.py).

### CLI Usage:
* **Default Multi-Pass Run:**
  ```bash
  python src/baseline_v3.py --val-size 2000 --top-k 80
  ```
* **Legacy Blocking Fallback Run:**
  ```bash
  python src/baseline_v3.py --val-size 2000 --top-k 80 --legacy-blocking
  ```

### Python API Integration:
```python
from search_engine import MultiPassSearchEngine

engine = MultiPassSearchEngine(default_top_k=80)

# 1. Ingest candidates
for record in s2_and_s3_records:
    engine.add_record(
        eid=record['entity_id'],
        bname=record.get('business_name', ''),
        baddr=record.get('business_address', ''),
        country=record.get('country', '')
    )

# 2. Finalize index
engine.finalize_index()

# 3. Retrieve candidates
for s1_rec in s1_records:
    candidate_tuples = engine.retrieve(s1_rec, top_k=80)
    # [("S2-000123", 28.5), ("S3-000456", 21.0), ...]
```

---

## 8. Known Limitations

1. **Empirical Extrapolation Caveat:** Full physical indexing of 10.3M records was not executed locally because the uncompressed 10.3M raw dataset has not been placed in `dataset/`. The 5.92 GB figure is an empirical linear projection from 1M records.
2. **Retrieval vs Downstream Scoring:** Candidate recall measures retrieval completeness (presence of true matches in the top 80). Final competition F0.5 score depends entirely on Person 3's ML classifier and decision thresholding.

---

## 9. Recommended Next Team Experiment

Now that candidate recall is 100%, **do not add more retrieval passes**. The recommended next team experiment for Person 1 and Person 3 is:

> **Experiment:** Train and evaluate Person 3's downstream GBDT matching model under two distinct candidate regimes:  
> 1. **Regime 1:** Using candidates retrieved by **Pass A + Pass B only** (~1.7 candidates / entity; pure exact matches).  
> 2. **Regime 2:** Using candidates retrieved by **Passes A through G** (80 candidates / entity; full multi-pass candidate pool).  
>  
> **Objective:** Evaluate whether the 78 additional candidates generated by Passes C through G provide high-utility hard negatives that improve the classifier's precision-recall frontier, or whether candidate volume should be tuned via `top_k`.

---

## 10. Person 1 Handoff Message (Copy into Team Chat)

```
Team Update — Role 2 (Search Engine / Candidate Generation) is finalized and ready for integration!

- Branch: role2/blocking-optimizer
- Latest Commit: 102e56a (Handoff package commit: [see git log])
- Candidate Recall: 100.00% on 2K and 10K validation sets (1,937/1,937 and 9,749/9,749 matches; 0 missed)
- Full-Entity Recall: 100.00% (zero-recall entities: 0)
- Memory Profiling: Real OS Peak RSS measured across 10K -> 1M records (1M Peak RSS: 545.4 MB; linear R^2 = 0.993010)
- Full-Scale Footprint (10.3M): Projected at 5.15 GB (~5.92 GB conservative with +15% buffer), leaving +10.08 GB free RAM under 16 GB budget
- Anti-Leakage: Verified PASS (zero ground-truth dependencies in candidate generation)
- Tests: 16/16 unit tests passing deterministically (0.016s)
- Integration: MultiPassSearchEngine is already wired into src/baseline_v3.py (run with default settings; legacy blocking preserved via --legacy-blocking)
- Caveat: 10.3M memory is an empirical extrapolation from 1M records; full 10.3M raw data was not executed locally.
- Recommended Next Step: Person 3 should compare downstream model training on Pass A+B candidates vs full Pass A..G candidates to evaluate hard-negative utility.
```
