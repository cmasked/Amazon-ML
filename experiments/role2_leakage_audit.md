# Role 2: Anti-Leakage and Data Integrity Audit Report
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Auditor:** Person 2 — Search Engine / Blocking Optimizer  
**Date:** September 2026  
**Result:** **PASS (Zero Data Leakage Confirmed)**

---

## 1. Executive Summary

A comprehensive, line-by-line static and runtime code audit was conducted across all files involved in candidate generation:
- `src/search_engine.py`
- `src/normalization.py`
- `src/candidate_evaluator.py`
- `src/benchmark_harness.py`
- `src/baseline_v3.py`
- `tests/test_candidate_generation.py`

**Audit Verdict: PASS**

The candidate generation engine is completely independent of:
- `train_ground_truth.tsv`
- Ground-truth matching IDs or target sets
- Validation labels / positive pair mappings
- Downstream evaluator outputs
- Any lookup tables derived from ground truth or target labels

Ground truth is strictly utilized post-hoc within the evaluation harness (`candidate_evaluator.py`) to compute recall statistics against generated candidates.

---

## 2. Production Code Path Trace

The production blocking pipeline follows a strictly unidirectional dataflow:

```
Candidate Source Files (S2, S3)
   │
   ▼
MultiPassSearchEngine.add_record(eid, bname, baddr, country)
   ├── normalization.build_record_representations() [Pure string transforms]
   ├── exact_name_index [Inverted index mapping name_core -> [idx]]
   ├── compact_name_index [Inverted index mapping name_compact -> [idx]]
   ├── exact_addr_index [Inverted index mapping addr_norm -> [idx]]
   ├── ngram_index [Inverted index mapping char_3gram -> [idx]]
   ├── token_index [Inverted index mapping word_token -> [idx]]
   ├── postal_index [Inverted index mapping postal_num -> [idx]]
   └── phonetic_index [Inverted index mapping country_soundex -> [idx]]
   │
   ▼
MultiPassSearchEngine.finalize_index() [Computes IDF from posting lengths; arrays compacted]
   │
   ▼
MultiPassSearchEngine.retrieve(s1_rec, top_k)
   │  [Input: Only S1 entity_id, business_name, business_address, country]
   │  [Processing: Scores accumulated from active inverted indexes]
   │  [Slot Reservation: Exact name & address guaranteed slots + top similarity]
   ▼
Generated Candidate List: [(candidate_id, score), ...]
   │
   ▼
Person 3 Downstream Classifier / Candidate Output (candidate_pairs.tsv)
```

---

## 3. Explicit Leakage Verification Checklist

| Audit Checkpoint | Code Location | Status | Evidence / Verification |
| :--- | :--- | :--- | :--- |
| **Ground truth passed to index?** | `search_engine.py:add_record()` | **NONE** | Function signature: `def add_record(self, eid, bname, baddr, country)`. Does not accept labels or GT. |
| **Ground truth passed to query?** | `search_engine.py:retrieve()` | **NONE** | Function signature: `def retrieve(self, s1_rec, top_k, active_passes)`. `s1_rec` only contains string fields from $S_1$. |
| **API compatibility wrapper clean?** | `search_engine.py:find_candidates_from_index()` | **NONE** | Takes `(name_tokens, addr_tokens, token_index, top_k, s1_rec)`. Zero GT parameters. |
| **Normalization utilities clean?** | `normalization.py` | **NONE** | Functions (`normalize_name_standard`, `strip_legal_suffixes`, `normalize_address_standard`) are pure regex and string functions. |
| **Evaluation harness isolated?** | `candidate_evaluator.py:evaluate_candidate_generation()` | **ISOLATED** | Accepts `candidates_by_s1` (generated candidates) and `ground_truth`. Only computes set intersections (`retrieved = len(cands & gt)`). No back-propagation into search engine. |
| **Synthetic benchmark clean?** | `benchmark_harness.py:run_experiment_suite()` | **ISOLATED** | Only passes `rec['entity_id']`, `rec['business_name']`, `rec['business_address']`, `rec['country']` to `engine.add_record()`. Synthetic metadata (`noise_type`, `matched_s1`) is not passed. |
| **Baseline pipeline clean?** | `baseline_v3.py:run_validation()` | **ISOLATED** | `gt_path` is parsed after index is built and only used at line 395 to score recall. In `mode='test'`, `gt_path` is explicitly `None`. |

---

## 4. Conclusion

The search engine is 100% blind to ground truth and target identities during both index construction and candidate querying. The measured candidate recall reflects genuine indexing, tokenization, sub-word n-gram similarity, and multi-pass retrieval performance.
