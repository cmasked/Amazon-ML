# Role 2: Baseline Candidate Generation Benchmark Report
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Component:** Search Engine / Blocking / Candidate Generation  
**Date:** September 2026  
**Status:** Baseline Measured & Reproduced

---

## 1. Reproduction Setup & Exact Commands

The existing blocking and search implementation in the repository relies on `src/baseline_v3.py` with:
1. `build_inverted_index([s2_path, s3_path])`
2. `find_candidates_from_index(name_tokens, addr_tokens, token_index, top_k=80)`

To reproduce the exact baseline candidate generation performance on the standardized validation environment:
```bash
python src/benchmark_harness.py
```
Or when running the end-to-end baseline pipeline in legacy mode:
```bash
python src/baseline_v3.py --val-size 2000 --top-k 80 --legacy-blocking
```

---

## 2. Measured Baseline Metrics

*All metrics are empirical measurements on the standardized validation setup (2,000 $S_1$ entities, 688 singletons, 1,312 matched entities with 1,978 ground truth match pairs across $S_2$ and $S_3$).*

| Metric | Measured Baseline Value |
| :--- | :--- |
| **Aggregate Candidate Recall** | **88.5743%** (1,752 / 1,978 true matches retrieved) |
| **Macro Candidate Recall** | **87.9827%** |
| **Full Entity Recall (100% matches retrieved)** | **83.6128%** (1,097 entities) |
| **Partial Recall (0% < Recall < 100%)** | **8.1555%** (107 entities) |
| **Zero Recall (0% retrieved / complete miss)** | **8.2317%** (108 entities) |
| **Multi-Match Recall** | **89.7243%** |
| **Mean Candidates / Entity** | **80.00** |
| **Median Candidates / Entity** | **80.0** |
| **P90 Candidates / Entity** | **80.0** |
| **P95 Candidates / Entity** | **80.0** |
| **P99 Candidates / Entity** | **80.0** |
| **Maximum Candidates / Entity** | **80** |
| **Total Candidate Pairs Generated** | **160,000** |
| **Index Construction Time** | **0.56s** |
| **Query Retrieval Time** | **5.19s** |
| **Total Pipeline Time** | **5.75s** |
| **Peak Memory Consumption** | **17.28 MB** |

---

## 3. Baseline Failure Mode Analysis

The legacy baseline misses **11.43%** of true matches due to five fundamental structural limitations in its inverted index architecture:

1. **Exact Word Overlap Dependency:**
   - The index relies strictly on whitespace-split exact tokens. If a name has an unlisted legal suffix or typographical error (e.g. "McDonnalds" vs "McDonald's", "Bharath" vs "Bharat"), the posting lookup returns zero candidates for that token.
2. **Aggressive Address Token Length Filtering (`len >= 5`):**
   - The legacy `build_inverted_index` drops all address tokens with length $< 5$.
   - This discards vital discriminating information: house numbers ("42", "101"), road/sector codes ("B-4", "Sec 5"), and standard abbreviations ("St", "Rd", "Ave", "Dr").
3. **No Character N-Gram or Sub-word Indexing:**
   - Single-character transpositions, OCR errors, or hyphenation variations completely break token matching.
4. **Token Pruning Blind Spots:**
   - Overly common tokens ($> 50,000$ postings) are deleted entirely. Businesses with generic core tokens (e.g. "Global Technologies", "Universal Logistics") receive zero candidates when all constituent tokens are pruned.
5. **No High-Precision Exact Path:**
   - Exact identical name and address matches are treated with the same scattering scoring function as broad token overlaps, allowing generic candidates with many low-IDF tokens to displace true exact matches.
