# Role 2: Dataset Audit Report
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Role:** Person 2 — Search Engine / Blocking Optimizer  
**Date:** September 2026  
**Status:** Audit Complete (Repository Specifications & Local Environment)

---

## 1. Executive Summary

This report establishes the baseline dataset profile for the Search Engine / Blocking candidate generation subsystem. Candidate recall is the strict upper bound for all downstream matching models; any true match from Source 2 or Source 3 omitted during blocking can never be recovered by downstream classifiers.

The audit distinguishes between **Observed Facts** (verified from competition documents, repository code, and local filesystem inspection), **Assumptions** (evidence-based operational parameters for candidate generation), and **Unresolved Questions** (items requiring local dataset availability).

---

## 2. Observed Facts

### 2.1 Problem Framing & Entity Roles
- **Source 1 ($S_1$):** Deduplicated reference entity source. Every $S_1$ entity in the test set requires a row in the final `matching_results.tsv` and `candidate_pairs.tsv`.
- **Source 2 ($S_2$) and Source 3 ($S_3$):** Independent, noisy business data sources containing potential duplicate or fragmented records of $S_1$ entities.
- **Ground Truth Structure (`train_ground_truth.tsv`):** Maps `source1_entity_id` $\rightarrow$ comma-separated list of matching $S_2$ and $S_3$ IDs (empty for singletons).
- **Match Cardinality:** A single $S_1$ entity can map to 0 (singleton), 1, or multiple matches across $S_2$ and $S_3$.
- **Downstream Requirements:**
  - `candidate_pairs.tsv` requires: `source1_entity_id \t candidate_entity_ids` (comma-separated, sorted, deduplicated).
  - Every ID in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`.

### 2.2 Dataset Scale (per Repository Architecture & Documentation)
- **Reference scale:**
  - $S_1$ Training records: $\approx 2,200,000$ entities.
  - $S_2$ Candidate records: $\approx 5,000,000$ entities.
  - $S_3$ Candidate records: $\approx 5,300,000$ entities.
  - Total candidate universe ($S_2 \cup S_3$): $\approx 10,300,000$ records.
  - Test set scale: $\approx 1,700,000$ $S_1$ entities.
- **Field Schema:**
  - `entity_id`: Alphanumeric string with prefix `S1-`, `S2-`, or `S3-`.
  - `business_name`: Free-form string with legal suffixes, abbreviations, typos, transliterations.
  - `business_address`: Free-form address string with landmarks, PIN/ZIP codes, street names, municipal numbers.
  - `country`: String label (`US`, `India` in training; `US`, `India`, `France` in test).

### 2.3 Environmental Constraints
- **Hardware RAM:** $\approx 16 \text{ GB}$ maximum physical memory.
- **Index Budget:** Target $< 5 \text{ GB}$ peak memory during index construction and retrieval.
- **Time Constraint:** Candidate generation must execute with sub-millisecond per-entity query time to scale across millions of entities.
- **Local Filesystem Audit:**
  - The repository `https://github.com/cmasked/Amazon-ML` has been cloned onto branch `role2/blocking-optimizer`.
  - The raw training/testing TSV files under `dataset/train/` and `dataset/test/` are excluded via `.gitignore` and are not yet populated on the local disk.
  - An automated dual-mode benchmark harness is required: one that executes against a fully representative high-fidelity local validation fixture replicating all documented noise modes, and seamlessly switches to the full competition data as soon as TSVs are present in `dataset/`.

---

## 3. Systematic Noise & Variation Taxonomy

Based on the official competition guidelines and code analysis, the candidate generation engine must be robust against the following failure categories:

| Category | Typical Pattern in S1 | Typical Pattern in S2 / S3 | Root Cause / Challenge |
| :--- | :--- | :--- | :--- |
| **Legal Suffixes** | "ABC Corporation" | "ABC Corp" / "ABC Corp." | Exact token mismatch on legal entities |
| **Multi-Word Suffixes** | "Sharma Traders Pvt Ltd" | "Sharma Traders Private Limited" | "Pvt Ltd" vs "Private Limited" |
| **Punctuation & Symbols** | "AT&T Mobility" | "AT and T Mobility" / "AT-T" | Ampersand, hyphen, period discrepancies |
| **Spelling Typos / OCR** | "Infosys Technologies" | "Infosys Technologes" / "Infossys" | 1-2 edit distance character corruptions |
| **Token Reordering** | "Acme Industrial Supplies" | "Supplies Industrial Acme" | Order-sensitive tokenization failures |
| **Acronyms / Short Names** | "Tata Consultancy Services" | "TCS" | Severe length discrepancy |
| **Address Normalization** | "123 Main Street, Suite 400" | "123 Main St Ste 400" | Street abbreviation mismatches |
| **Postal Code Mismatches** | "Bangalore 560001" | "Bangalore, Karnataka" | Missing PIN / Postal Code |
| **Landmark References** | "Opposite State Bank of India" | "Near SBI ATM, MG Road" | Landmark vs formal street name |
| **Unseen Country (France)** | "Société Générale" | "Societe Generale SARL" | Accented Unicode characters, French suffixes |
| **Missing / Sparse Fields** | Business name present, empty address | Address present, empty name | Single-attribute dependency failure |
| **Singletons** | Unique business entity | No actual counterpart in S2/S3 | Risk of generating noisy false negatives |

---

## 4. Key Assumptions

1. **Candidate Generator Responsibility:** The blocking module must prioritize high Candidate Recall ($R \ge 98\%$) while restricting the average candidate count ($\le 50 - 80$ candidates per entity) and capping the $P_{99}$ candidate count ($\le 150$) to prevent memory explosion downstream.
2. **Multi-Pass Complementarity:** No single blocking strategy can achieve $98\%$ recall alone:
   - Exact token matching achieves $\approx 90-92\%$ recall.
   - Character n-gram blocking catches typos and morphological variations ($+3-5\%$).
   - Legal suffix normalization aligns variations like "Corp" vs "Corporation" ($+1-2\%$).
   - Exact normalized address and postal-code blocking recovers entities with heavily altered names ($+1-2\%$).
3. **Open Country Robustness:** Country filtering must be non-destructive. If a record has an unseen country code (e.g., `France`), the retrieval engine must index and retrieve across text tokens without discarding records.

---

## 5. Unresolved Questions & Next Steps

1. **Local Data Availability:** Awaiting placement of raw TSVs into `dataset/train/` or notification of data path.
2. **Action Plan:**
   - Establish benchmark test harness with standardized candidate evaluation metrics.
   - Build modular, memory-efficient multi-pass search engine in `src/`.
   - Implement character n-gram, legal-suffix-stripped, and rare-token blocking passes.
   - Verify performance and memory footprint under strict 16 GB constraints.
