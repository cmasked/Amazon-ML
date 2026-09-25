# Person 4 Validation Harness & Integration Guide
**Amazon ML Challenge 2026 — Business Entity Resolution**

This guide provides Person 1 (Pipeline Lead), Person 2 (Candidate Generation / Blocking), and Person 3 (Matching Model) with clean, tested interfaces to evaluate their work against the official competition metric without leakage.

---

## 🚨 The Golden Rule: Split Usage

We have generated 4 reproducible, leakage-free validation partitions in `splits/`:

| Split File | Size | Purpose | Who Should Use It |
|---|---|---|---|
| `splits/fast_dev_50k.json` | **49,997** entities | Sub-minute local iteration, threshold tuning, and feature benchmarking. | **Person 2 & Person 3** (Default for daily experiments) |
| `splits/cv_5folds.json` | **1,875,801** entities (5 folds) | Leakage-free cross-validation for training final ML models. | **Person 1 & Person 3** |
| `splits/loco_splits.json` | US vs. India splits | Leave-One-Country-Out evaluation to simulate generalization to **France**. | **All Teammates** |
| `splits/frozen_holdout.json` | **331,020** entities (15.0%) | Sealed holdout for final pre-submission sanity checks. **NEVER TUNE ON THIS ITERATIVELY.** | **Person 1 (Pre-submission only)** |

> [!WARNING]
> **Strict Leakage Prevention**:
> Never tune thresholds, select features, or optimize blocking parameters repeatedly against `splits/frozen_holdout.json`. If you overfit the holdout, our leaderboard score on the private test set will drop.

---

## ⚡ Quickstart for Person 2 (Candidate Generation / Blocking)

Your goal is to maximize **Candidate Recall Ceiling** while keeping **Average Candidates per S1** manageable (under 50-100).

### Python Interface:
```python
from src.evaluate import evaluate_blocking, load_id_mapping_tsv
from src.split_manager import load_split_ids
from src.config import TRAIN_GT

# 1. Load ground truth for fast_dev entities
target_ids = load_split_ids('splits/fast_dev_50k.json')
all_gt = load_id_mapping_tsv(TRAIN_GT, 'source1_entity_id', 'matched_entity_ids')
eval_gt = {eid: all_gt[eid] for eid in target_ids if eid in all_gt}

# 2. Your candidate generation dictionary: {s1_id: set_of_candidate_ids}
my_candidates = find_candidates_for_entities(target_ids)

# 3. Evaluate blocking performance
metrics = evaluate_blocking(eval_gt, my_candidates, verbose=True)
# metrics contains:
# - 'candidate_recall_ceiling': % of true matches captured (aim for > 95%)
# - 'avg_candidates_per_s1': mean candidate count per S1 entity
# - 'singletons_candidate_free': singletons where no candidates were generated (bonus!)
```

### CLI Command:
If your blocking module exports a TSV file:
```bash
python experiments/run_validation_suite.py --candidate output/my_candidates.tsv --split fast_dev
```

---

## 🎯 Quickstart for Person 3 (Matching Model & Feature Engineering)

Your goal is to optimize the official **Macro F0.5 Score**. Remember that $F_{0.5}$ is precision-heavy: **false merges on singletons or non-matches are penalized $2\times$ harder than missed matches!**

### Python Interface:
```python
from src.evaluate import evaluate_predictions, load_id_mapping_tsv
from src.split_manager import load_split_ids
from src.config import TRAIN_GT

# 1. Load fast_dev ground truth
target_ids = load_split_ids('splits/fast_dev_50k.json')
all_gt = load_id_mapping_tsv(TRAIN_GT, 'source1_entity_id', 'matched_entity_ids')
eval_gt = {eid: all_gt[eid] for eid in target_ids if eid in all_gt}

# 2. Your model predictions dictionary: {s1_id: set_of_matched_ids}
my_predictions = my_model.predict(target_ids)

# 3. Comprehensive evaluation
metrics = evaluate_predictions(my_predictions, eval_gt, verbose=True)
# metrics contains:
# - 'macro_f05': official leaderboard metric
# - 'singleton_accuracy': accuracy on identifying 0-match entities (worth 5.6% of total score!)
# - 'avg_precision_matched' vs 'avg_recall_matched'
# - 'by_country': breakdown of scores for US vs India
```

### CLI Command:
If your model exports a TSV file:
```bash
python experiments/run_validation_suite.py --matching output/my_matching_results.tsv --split fast_dev
```

---

## 🇫🇷 France Robustness Testing (Unseen Country in Test Set)

The test set contains **259,452 entities from France** (~15% of the test set). France does not appear anywhere in the training data!

### Critical Traps to Avoid:
1. **Never filter by country**: Avoid `df[df['country'].isin(['US', 'India'])]`. France records must be processed!
2. **Never hardcode postal codes to 6 digits**: India has 6 digits, but France and US have 5 digits. Use `extract_french_postal_code` or generic numeric tokenizers.
3. **Beware of generic token-set false positives**:
   - Our tests revealed that `Pharmacie de la Mairie` and `Pharmacie de la Gare` score **> 0.90** under naive RapidFuzz `token_set_ratio`!
   - Under $F_{0.5}$, merging them yields a score of **0.0** on that entity.
   - Always verify that legal suffix removal supports French forms (`SARL, SAS, SA, EURL, SCI, SNC`).

### Audit Your Function:
```python
from src.france_stress import audit_similarity_function_on_france

# Pass your similarity function (returns 0.0 - 1.0)
audit_similarity_function_on_france(my_similarity_function, "My Model Scorer")
```

---

## 🚀 Pre-Flight Submission Audit (Person 1 / Team Lead)

Before uploading any file to the portal or bundling the final zip:

```bash
python src/audit_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

If it prints `AUDIT RESULT: ALL CRITICAL SUBMISSION CHECKS PASSED`, the file is 100% compliant and will not get rejected by the competition portal.
