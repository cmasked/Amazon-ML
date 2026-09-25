"""
Evaluation Framework for Business Entity Resolution Challenge.
Independent, robust implementation for Person 4 (Validation & Compliance).

Enforces exact official competition metrics:
- Per-entity F0.5 macro-average across all S1 entities
- Exact singleton rules:
    - true=empty, pred=empty => 1.0 (correct singleton)
    - true=empty, pred=non-empty => 0.0 (false merge on singleton)
    - true=non-empty, pred=empty => 0.0 (missed all matches)
- Candidate blocking evaluation: recall ceiling, reduction ratio, candidate statistics
- Detailed breakdowns by country (US, India, unseen/France) and match cardinality
- Memory-efficient streaming file evaluations for large datasets (2M+ records)
"""

from __future__ import annotations
import csv
import json
import math
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd


def f05_single_entity(
    predicted_set: Union[Set[str], Iterable[str]],
    ground_truth_set: Union[Set[str], Iterable[str]],
) -> float:
    """
    Compute official F0.5 for a single Source 1 entity.

    Formula:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Precision is weighted 2x over Recall:
        beta = 0.5 => beta^2 = 0.25
        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
               = 1.25 * P * R / (0.25 * P + R)

    Special cases per official rules:
    - Both empty: 1.0 (correctly identified singleton)
    - Ground truth empty, pred non-empty: 0.0 (false merge penalty)
    - Ground truth non-empty, pred empty: 0.0 (missed matches)
    - True positives = 0: 0.0
    """
    pred = predicted_set if isinstance(predicted_set, set) else set(predicted_set)
    gt = ground_truth_set if isinstance(ground_truth_set, set) else set(ground_truth_set)

    len_gt = len(gt)
    len_pred = len(pred)

    # Singleton correctly predicted empty
    if len_gt == 0 and len_pred == 0:
        return 1.0

    # False merge on singleton
    if len_gt == 0 and len_pred > 0:
        return 0.0

    # Missed all matches
    if len_gt > 0 and len_pred == 0:
        return 0.0

    # Calculate TP, Precision, Recall
    tp = len(pred & gt)
    if tp == 0:
        return 0.0

    precision = tp / len_pred
    recall = tp / len_gt

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """
    Compute official macro-averaged F0.5 across all evaluated entities.

    Missing predictions in `predictions` default to empty sets (singleton prediction).
    """
    if not ground_truth:
        return 0.0

    total_score = 0.0
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_score += f05_single_entity(pred_set, gt_set)

    return total_score / len(ground_truth)


def evaluate_predictions(
    pred_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    country_dict: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, dict]]:
    """
    Comprehensive evaluation of predictions against ground truth.

    Args:
        pred_dict: mapping of source1_entity_id -> set of predicted IDs
        gt_dict: mapping of source1_entity_id -> set of ground truth IDs
        country_dict: optional mapping of source1_entity_id -> country string
        verbose: whether to print formatted diagnostic table

    Returns:
        Structured evaluation metrics dictionary.
    """
    total_entities = len(gt_dict)
    if total_entities == 0:
        return {"macro_f05": 0.0, "total_entities": 0}

    scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    # Match counts breakdown
    # 0 = singleton, 1 = 1-to-1 match, 2+ = multi-match
    singleton_total = 0
    singleton_correct = 0
    singleton_false_merges = 0

    single_match_total = 0
    single_match_scores: List[float] = []

    multi_match_total = 0
    multi_match_scores: List[float] = []

    # Country breakdown: country -> list of scores
    country_scores: Dict[str, List[float]] = {}

    # Category counters for matched entities
    n_matched_perfect = 0
    n_matched_partial = 0
    n_matched_zero = 0

    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        score = f05_single_entity(pred_set, gt_set)
        scores.append(score)

        country = country_dict.get(s1_id, "Unknown") if country_dict else "All"
        if country not in country_scores:
            country_scores[country] = []
        country_scores[country].append(score)

        gt_len = len(gt_set)
        pred_len = len(pred_set)

        if gt_len == 0:
            singleton_total += 1
            if pred_len == 0:
                singleton_correct += 1
            else:
                singleton_false_merges += 1
        else:
            if gt_len == 1:
                single_match_total += 1
                single_match_scores.append(score)
            else:
                multi_match_total += 1
                multi_match_scores.append(score)

            if score == 1.0:
                n_matched_perfect += 1
            elif score > 0.0:
                n_matched_partial += 1
            else:
                n_matched_zero += 1

            if pred_len > 0:
                tp = len(pred_set & gt_set)
                precisions.append(tp / pred_len)
                recalls.append(tp / gt_len)
            else:
                precisions.append(0.0)
                recalls.append(0.0)

    overall_macro_f05 = sum(scores) / len(scores)
    matched_total = single_match_total + multi_match_total

    results: Dict[str, Union[float, int, dict]] = {
        "macro_f05": overall_macro_f05,
        "n_entities": total_entities,
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_false_merges": singleton_false_merges,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total else 1.0,
        "matched_total": matched_total,
        "n_matched_perfect": n_matched_perfect,
        "n_matched_partial": n_matched_partial,
        "n_matched_zero": n_matched_zero,
        "single_match_f05": sum(single_match_scores) / len(single_match_scores) if single_match_scores else 0.0,
        "multi_match_f05": sum(multi_match_scores) / len(multi_match_scores) if multi_match_scores else 0.0,
        "avg_precision_matched": sum(precisions) / len(precisions) if precisions else 0.0,
        "avg_recall_matched": sum(recalls) / len(recalls) if recalls else 0.0,
        "by_country": {
            c: {
                "macro_f05": sum(s) / len(s),
                "count": len(s),
            }
            for c, s in country_scores.items()
        },
    }

    if verbose:
        print_evaluation_report(results)

    return results


def evaluate_blocking(
    gt_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
    total_target_pool: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int]]:
    """
    Evaluate candidate generation / blocking stage (Person 2 deliverable).

    Metrics:
    - Candidate Recall Ceiling: What % of true matches are present in candidates?
      recall_ceiling = sum(|GT_i & Cand_i|) / sum(|GT_i|)
    - Fully Retained Entities: What % of matched entities have ALL true matches in candidates?
    - Candidate Set Size: Mean, median, max number of candidates per entity
    - Reduction Ratio: 1 - (Total Candidates / (Total S1 * Total Targets))
    """
    total_true_matches = 0
    found_true_matches = 0

    entities_with_all_found = 0
    entities_with_partial_found = 0
    entities_with_zero_found = 0
    total_matched_entities = 0

    candidate_lengths: List[int] = []
    singleton_with_candidates = 0
    singleton_candidate_free = 0

    for s1_id, gt_set in gt_dict.items():
        cand_set = candidate_dict.get(s1_id, set())
        cand_len = len(cand_set)
        candidate_lengths.append(cand_len)

        gt_len = len(gt_set)
        if gt_len == 0:
            if cand_len > 0:
                singleton_with_candidates += 1
            else:
                singleton_candidate_free += 1
        else:
            total_matched_entities += 1
            total_true_matches += gt_len
            captured = len(gt_set & cand_set)
            found_true_matches += captured

            if captured == gt_len:
                entities_with_all_found += 1
            elif captured > 0:
                entities_with_partial_found += 1
            else:
                entities_with_zero_found += 1

    candidate_lengths.sort()
    n = len(candidate_lengths)
    total_candidates = sum(candidate_lengths)
    avg_cands = total_candidates / n if n > 0 else 0.0
    median_cands = candidate_lengths[n // 2] if n > 0 else 0
    p95_cands = candidate_lengths[int(n * 0.95)] if n > 0 else 0
    max_cands = candidate_lengths[-1] if n > 0 else 0

    recall_ceiling = (found_true_matches / total_true_matches) if total_true_matches > 0 else 1.0

    reduction_ratio = None
    if total_target_pool and n > 0:
        total_possible_pairs = n * total_target_pool
        reduction_ratio = 1.0 - (total_candidates / total_possible_pairs)

    result = {
        "candidate_recall_ceiling": recall_ceiling,
        "total_true_matches": total_true_matches,
        "found_true_matches": found_true_matches,
        "missed_true_matches": total_true_matches - found_true_matches,
        "matched_entities_100pct_retained": entities_with_all_found,
        "matched_entities_partial_retained": entities_with_partial_found,
        "matched_entities_zero_retained": entities_with_zero_found,
        "total_matched_entities": total_matched_entities,
        "entity_full_recall_rate": (entities_with_all_found / total_matched_entities) if total_matched_entities else 1.0,
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "p95_candidates_per_s1": p95_cands,
        "max_candidates_per_s1": max_cands,
        "singletons_candidate_free": singleton_candidate_free,
        "singletons_with_candidates": singleton_with_candidates,
        "reduction_ratio": reduction_ratio if reduction_ratio is not None else 0.0,
    }

    if verbose:
        print("\n=== BLOCKING / CANDIDATE GENERATION EVALUATION ===")
        print(f"Candidate Recall Ceiling: {result['candidate_recall_ceiling'] * 100:.3f}% ({found_true_matches:,}/{total_true_matches:,} true links)")
        print(f"Entities with 100% recall: {result['matched_entities_100pct_retained']:,} / {total_matched_entities:,} ({result['entity_full_recall_rate'] * 100:.2f}%)")
        print(f"Candidate Count Distribution: avg={avg_cands:.1f}, median={median_cands}, p95={p95_cands}, max={max_cands}")
        print(f"Singletons candidate-free: {singleton_candidate_free:,} (with candidates: {singleton_with_candidates:,})")
        if reduction_ratio is not None:
            print(f"Reduction Ratio: {reduction_ratio * 100:.5f}%")

    return result


def print_evaluation_report(res: dict) -> None:
    """Print clean terminal report for Person 4 evaluation."""
    print("\n" + "=" * 60)
    print("      OFFICIAL ENTITY RESOLUTION EVALUATION REPORT      ")
    print("=" * 60)
    print(f"Macro F0.5 Score           : {res['macro_f05']:.6f}")
    print(f"Total S1 Entities Evaluated: {res['n_entities']:,}")
    print("-" * 60)
    print("SINGLETON BREAKDOWN:")
    print(f"  Total Singletons         : {res['singleton_total']:,}")
    print(f"  Correctly Empty (1.0)    : {res['singleton_correct']:,} ({res['singleton_accuracy'] * 100:.2f}%)")
    print(f"  False Merges (0.0)       : {res['singleton_false_merges']:,}")
    print("-" * 60)
    print("MATCHED ENTITIES BREAKDOWN:")
    print(f"  Total Matched Entities   : {res['matched_total']:,}")
    print(f"  Perfect Matches (1.0)    : {res['n_matched_perfect']:,}")
    print(f"  Partial Matches (0 < F)  : {res['n_matched_partial']:,}")
    print(f"  Zero Matches (0.0)       : {res['n_matched_zero']:,}")
    print(f"  1-to-1 Match F0.5        : {res['single_match_f05']:.6f}")
    print(f"  1-to-Many Match F0.5     : {res['multi_match_f05']:.6f}")
    print(f"  Avg Precision (matched)  : {res['avg_precision_matched']:.4f}")
    print(f"  Avg Recall (matched)     : {res['avg_recall_matched']:.4f}")
    if "by_country" in res and res["by_country"]:
        print("-" * 60)
        print("PER-COUNTRY MACRO F0.5:")
        for c, data in res["by_country"].items():
            print(f"  {c:<20}: {data['macro_f05']:.6f}  (n = {data['count']:,})")
    print("=" * 60 + "\n")


def load_id_mapping_tsv(
    file_path: str,
    id_col: str = "source1_entity_id",
    list_col: str = "matched_entity_ids",
) -> Dict[str, Set[str]]:
    """
    Load TSV with comma-separated entity IDs into memory-efficient dict of sets.
    Uses stdlib csv reader for minimal memory footprint and fast streaming.
    """
    mapping: Dict[str, Set[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return mapping

        try:
            id_idx = header.index(id_col)
            list_idx = header.index(list_col)
        except ValueError:
            # Fall back to columns 0 and 1
            id_idx, list_idx = 0, 1

        for row in reader:
            if not row or len(row) <= id_idx:
                continue
            s1_id = row[id_idx].strip()
            if not s1_id:
                continue

            if len(row) > list_idx and row[list_idx].strip():
                targets = {item.strip() for item in row[list_idx].split(",") if item.strip()}
            else:
                targets = set()
            mapping[s1_id] = targets

    return mapping
