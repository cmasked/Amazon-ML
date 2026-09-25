"""
Unit tests for Person 4 Evaluation Framework.
Verifies the official competition metric, edge cases, singletons, and blocking metrics.
"""

import pytest
import math
from src.evaluate import f05_single_entity, macro_f05, evaluate_predictions, evaluate_blocking


def test_official_competition_example():
    """
    Test the exact example from the Amazon ML Challenge problem statement:
    - Pred: [S2-00047, S2-00193, S3-00812]
    - Ground Truth: [S2-00047, S3-00812]
    - Precision: 2/3 = 0.66667, Recall = 1.0
    - Expected F0.5: 5/7 ~= 0.7142857
    """
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}

    score = f05_single_entity(pred, gt)
    expected = 5.0 / 7.0
    assert abs(score - expected) < 1e-6, f"Expected {expected:.6f}, got {score:.6f}"


def test_singleton_correct():
    """
    Per official rules:
    A Source 1 entity with no true matches scores 1.0 when predicted empty.
    """
    assert f05_single_entity(set(), set()) == 1.0
    assert f05_single_entity([], []) == 1.0


def test_singleton_false_merge_penalty():
    """
    Per official rules:
    A Source 1 entity with no true matches scores 0.0 when you predict ANY match for it.
    """
    assert f05_single_entity({"S2-00001"}, set()) == 0.0
    assert f05_single_entity({"S2-00001", "S3-00002"}, set()) == 0.0


def test_missed_all_matches():
    """
    When ground truth has matches but prediction is empty, score is 0.0.
    """
    assert f05_single_entity(set(), {"S2-00001"}) == 0.0
    assert f05_single_entity(set(), {"S2-00001", "S3-00002"}) == 0.0


def test_perfect_matches():
    """
    When prediction exactly equals ground truth, score is 1.0.
    """
    assert f05_single_entity({"S2-00001"}, {"S2-00001"}) == 1.0
    assert f05_single_entity({"S2-00001", "S3-00002"}, {"S2-00001", "S3-00002"}) == 1.0


def test_disjoint_predictions():
    """
    When prediction has no overlap with ground truth, score is 0.0.
    """
    assert f05_single_entity({"S2-99999"}, {"S2-00001"}) == 0.0
    assert f05_single_entity({"S3-11111", "S2-22222"}, {"S2-00001", "S3-00002"}) == 0.0


def test_precision_heavy_asymmetry():
    """
    Verify that F0.5 penalizes false positives (false merges) more than false negatives (misses).
    Suppose GT has 2 items: {A, B}
    - 1 False Positive ({A, B, C}): F0.5 = 5/7 = 0.714286
    - 1 False Negative ({A}):       F0.5 = 5/6 = 0.833333
    F0.5(1 FN) > F0.5(1 FP) proves precision is weighted 2x over recall.
    """
    gt = {"A", "B"}
    score_with_fp = f05_single_entity({"A", "B", "C"}, gt)
    score_with_fn = f05_single_entity({"A"}, gt)

    assert score_with_fn > score_with_fp
    assert abs(score_with_fp - (5.0 / 7.0)) < 1e-6
    assert abs(score_with_fn - (5.0 / 6.0)) < 1e-6


def test_macro_f05():
    """
    Verify macro averaging across diverse entities:
    - Entity 1: perfect match (1.0)
    - Entity 2: correct singleton (1.0)
    - Entity 3: false merge on singleton (0.0)
    - Entity 4: official example (5/7 = 0.714286)
    Expected macro mean: (1.0 + 1.0 + 0.0 + 5/7) / 4 = 2.714286 / 4 = 0.678571
    """
    gt = {
        "S1-1": {"S2-A"},
        "S1-2": set(),
        "S1-3": set(),
        "S1-4": {"S2-47", "S3-812"},
    }
    pred = {
        "S1-1": {"S2-A"},
        "S1-2": set(),
        "S1-3": {"S2-X"},  # False merge!
        "S1-4": {"S2-47", "S2-193", "S3-812"},
    }

    macro_score = macro_f05(pred, gt)
    expected = (1.0 + 1.0 + 0.0 + (5.0 / 7.0)) / 4.0
    assert abs(macro_score - expected) < 1e-6


def test_evaluate_blocking_metrics():
    """
    Test candidate evaluation metrics for Person 2:
    - Candidate recall ceiling
    - Reduction ratio
    - Statistics on candidate sizes
    """
    gt = {
        "S1-1": {"S2-A", "S3-B"},
        "S1-2": {"S2-C"},
        "S1-3": set(),  # singleton
    }
    # Candidates contain 2 of 3 true matches: S2-A, S2-C (missed S3-B)
    candidates = {
        "S1-1": {"S2-A", "S2-Z"},
        "S1-2": {"S2-C", "S3-W"},
        "S1-3": set(),
    }

    res = evaluate_blocking(gt, candidates, total_target_pool=100)
    # Total true matches = 3, captured = 2 => 2/3 recall ceiling
    assert abs(res["candidate_recall_ceiling"] - (2.0 / 3.0)) < 1e-6
    assert res["singletons_candidate_free"] == 1
    assert res["singletons_with_candidates"] == 0
    assert res["total_matched_entities"] == 2
    assert res["matched_entities_100pct_retained"] == 1  # S1-2 has all true matches
    assert res["matched_entities_partial_retained"] == 1  # S1-1 has 1 of 2
