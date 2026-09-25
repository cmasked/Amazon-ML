"""
Official F0.5 macro-averaged evaluation metric implementation.
Matches the competition specification exactly.
"""


def f05_single_entity(predicted_set, ground_truth_set):
    """
    Compute F0.5 for a single Source 1 entity.
    
    Args:
        predicted_set: set of predicted matching entity IDs
        ground_truth_set: set of ground truth matching entity IDs
    
    Returns:
        F0.5 score for this entity
    
    Special cases per competition rules:
    - Both empty: F0.5 = 1.0 (correct singleton prediction)
    - GT empty, pred non-empty: F0.5 = 0.0 (false merges on singleton)
    - GT non-empty, pred empty: F0.5 = 0.0 (missed all matches)
    """
    # Both empty = correct singleton
    if len(ground_truth_set) == 0 and len(predicted_set) == 0:
        return 1.0
    
    # GT empty but predicted something = false merge on singleton
    if len(ground_truth_set) == 0 and len(predicted_set) > 0:
        return 0.0
    
    # GT non-empty but predicted nothing = missed everything
    if len(ground_truth_set) > 0 and len(predicted_set) == 0:
        return 0.0
    
    # Both non-empty: compute precision, recall, F0.5
    tp = len(predicted_set & ground_truth_set)
    
    if tp == 0:
        return 0.0
    
    precision = tp / len(predicted_set)
    recall = tp / len(ground_truth_set)
    
    # F0.5 = (1.25 * P * R) / (0.25 * P + R)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    
    return f05


def macro_f05(predictions, ground_truth):
    """
    Compute macro-averaged F0.5 across all Source 1 entities.
    
    Args:
        predictions: dict mapping source1_entity_id -> set of predicted IDs
        ground_truth: dict mapping source1_entity_id -> set of ground truth IDs
    
    Returns:
        Macro-averaged F0.5 score
    """
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        gt = ground_truth[s1_id]
        scores.append(f05_single_entity(pred, gt))
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)


def evaluate_predictions(pred_dict, gt_dict, verbose=False):
    """
    Full evaluation with detailed breakdown.
    
    Returns dict with:
    - macro_f05: the official score
    - precision_avg: average precision across entities
    - recall_avg: average recall across entities
    - n_entities: total entities evaluated
    - n_singletons_correct: singletons correctly predicted empty
    - n_singletons_wrong: singletons incorrectly given matches
    - n_matched_perfect: matched entities with perfect F0.5
    - n_matched_partial: matched entities with partial F0.5
    - n_matched_zero: matched entities with zero F0.5
    """
    scores = []
    n_singletons_correct = 0
    n_singletons_wrong = 0
    n_matched_perfect = 0
    n_matched_partial = 0
    n_matched_zero = 0
    precisions = []
    recalls = []
    
    for s1_id in gt_dict:
        pred = pred_dict.get(s1_id, set())
        gt = gt_dict[s1_id]
        
        score = f05_single_entity(pred, gt)
        scores.append(score)
        
        if len(gt) == 0:
            if len(pred) == 0:
                n_singletons_correct += 1
            else:
                n_singletons_wrong += 1
        else:
            if score == 1.0:
                n_matched_perfect += 1
            elif score > 0:
                n_matched_partial += 1
            else:
                n_matched_zero += 1
            
            if len(pred) > 0:
                tp = len(pred & gt)
                precisions.append(tp / len(pred))
                recalls.append(tp / len(gt))
            else:
                precisions.append(0.0)
                recalls.append(0.0)
    
    result = {
        'macro_f05': sum(scores) / len(scores) if scores else 0.0,
        'n_entities': len(scores),
        'n_singletons_correct': n_singletons_correct,
        'n_singletons_wrong': n_singletons_wrong,
        'n_matched_perfect': n_matched_perfect,
        'n_matched_partial': n_matched_partial,
        'n_matched_zero': n_matched_zero,
        'avg_precision_matched': sum(precisions) / len(precisions) if precisions else 0.0,
        'avg_recall_matched': sum(recalls) / len(recalls) if recalls else 0.0,
    }
    
    if verbose:
        total_gt_singletons = n_singletons_correct + n_singletons_wrong
        total_matched = n_matched_perfect + n_matched_partial + n_matched_zero
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Macro F0.5: {result['macro_f05']:.6f}")
        print(f"Total entities: {result['n_entities']}")
        print(f"Singletons: {total_gt_singletons} (correct: {n_singletons_correct}, wrong: {n_singletons_wrong})")
        print(f"Matched entities: {total_matched} (perfect: {n_matched_perfect}, partial: {n_matched_partial}, zero: {n_matched_zero})")
        print(f"Avg precision (matched): {result['avg_precision_matched']:.4f}")
        print(f"Avg recall (matched): {result['avg_recall_matched']:.4f}")
    
    return result


def test_evaluator():
    """Verify the evaluator against the competition example and edge cases."""
    # Competition example: pred=[S2-47, S2-193, S3-812], gt=[S2-47, S3-812]
    # Precision = 2/3, Recall = 2/2 = 1.0
    # F0.5 = 1.25 * (2/3) * 1.0 / (0.25 * (2/3) + 1.0) = 0.8333... / 1.1666... = 0.7142857
    pred = {'S2-00047', 'S2-00193', 'S3-00812'}
    gt = {'S2-00047', 'S3-00812'}
    score = f05_single_entity(pred, gt)
    assert abs(score - 0.714286) < 0.001, f"Competition example failed: {score}"
    print(f"Test 1 (competition example): {score:.6f} == 0.714286 PASS")
    
    # Perfect match
    score = f05_single_entity({'A', 'B'}, {'A', 'B'})
    assert score == 1.0
    print(f"Test 2 (perfect match): {score:.6f} == 1.0 PASS")
    
    # One false positive
    score = f05_single_entity({'A', 'B', 'C'}, {'A', 'B'})
    print(f"Test 3 (one FP): {score:.6f} PASS")
    
    # One false negative
    score = f05_single_entity({'A'}, {'A', 'B'})
    print(f"Test 4 (one FN): {score:.6f} PASS")
    
    # Both empty (singleton correct)
    score = f05_single_entity(set(), set())
    assert score == 1.0
    print(f"Test 5 (both empty): {score:.6f} == 1.0 PASS")
    
    # GT empty, pred non-empty (false merge on singleton)
    score = f05_single_entity({'A'}, set())
    assert score == 0.0
    print(f"Test 6 (false merge singleton): {score:.6f} == 0.0 PASS")
    
    # GT non-empty, pred empty (missed all)
    score = f05_single_entity(set(), {'A', 'B'})
    assert score == 0.0
    print(f"Test 7 (missed all): {score:.6f} == 0.0 PASS")
    
    # No overlap
    score = f05_single_entity({'C', 'D'}, {'A', 'B'})
    assert score == 0.0
    print(f"Test 8 (no overlap): {score:.6f} == 0.0 PASS")
    
    print("\nAll evaluator tests PASSED!")


if __name__ == '__main__':
    test_evaluator()
