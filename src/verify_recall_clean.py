"""
Standalone Clean Process Verification of Candidate Recall.
Re-runs from scratch without any caching or state persistence.
Directly verifies numerator and denominator against ground truth.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark
from search_engine import MultiPassSearchEngine
from candidate_evaluator import evaluate_candidate_generation

def run_clean_verification():
    print("=" * 60)
    print("TASK 2: STANDALONE CLEAN PROCESS RECALL VERIFICATION")
    print("=" * 60)

    # 1. Generate benchmark from scratch
    s1_recs, s2s3_recs, gt = generate_synthetic_benchmark(n_s1=2000, singleton_rate=0.35, random_seed=42)
    
    # 2. Directly count true matches from ground truth
    total_entities = len(gt)
    singletons = [sid for sid, matches in gt.items() if len(matches) == 0]
    matched_entities = [sid for sid, matches in gt.items() if len(matches) > 0]
    total_true_matches = sum(len(matches) for matches in gt.values())
    
    print(f"Ground Truth Audit:")
    print(f"  Total S1 entities: {total_entities:,}")
    print(f"  Singletons (0 true matches): {len(singletons):,} ({100*len(singletons)/total_entities:.2f}%)")
    print(f"  Matched S1 entities: {len(matched_entities):,} ({100*len(matched_entities)/total_entities:.2f}%)")
    print(f"  Total true match pairs (DENOMINATOR): {total_true_matches:,}")

    # 3. Build MultiPassSearchEngine fresh
    print("\nBuilding MultiPassSearchEngine index...")
    engine = MultiPassSearchEngine(default_top_k=80)
    for r in s2s3_recs:
        engine.add_record(r['entity_id'], r['business_name'], r['business_address'], r['country'])
    engine.finalize_index()
    print(f"  Total candidates indexed: {len(engine.id_table):,}")

    # 4. Query each S1 entity and track matches directly
    print("\nQuerying all 2,000 S1 entities...")
    retrieved_true_matches = 0
    full_recall_entities = 0
    partial_recall_entities = 0
    zero_recall_entities = 0
    candidates_by_s1 = {}

    for s1_id in matched_entities:
        s1 = s1_recs[s1_id]
        cands = engine.retrieve(s1, top_k=80)
        cand_ids = {cid for cid, _ in cands}
        candidates_by_s1[s1_id] = cand_ids
        
        gt_set = gt[s1_id]
        retrieved = len(gt_set & cand_ids)
        retrieved_true_matches += retrieved
        
        if retrieved == len(gt_set):
            full_recall_entities += 1
        elif retrieved > 0:
            partial_recall_entities += 1
        else:
            zero_recall_entities += 1

    for s1_id in singletons:
        s1 = s1_recs[s1_id]
        cands = engine.retrieve(s1, top_k=80)
        candidates_by_s1[s1_id] = {cid for cid, _ in cands}

    # 5. Direct verification
    candidate_recall = retrieved_true_matches / total_true_matches
    full_entity_recall = full_recall_entities / len(matched_entities)

    print("\nDirect Verification Results:")
    print(f"  Total True Matches (Denominator): {total_true_matches}")
    print(f"  Total Retrieved Matches (Numerator): {retrieved_true_matches}")
    print(f"  Aggregate Candidate Recall: {candidate_recall:.6%} ({retrieved_true_matches}/{total_true_matches})")
    print(f"  Full Entity Recall (100% matches): {full_entity_recall:.6%} ({full_recall_entities}/{len(matched_entities)})")
    print(f"  Partial Recall Entities: {partial_recall_entities}")
    print(f"  Zero Recall Entities: {zero_recall_entities}")

    assert retrieved_true_matches == total_true_matches == 1978, f"Mismatch: {retrieved_true_matches} != 1978"
    assert candidate_recall == 1.0, f"Recall != 1.0: {candidate_recall}"
    assert full_entity_recall == 1.0, f"Full entity recall != 1.0: {full_entity_recall}"
    print("\nRECALL VERIFICATION: PASSED (100.00% confirmed)")

if __name__ == '__main__':
    run_clean_verification()
