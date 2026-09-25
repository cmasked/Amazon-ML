"""
Granular Multi-Match Entity Analysis for Role 2.
Evaluates recall and candidate volume broken down by ground truth match count:
  - 1 true match (Single link)
  - 2 true matches (Dual link)
  - 3+ true matches (Multi link)
Reports percentage with 100% true matches retrieved, average recall, and candidate count.
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark
from search_engine import MultiPassSearchEngine

def analyze_multimatch_breakdown():
    print("=" * 65)
    print("TASK 10: MULTI-MATCH ENTITY GRANULAR ANALYSIS")
    print("=" * 65)

    s1_recs, s2s3_recs, gt = generate_synthetic_benchmark(n_s1=2000, singleton_rate=0.35, random_seed=42)

    engine = MultiPassSearchEngine(default_top_k=80)
    for r in s2s3_recs:
        engine.add_record(r['entity_id'], r['business_name'], r['business_address'], r['country'])
    engine.finalize_index()

    bins = {
        '1 Match': [],
        '2 Matches': [],
        '3+ Matches': []
    }

    for s1_id, gt_set in gt.items():
        n_gt = len(gt_set)
        if n_gt == 0:
            continue
        elif n_gt == 1:
            bin_name = '1 Match'
        elif n_gt == 2:
            bin_name = '2 Matches'
        else:
            bin_name = '3+ Matches'

        s1 = s1_recs[s1_id]
        cands = engine.retrieve(s1, top_k=80)
        cand_ids = {cid for cid, _ in cands}

        retrieved = len(gt_set & cand_ids)
        recall = retrieved / n_gt
        is_perfect = (retrieved == n_gt)

        bins[bin_name].append({
            's1_id': s1_id,
            'true_matches': n_gt,
            'retrieved_matches': retrieved,
            'recall': recall,
            'perfect': is_perfect,
            'candidate_count': len(cands)
        })

    print(f"{'Category':<15} | {'Entities':<10} | {'All Retrieved (%)':<18} | {'Avg Recall':<12} | {'Avg Cands':<10}")
    print("-" * 75)

    for bin_name, records in bins.items():
        df = pd.DataFrame(records)
        pct_perfect = 100.0 * df['perfect'].mean()
        avg_recall = df['recall'].mean()
        avg_cands = df['candidate_count'].mean()
        n_entities = len(df)
        print(f"{bin_name:<15} | {n_entities:<10} | {pct_perfect:<17.2f}% | {avg_recall:<12.4%} | {avg_cands:<10.2f}")

    return bins

if __name__ == '__main__':
    analyze_multimatch_breakdown()
