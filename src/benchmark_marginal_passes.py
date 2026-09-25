"""
Systematic Marginal True-Match Recovery and Multi-Pass Contribution Benchmark.
Evaluates:
  Pass A
  Pass A + B
  Pass A + B + C
  Pass A + B + C + D
  Pass A + B + C + D + E
  Pass A + B + C + D + E + F
  Pass A + B + C + D + E + F + G
Calculates exact recall, full entity recall, candidate volume (mean, P95, P99),
and unique true matches newly recovered at each stage.
Saves to experiments/role2_marginal_passes.csv.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark
from search_engine import MultiPassSearchEngine
from candidate_evaluator import evaluate_candidate_generation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)
CSV_PATH = os.path.join(EXP_DIR, 'role2_marginal_passes.csv')


def benchmark_marginal_passes():
    print("=" * 70)
    print("TASK 4: SYSTEMATIC MULTI-PASS CUMULATIVE CONTRIBUTION BENCHMARK")
    print("=" * 70)

    s1_recs, s2s3_recs, gt = generate_synthetic_benchmark(n_s1=2000, singleton_rate=0.35, random_seed=42)

    # Build full search engine
    t0 = time.time()
    engine = MultiPassSearchEngine(default_top_k=80)
    for r in s2s3_recs:
        engine.add_record(r['entity_id'], r['business_name'], r['business_address'], r['country'])
    engine.finalize_index()
    t_index = time.time() - t0

    stages = [
        ('Pass A', 'Exact Name', {'exact_name'}),
        ('Pass A + B', '+ Exact Address', {'exact_name', 'exact_addr'}),
        ('Pass A + B + C', '+ Char 3-Grams', {'exact_name', 'exact_addr', 'char_ngram'}),
        ('Pass A..D', '+ Word TF-IDF', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf'}),
        ('Pass A..E', '+ Rare Tokens', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens'}),
        ('Pass A..F', '+ Postal / Numeric', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens', 'addr_numeric'}),
        ('Pass A..G', '+ Phonetic Soundex', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens', 'addr_numeric', 'phonetic'}),
    ]

    all_results = []
    previous_recovered: set = set()

    for stage_label, desc, active_passes in stages:
        t_q0 = time.time()
        cands_by_s1 = {}
        for s1_id, s1 in s1_recs.items():
            cands = engine.retrieve(s1, top_k=80, active_passes=active_passes)
            cands_by_s1[s1_id] = {cid for cid, _ in cands}
        t_query = time.time() - t_q0

        eval_res = evaluate_candidate_generation(
            cands_by_s1, gt,
            timing_stats={'index_time_s': t_index, 'query_time_s': t_query, 'total_time_s': t_index + t_query}
        )
        m = eval_res['metrics']

        # Determine exact recovered pairs
        current_recovered = set()
        for s1_id, gt_set in gt.items():
            for mid in gt_set & cands_by_s1.get(s1_id, set()):
                current_recovered.add((s1_id, mid))

        if not previous_recovered:
            newly_recovered = len(current_recovered)
        else:
            newly_recovered = len(current_recovered - previous_recovered)
        previous_recovered = current_recovered

        print(f"\n{stage_label} ({desc}):")
        print(f"  Candidate Recall:         {m['candidate_recall']:.4%} ({m['total_retrieved_true_matches']}/{m['total_true_matches']})")
        print(f"  Full Entity Recall:       {m['full_entity_recall']:.4%} ({m['n_full_recall']}/{m['matched_entities']})")
        print(f"  Newly Recovered Matches:  +{newly_recovered}")
        print(f"  Mean Candidates:          {m['mean_candidates']:.2f}")
        print(f"  P95 Candidates:           {m['p95_candidates']:.1f}")
        print(f"  P99 Candidates:           {m['p99_candidates']:.1f}")
        print(f"  Total Candidates:         {m['total_candidates']:,}")
        print(f"  Query Time:               {t_query:.2f}s")

        all_results.append({
            'stage': stage_label,
            'description': desc,
            'passes_count': len(active_passes),
            'candidate_recall': m['candidate_recall'],
            'full_entity_recall': m['full_entity_recall'],
            'unique_true_matches_newly_recovered': newly_recovered,
            'total_true_matches_retrieved': m['total_retrieved_true_matches'],
            'mean_candidates': m['mean_candidates'],
            'median_candidates': m['median_candidates'],
            'p95_candidates': m['p95_candidates'],
            'p99_candidates': m['p99_candidates'],
            'max_candidates': m['max_candidates'],
            'total_candidates': m['total_candidates'],
            'query_time_s': t_query,
            'total_time_s': m['total_time_s']
        })

    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved marginal pass benchmark results to {CSV_PATH}")
    return df


if __name__ == '__main__':
    benchmark_marginal_passes()
