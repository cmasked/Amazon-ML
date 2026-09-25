"""
Dedicated Character N-Gram Retrieval Benchmark for Role 2.
Investigates 2-3 grams, 3-4 grams, 3-5 grams, and 2-5 grams
measuring recall, candidate count, memory, index build time, and query time.
"""

import sys
import os
import time
import tracemalloc
import pandas as pd
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark
from candidate_evaluator import evaluate_candidate_generation
from search_engine import MultiPassSearchEngine

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
CSV_PATH = os.path.join(EXP_DIR, 'role2_ngram_benchmarks.csv')


def benchmark_ngram_configurations():
    print("=" * 65)
    print("ROLE 2: CHARACTER N-GRAM RETRIEVAL BENCHMARK")
    print("=" * 65)

    s1_records, s2s3_records, ground_truth = generate_synthetic_benchmark(n_s1=2000, singleton_rate=0.35, random_seed=42)

    configs = [
        ('NG-01', '2-3 grams', (2, 3)),
        ('NG-02', '3-4 grams', (3, 4)),
        ('NG-03', '3-5 grams', (3, 5)),
        ('NG-04', '2-5 grams', (2, 5)),
    ]

    results = []

    for exp_id, label, ngram_range in configs:
        print(f"\nEvaluating configuration: {label} {ngram_range}...")
        tracemalloc.start()
        t0 = time.time()

        engine = MultiPassSearchEngine(
            enable_exact_name=False,
            enable_exact_addr=False,
            enable_char_ngram=True,
            enable_token_tfidf=False,
            enable_rare_tokens=False,
            enable_phonetic=False,
            enable_addr_numeric=False,
            ngram_range=ngram_range,
            default_top_k=80
        )
        for r in s2s3_records:
            engine.add_record(r['entity_id'], r['business_name'], r['business_address'], r['country'])
        engine.finalize_index()
        t_index = time.time() - t0

        t_q0 = time.time()
        cands_map = {}
        for s1_id, s1_rec in s1_records.items():
            cands = engine.retrieve(s1_rec, top_k=80, active_passes={'char_ngram'})
            cands_map[s1_id] = {cid for cid, _ in cands}
        t_query = time.time() - t_q0

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        eval_res = evaluate_candidate_generation(
            cands_map, ground_truth,
            timing_stats={'index_time_s': t_index, 'query_time_s': t_query, 'total_time_s': t_index + t_query},
            peak_ram_mb=peak_mem / (1024 * 1024)
        )
        m = eval_res['metrics']

        n_ngrams = len(engine.ngram_index)
        print(f"  Vocabulary size: {n_ngrams:,} n-grams")
        print(f"  Recall: {m['candidate_recall']:.4%}")
        print(f"  Peak Memory: {m['peak_memory_mb']:.2f} MB")
        print(f"  Total Time: {m['total_time_s']:.2f}s (Index: {t_index:.2f}s, Query: {t_query:.2f}s)")

        results.append({
            'experiment_id': exp_id,
            'ngram_range': f"{ngram_range[0]}-{ngram_range[1]}",
            'vocabulary_size': n_ngrams,
            'candidate_recall': m['candidate_recall'],
            'full_entity_recall': m['full_entity_recall'],
            'mean_candidates': m['mean_candidates'],
            'index_time_s': t_index,
            'query_time_s': t_query,
            'total_time_s': m['total_time_s'],
            'peak_memory_mb': m['peak_memory_mb'],
        })

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved character n-gram benchmark results to {CSV_PATH}")
    return df


if __name__ == '__main__':
    benchmark_ngram_configurations()
