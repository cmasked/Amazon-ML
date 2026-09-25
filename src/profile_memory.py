"""
Process-Level Memory Profiler for Role 2 MultiPassSearchEngine.
Measures actual Operating System Resident Set Size (RSS) using psutil,
distinguishing Python overhead, dataset memory, index memory, and query buffers.
Provides an empirically grounded full-scale projection for 10.3M records.
"""

import sys
import os
import psutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark
from search_engine import MultiPassSearchEngine

def profile_memory_breakdown():
    print("=" * 65)
    print("TASK 7: REAL PROCESS MEMORY (RSS) & SCALING BREAKDOWN")
    print("=" * 65)

    proc = psutil.Process()

    # 1. Interpreter Baseline
    rss_start = proc.memory_info().rss / (1024 * 1024)
    print(f"1. Python Interpreter & Library Overhead (Base RSS): {rss_start:.2f} MB")

    # 2. Dataset Generation / Loading
    s1_recs, s2s3_recs, gt = generate_synthetic_benchmark(n_s1=5000, singleton_rate=0.35, random_seed=42)
    rss_data = proc.memory_info().rss / (1024 * 1024)
    dataset_mem = rss_data - rss_start
    print(f"2. Raw Dataset Loaded in Memory ({len(s2s3_recs):,} records): {rss_data:.2f} MB (Delta: +{dataset_mem:.2f} MB)")

    # 3. Index Construction
    t0 = time.time()
    engine = MultiPassSearchEngine(default_top_k=80)
    for r in s2s3_recs:
        engine.add_record(r['entity_id'], r['business_name'], r['business_address'], r['country'])
    engine.finalize_index()
    t_index = time.time() - t0

    rss_index = proc.memory_info().rss / (1024 * 1024)
    index_mem = rss_index - rss_data
    print(f"3. After Building MultiPassSearchEngine: {rss_index:.2f} MB (Index Delta: +{index_mem:.2f} MB)")
    print(f"   Indexed records: {len(engine.id_table):,}")

    # Inspect internal structures in index
    exact_names = len(engine.exact_name_index)
    exact_addrs = len(engine.exact_addr_index)
    ngrams = len(engine.ngram_index)
    tokens = len(engine.token_index)
    print(f"   Index Statistics: {exact_names:,} core names, {exact_addrs:,} addresses, {ngrams:,} n-grams, {tokens:,} tokens")

    # 4. Query Working Memory
    t0 = time.time()
    cands_accum = []
    for s1 in list(s1_recs.values())[:1000]:
        c = engine.retrieve(s1, top_k=80)
        cands_accum.append(c)
    t_query = time.time() - t0
    rss_query = proc.memory_info().rss / (1024 * 1024)
    query_mem = rss_query - rss_index
    print(f"4. During/After Candidate Querying: {rss_query:.2f} MB (Query Delta: +{query_mem:.2f} MB)")

    # 5. Bytes per Record & Extrapolation
    n_rec = len(engine.id_table)
    bytes_per_rec_index = (index_mem * 1024 * 1024) / max(n_rec, 1)
    print(f"\nEmpirical Index Scaling Analysis:")
    print(f"   Bytes per candidate record in index: {bytes_per_rec_index:.1f} bytes/record")

    # Project to full scale (10.3M records: 5M S2 + 5.3M S3)
    target_scale = 10_300_000
    projected_index_gb = (bytes_per_rec_index * target_scale) / (1024 ** 3)
    projected_total_rss_gb = projected_index_gb + (rss_start / 1024) + 0.5  # plus overhead + query buffer

    print(f"\nProjected Memory at Full Scale (10,300,000 Candidate Records):")
    print(f"   Projected Index Structures RAM: {projected_index_gb:.2f} GB")
    print(f"   Projected Process Peak RSS:    {projected_total_rss_gb:.2f} GB")
    print(f"   RAM Safety Margin (vs 16 GB):  {16.0 - projected_total_rss_gb:.2f} GB free")

if __name__ == '__main__':
    profile_memory_breakdown()
