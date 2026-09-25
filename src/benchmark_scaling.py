"""
Role 2: Controlled Memory Scaling Experiment.
Measures process-level Peak RSS across scaling checkpoints:
10K, 25K, 50K, 100K, 250K, 500K, 1M candidate records.
"""

import sys
import os
import time
import math
import array
import json
import random
import argparse
import psutil
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_engine import MultiPassSearchEngine

BUSINESS_ROOTS = [
    "Apex Global", "Titanium Tech", "Quantum Dynamics", "Zenith Logistics", "Blue Star",
    "Premier Solutions", "Vanguard Industries", "Omega Health", "Summit Energy", "Horizon Media",
    "Pinnacle Systems", "Alpha Financial", "Nexus Retail", "Matrix Bio", "Optima Capital",
    "Starlight Foods", "Silverline Construction", "Falcon Aerospace", "Crestview Holdings",
    "Ironclad Security", "Velocity Motors", "Trident Marine", "Equinox Hospitality",
    "Synergy Consulting", "Beacon Diagnostics", "Aurora BioPharma", "Cascade Telecom",
    "Frontier Robotics", "Keystone Real Estate", "Compass Minerals", "Highland Manufacturing",
    "Evergreen Agriculture", "Radiant Cleaners", "Boutique Elegance", "Atelier Du Pain",
    "Comptoir Industriel", "Kalyan Textiles", "Tata Consulting", "Infosys Technologies",
    "Reliance Petroleum", "Siemens Industrie", "Bayer Pharma", "Schneider Electric"
]

LEGAL_SUFFIXES = [
    "Inc", "LLC", "Corp", "Corporation", "Limited", "Ltd", "Private Limited", "Pvt Ltd",
    "Group", "Enterprises", "Services", "Holding", "GmbH", "AG", "SAS", "SARL", "Co"
]

STREETS = [
    "Main St", "Park Ave", "Broadway", "Industrial Blvd", "Market St", "Peachtree Rd",
    "Michigan Ave", "Sunset Blvd", "MG Road", "Brigade Road", "SV Road", "Nehru Place",
    "Ring Road", "Anna Salai", "Rue de Rivoli", "Boulevard Haussmann", "Avenue Montaigne",
    "Friedrichstrasse", "Kurfurstendamm", "Oxford St", "High St"
]

CITIES = [
    ("New York", "NY", "10001", "US"),
    ("Seattle", "WA", "98101", "US"),
    ("Chicago", "IL", "60601", "US"),
    ("Austin", "TX", "78701", "US"),
    ("Bangalore", "KA", "560001", "India"),
    ("Mumbai", "MH", "400001", "India"),
    ("Chennai", "TN", "600001", "India"),
    ("Delhi", "DL", "110001", "India"),
    ("Paris", "IDF", "75001", "France"),
    ("Lyon", "ARA", "69001", "France"),
    ("Berlin", "BE", "10115", "Germany"),
    ("Munich", "BY", "80331", "Germany"),
    ("London", "GLA", "EC1A1BB", "UK")
]


def measure_dict_memory(d: Dict[str, Any]) -> int:
    """Recursively estimate memory of a dictionary containing arrays/lists/strings/floats."""
    total = sys.getsizeof(d)
    for k, v in d.items():
        total += sys.getsizeof(k)
        if isinstance(v, (array.array, list)):
            total += sys.getsizeof(v)
            if isinstance(v, list) and v:
                total += len(v) * 8  # pointer table
        elif isinstance(v, float):
            total += sys.getsizeof(v)
    return total


def run_single_scale(n: int, seed: int = 42) -> Dict[str, Any]:
    """Build index for n records and capture process RSS and component metrics."""
    random.seed(seed)
    process = psutil.Process(os.getpid())
    
    # 1. Base RSS before building index
    base_rss = process.memory_info().rss / (1024 * 1024)
    
    engine = MultiPassSearchEngine()
    
    t0 = time.time()
    
    # Stream records directly into add_record without holding giant array of dicts
    for i in range(1, n + 1):
        eid = f"S2-{i:08d}"
        root = random.choice(BUSINESS_ROOTS)
        suffix = random.choice(LEGAL_SUFFIXES)
        # Create realistic variation
        mod = i % 1000
        name = f"{root} {mod} {suffix}"
        
        city, state, pin, country = random.choice(CITIES)
        street = random.choice(STREETS)
        num = (i * 17) % 9999 + 1
        addr = f"{num} {street}, {city}, {state} {pin}"
        
        engine.add_record(eid, name, addr, country)
        
    t_add = time.time() - t0
    
    rss_after_add = process.memory_info().rss / (1024 * 1024)
    
    t_fin_0 = time.time()
    engine.finalize_index()
    t_finalize = time.time() - t_fin_0
    
    t_total = time.time() - t0
    
    peak_rss = process.memory_info().rss / (1024 * 1024)
    incremental_rss = peak_rss - base_rss
    bytes_per_rec = (incremental_rss * 1024 * 1024) / n
    
    # Measure component sizes
    id_table_mem = sys.getsizeof(engine.id_table) + sum(sys.getsizeof(eid) for eid in engine.id_table[:min(n, 1000)]) * (n / min(n, 1000))
    id_to_idx_mem = sys.getsizeof(engine.id_to_idx)
    exact_name_mem = measure_dict_memory(engine.exact_name_index)
    compact_name_mem = measure_dict_memory(engine.compact_name_index)
    exact_addr_mem = measure_dict_memory(engine.exact_addr_index)
    token_mem = measure_dict_memory(engine.token_index) + measure_dict_memory(engine.token_idf)
    ngram_mem = measure_dict_memory(engine.ngram_index) + measure_dict_memory(engine.ngram_idf)
    postal_mem = measure_dict_memory(engine.postal_index)
    phonetic_mem = measure_dict_memory(engine.phonetic_index)
    
    component_sum_mb = (
        id_table_mem + id_to_idx_mem + exact_name_mem + compact_name_mem +
        exact_addr_mem + token_mem + ngram_mem + postal_mem + phonetic_mem
    ) / (1024 * 1024)
    
    result = {
        'n_records': n,
        'build_time_s': round(t_total, 3),
        'add_time_s': round(t_add, 3),
        'finalize_time_s': round(t_finalize, 3),
        'base_rss_mb': round(base_rss, 2),
        'rss_after_add_mb': round(rss_after_add, 2),
        'peak_rss_mb': round(peak_rss, 2),
        'incremental_rss_mb': round(incremental_rss, 2),
        'bytes_per_record': round(bytes_per_rec, 2),
        'id_table_mb': round(id_table_mem / (1024 * 1024), 2),
        'id_to_idx_mb': round(id_to_idx_mem / (1024 * 1024), 2),
        'exact_name_mb': round((exact_name_mem + compact_name_mem) / (1024 * 1024), 2),
        'exact_addr_mb': round(exact_addr_mem / (1024 * 1024), 2),
        'token_index_mb': round(token_mem / (1024 * 1024), 2),
        'ngram_index_mb': round(ngram_mem / (1024 * 1024), 2),
        'postal_index_mb': round(postal_mem / (1024 * 1024), 2),
        'phonetic_index_mb': round(phonetic_mem / (1024 * 1024), 2),
        'component_sum_mb': round(component_sum_mb, 2),
        'vocab_tokens': len(engine.token_index),
        'vocab_ngrams': len(engine.ngram_index),
        'vocab_names': len(engine.exact_name_index),
        'vocab_addrs': len(engine.exact_addr_index)
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=10000, help='Number of records to index')
    args = parser.parse_args()
    
    res = run_single_scale(args.scale)
    # Output JSON string on last line for caller to parse
    print("RESULT_JSON:" + json.dumps(res))
