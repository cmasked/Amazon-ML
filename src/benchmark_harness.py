"""
Comprehensive Benchmark Harness for Role 2: Search Engine & Blocking Optimizer.
Measures baseline candidate recall, benchmarks individual and multi-pass retrieval methods,
tracks marginal true match recovery, memory, runtime, and logs to experiments/role2_experiments.csv.
"""

import sys
import os
import time
import random
import tracemalloc
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization import (
    normalize_name_standard,
    strip_legal_suffixes,
    normalize_name_compact,
    normalize_address_standard,
    extract_numbers,
    extract_char_ngrams,
    build_record_representations
)
from candidate_evaluator import evaluate_candidate_generation, format_evaluation_summary
from search_engine import MultiPassSearchEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)
CSV_TRACKER_PATH = os.path.join(EXP_DIR, 'role2_experiments.csv')


def generate_synthetic_benchmark(
    n_s1: int = 2000,
    singleton_rate: float = 0.35,
    random_seed: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Set[str]]]:
    """
    Generates a realistic competition-grade benchmark dataset reflecting
    the exact schema, multi-country distribution (US, India, France),
    and noise patterns specified in the competition problem statement.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    us_streets = ["Main St", "Broadway", "Market St", "Washington Ave", "5th Ave", "Peachtree Rd", "Industrial Blvd", "Park Ave"]
    us_cities = [("New York", "NY", "10001"), ("San Francisco", "CA", "94105"), ("Chicago", "IL", "60607"), ("Austin", "TX", "78701"), ("Seattle", "WA", "98101")]
    
    in_streets = ["MG Road", "Brigade Road", "Nehru Marg", "Connaught Place", "Ring Road", "Station Road", "Anna Salai", "SV Road"]
    in_cities = [("Bangalore", "Karnataka", "560001"), ("Mumbai", "Maharashtra", "400001"), ("Delhi", "Delhi", "110001"), ("Hyderabad", "Telangana", "500001"), ("Chennai", "Tamil Nadu", "600001")]
    
    fr_streets = ["Rue de Rivoli", "Boulevard Saint-Germain", "Avenue des Champs-Elysees", "Rue de la Paix", "Boulevard Haussmann"]
    fr_cities = [("Paris", "Ile-de-France", "75001"), ("Lyon", "Auvergne-Rhone-Alpes", "69001"), ("Marseille", "PACA", "13001")]

    base_business_roots = [
        ("Acme Logistics", "Logistics"),
        ("Apex Global Solutions", "Technology"),
        ("Sharma Traders", "Commerce"),
        ("Patel Engineering Works", "Engineering"),
        ("Horizon Healthcare", "Healthcare"),
        ("Bluebird Coffee Roasters", "Food"),
        ("Titanium Security Services", "Security"),
        ("Vanguard Financial Advisors", "Finance"),
        ("Zenith Software Labs", "Technology"),
        ("Greenfield Agricultural Supplies", "Agriculture"),
        ("Metro Freight Forwarders", "Logistics"),
        ("Redwood Construction", "Construction"),
        ("Pinnacle Marketing Group", "Marketing"),
        ("Silverline Electricals", "Retail"),
        ("Crown Jewels and Gems", "Jewelry"),
        ("Paramount Auto Spares", "Automotive"),
        ("Omega Chemical Industries", "Manufacturing"),
        ("Universal Packaging Solutions", "Packaging"),
        ("Summit Hospitality", "Hospitality"),
        ("Matrix Data Analytics", "Technology"),
        ("Societe Generale Transport", "Logistics"),
        ("Boutique Elegance", "Retail"),
        ("Atelier Du Pain", "Food"),
        ("Comptoir Industriel", "Manufacturing")
    ]

    s1_records = {}
    s2s3_records = []
    ground_truth = defaultdict(set)

    s2_counter = 1
    s3_counter = 1

    noise_types = [
        "suffix_corp_corporation",
        "suffix_pvt_ltd",
        "suffix_inc",
        "spelling_typo",
        "token_reorder",
        "address_abbrev",
        "landmark_added",
        "missing_postal",
        "char_concatenation",
        "slight_address_variation"
    ]

    for i in range(1, n_s1 + 1):
        s1_id = f"S1-{i:06d}"
        root_name, category = random.choice(base_business_roots)
        root_name = f"{root_name} {i % 100}"

        country_roll = random.random()
        if country_roll < 0.45:
            country = "US"
            street = random.choice(us_streets)
            city, state, pin = random.choice(us_cities)
            num = random.randint(10, 9999)
            addr = f"{num} {street}, {city}, {state} {pin}"
            legal_suffix = random.choice(["Corporation", "Inc.", "LLC", "Company"])
        elif country_roll < 0.85:
            country = "India"
            street = random.choice(in_streets)
            city, state, pin = random.choice(in_cities)
            num = random.randint(1, 400)
            addr = f"No. {num}, {street}, Near SBI Bank, {city} {pin}"
            legal_suffix = random.choice(["Private Limited", "Pvt Ltd", "Limited", "and Sons"])
        else:
            country = "France"
            street = random.choice(fr_streets)
            city, state, pin = random.choice(fr_cities)
            num = random.randint(1, 150)
            addr = f"{num} {street}, {pin} {city}"
            legal_suffix = random.choice(["SARL", "SAS", "SA", "EURL"])

        s1_full_name = f"{root_name} {legal_suffix}"
        s1_records[s1_id] = {
            'entity_id': s1_id,
            'business_name': s1_full_name,
            'business_address': addr,
            'country': country,
            **build_record_representations(s1_id, s1_full_name, addr, country)
        }

        # Determine number of matches
        if random.random() < singleton_rate:
            # Singleton: 0 matches
            ground_truth[s1_id] = set()
            continue

        n_matches = random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10])[0]

        for m_idx in range(n_matches):
            use_s2 = (random.random() < 0.5)
            if use_s2:
                cand_id = f"S2-{s2_counter:06d}"
                s2_counter += 1
            else:
                cand_id = f"S3-{s3_counter:06d}"
                s3_counter += 1

            ground_truth[s1_id].add(cand_id)

            # Apply realistic perturbations to candidate
            applied_noise = random.choice(noise_types)
            c_name = root_name
            c_addr = addr

            if applied_noise == "suffix_corp_corporation":
                c_name = f"{root_name} Corp"
            elif applied_noise == "suffix_pvt_ltd":
                c_name = f"{root_name} Pvt. Ltd."
            elif applied_noise == "suffix_inc":
                c_name = f"{root_name} Incorporated"
            elif applied_noise == "spelling_typo":
                # Inject 1 character typo into name
                if len(root_name) > 5:
                    pos = random.randint(2, len(root_name) - 2)
                    c_name = root_name[:pos] + root_name[pos+1] + root_name[pos] + root_name[pos+2:]
                else:
                    c_name = root_name + "s"
            elif applied_noise == "token_reorder":
                parts = root_name.split()
                if len(parts) >= 2:
                    c_name = " ".join(reversed(parts))
            elif applied_noise == "address_abbrev":
                c_addr = addr.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave").replace("Boulevard", "Blvd")
            elif applied_noise == "landmark_added":
                c_addr = f"Opposite City Mall, {addr}"
            elif applied_noise == "missing_postal":
                # Remove PIN / ZIP code
                c_addr = re_clean = "".join([c for c in addr if not c.isdigit()]).strip().rstrip(',')
            elif applied_noise == "char_concatenation":
                parts = root_name.split()
                if len(parts) >= 2:
                    c_name = parts[0] + "-" + "".join(parts[1:])
            else:
                c_addr = addr.replace("No.", "Plot").replace("Suite", "Ste")

            # Append legal suffix variation
            if "Corp" not in c_name and "Ltd" not in c_name and "Inc" not in c_name and "SARL" not in c_name:
                alt_suffix = random.choice(["Co.", "Enterprises", "Group", "Services", "L.L.C.", "SAS"])
                c_name = f"{c_name} {alt_suffix}"

            s2s3_records.append({
                'entity_id': cand_id,
                'business_name': c_name,
                'business_address': c_addr,
                'country': country,
                'noise_type': applied_noise,
                'matched_s1': s1_id
            })

    # Add realistic hard-negative distractors (businesses in same city or with common business words)
    n_distractors = int(len(s2s3_records) * 0.4)
    for d in range(n_distractors):
        use_s2 = (random.random() < 0.5)
        if use_s2:
            cand_id = f"S2-{s2_counter:06d}"
            s2_counter += 1
        else:
            cand_id = f"S3-{s3_counter:06d}"
            s3_counter += 1

        root_name, _ = random.choice(base_business_roots)
        distractor_name = f"{root_name} Global Logistics {random.randint(500, 999)}"
        c_country = random.choice(["US", "India", "France"])
        if c_country == "US":
            d_addr = f"{random.randint(100, 9999)} Main St, New York, NY 10001"
        elif c_country == "India":
            d_addr = f"Plot {random.randint(1, 500)}, MG Road, Bangalore 560001"
        else:
            d_addr = f"{random.randint(1, 100)} Rue de Rivoli, 75001 Paris"

        s2s3_records.append({
            'entity_id': cand_id,
            'business_name': distractor_name,
            'business_address': d_addr,
            'country': c_country,
            'noise_type': 'distractor',
            'matched_s1': None
        })

    # Shuffle candidate records
    random.shuffle(s2s3_records)
    return s1_records, s2s3_records, dict(ground_truth)


def run_legacy_baseline(
    s1_records: Dict[str, Dict[str, Any]],
    s2s3_records: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    top_k: int = 80
) -> Dict[str, Any]:
    """
    Executes the exact legacy baseline as implemented in baseline_v3.py:
    build_inverted_index() with exact name and address tokens, and find_candidates_from_index().
    """
    tracemalloc.start()
    t0 = time.time()

    # Pass 1: Build inverted index
    token_index = defaultdict(set)
    for rec in s2s3_records:
        eid = rec['entity_id']
        name_norm = normalize_name_standard(rec['business_name'])
        core_name, _ = strip_legal_suffixes(name_norm)
        for token in core_name.split():
            if len(token) >= 3:
                token_index[token].add(eid)
        addr_norm = normalize_address_standard(rec['business_address'])
        for token in addr_norm.split():
            if len(token) >= 5:  # legacy threshold
                token_index[token].add(eid)

    # Pruning legacy
    pruned_index = {}
    for t, ids in token_index.items():
        if len(ids) <= 50000:
            pruned_index[t] = ids
    del token_index

    t_index = time.time() - t0

    # Pass 2: Querying
    t_query_start = time.time()
    candidates_by_s1 = {}
    for s1_id, s1_rec in s1_records.items():
        name_tokens = list(s1_rec['core_tokens'])
        addr_tokens = list(s1_rec['addr_tokens'])

        scores = Counter()
        for token in name_tokens:
            if token in pruned_index:
                posting = pruned_index[token]
                weight = 1.0 / (1.0 + np.log1p(len(posting)))
                for cid in posting:
                    scores[cid] += weight * 2.0
        for token in addr_tokens:
            if token in pruned_index:
                posting = pruned_index[token]
                weight = 1.0 / (1.0 + np.log1p(len(posting)))
                for cid in posting:
                    scores[cid] += weight

        top_cands = {cid for cid, _ in scores.most_common(top_k)}
        candidates_by_s1[s1_id] = top_cands

    t_query = time.time() - t_query_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timing = {
        'index_time_s': t_index,
        'query_time_s': t_query,
        'total_time_s': t_index + t_query
    }
    eval_res = evaluate_candidate_generation(
        candidates_by_s1, ground_truth, timing_stats=timing, peak_ram_mb=peak_mem / (1024 * 1024)
    )
    return eval_res, candidates_by_s1


def run_experiment_suite():
    """
    Executes all individual blocking benchmarks and multi-pass retrieval combinations.
    Logs every experiment to experiments/role2_experiments.csv and missed matches to role2_missed_matches.csv.
    """
    print("=" * 70)
    print("ROLE 2: CANDIDATE GENERATION BENCHMARK SUITE")
    print("=" * 70)

    # Check for real data
    real_train_s1 = os.path.join(BASE_DIR, 'dataset/train/train_source1.tsv')
    has_real_data = os.path.exists(real_train_s1)

    if has_real_data:
        print("Real competition data detected. Running on local validation split...")
        s1_df = pd.read_csv(real_train_s1, sep='\t', dtype=str).fillna('')
        s1_records = {}
        for r in s1_df.head(2000).to_dict('records'):
            eid = r['entity_id']
            bname = r.get('business_name', '')
            baddr = r.get('business_address', '')
            cntry = r.get('country', '')
            s1_records[eid] = {
                'entity_id': eid,
                'business_name': bname,
                'business_address': baddr,
                'country': cntry,
                **build_record_representations(eid, bname, baddr, cntry)
            }
        s2_path = os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv')
        s3_path = os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv')
        gt_path = os.path.join(BASE_DIR, 'dataset/train/train_ground_truth.tsv')
        s2s3_records = []
        if os.path.exists(s2_path):
            s2s3_records.extend(pd.read_csv(s2_path, sep='\t', dtype=str).fillna('').to_dict('records'))
        if os.path.exists(s3_path):
            s2s3_records.extend(pd.read_csv(s3_path, sep='\t', dtype=str).fillna('').to_dict('records'))
        from baseline_v3 import load_ground_truth
        ground_truth = load_ground_truth(gt_path) if os.path.exists(gt_path) else {}
        print(f"Loaded {len(s1_records)} S1 entities, {len(s2s3_records)} candidate records.")
    else:
        print("Competition dataset not present on local disk. Generating high-fidelity benchmark...")
        s1_records, s2s3_records, ground_truth = generate_synthetic_benchmark(n_s1=2000, singleton_rate=0.35, random_seed=42)
        print(f"Generated {len(s1_records)} S1 entities ({sum(1 for v in ground_truth.values() if not v)} singletons), {len(s2s3_records)} candidate records.")

    all_experiments = []
    recovered_tracker: Dict[str, Set[Tuple[str, str]]] = {}

    # --- 1. RUN LEGACY BASELINE ---
    print("\n[1/10] Running Legacy Baseline (Exact token overlap + top-k)...")
    baseline_eval, baseline_cands = run_legacy_baseline(s1_records, s2s3_records, ground_truth, top_k=80)
    base_metrics = baseline_eval['metrics']
    print(format_evaluation_summary(base_metrics))

    base_recovered = set()
    for s1_id, gt_set in ground_truth.items():
        retrieved = gt_set & baseline_cands.get(s1_id, set())
        for r in retrieved:
            base_recovered.add((s1_id, r))
    recovered_tracker['EXP-01_Baseline'] = base_recovered

    all_experiments.append({
        'experiment_id': 'EXP-01',
        'method': 'Legacy Token Inverted Index Baseline',
        'parameters': 'name_tokens>=3, addr_tokens>=5, max_posting=50000, top_k=80',
        'candidate_recall': base_metrics['candidate_recall'],
        'full_entity_recall': base_metrics['full_entity_recall'],
        'mean_candidates': base_metrics['mean_candidates'],
        'median_candidates': base_metrics['median_candidates'],
        'p95_candidates': base_metrics['p95_candidates'],
        'p99_candidates': base_metrics['p99_candidates'],
        'max_candidates': base_metrics['max_candidates'],
        'total_candidates': base_metrics['total_candidates'],
        'runtime_seconds': base_metrics['total_time_s'],
        'peak_memory_mb': base_metrics['peak_memory_mb'],
        'unique_true_matches_recovered': 0,
        'notes': 'Initial baseline measurement on standardized validation setup'
    })

    # --- 2. INDIVIDUAL BLOCKING PASSES ---
    # Construct Full MultiPassSearchEngine
    tracemalloc.start()
    t_idx0 = time.time()
    engine = MultiPassSearchEngine(
        enable_exact_name=True,
        enable_exact_addr=True,
        enable_char_ngram=True,
        enable_token_tfidf=True,
        enable_rare_tokens=True,
        enable_phonetic=True,
        enable_addr_numeric=True,
        ngram_range=(3, 3),
        max_posting_size=25000,
        rare_token_cutoff=100,
        default_top_k=80
    )
    for rec in s2s3_records:
        engine.add_record(rec['entity_id'], rec['business_name'], rec['business_address'], rec['country'])
    engine.finalize_index()
    t_idx_full = time.time() - t_idx0
    _, peak_idx_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    individual_methods = [
        ('EXP-02', 'Exact Normalized Name', {'exact_name'}),
        ('EXP-03', 'Exact Normalized Address', {'exact_addr'}),
        ('EXP-04', 'Word Token TF-IDF Overlap', {'token_tfidf'}),
        ('EXP-05', 'Rare Token Blocking', {'rare_tokens'}),
        ('EXP-06', 'Character N-Gram (3-gram) Retrieval', {'char_ngram'}),
        ('EXP-07', 'Address Numeric / Postal Code', {'addr_numeric'}),
        ('EXP-08', 'Phonetic Soundex Blocking', {'phonetic'}),
    ]

    for exp_id, name, pass_set in individual_methods:
        print(f"\nEvaluating individual method: {name}...")
        t_q0 = time.time()
        cands_map = {}
        for s1_id, s1_rec in s1_records.items():
            cands = engine.retrieve(s1_rec, top_k=80, active_passes=pass_set)
            cands_map[s1_id] = {cid for cid, _ in cands}
        t_query = time.time() - t_q0

        eval_res = evaluate_candidate_generation(
            cands_map, ground_truth,
            timing_stats={'index_time_s': t_idx_full, 'query_time_s': t_query, 'total_time_s': t_idx_full + t_query},
            peak_ram_mb=peak_idx_mem / (1024 * 1024)
        )
        m = eval_res['metrics']

        # Marginal recovery over baseline
        curr_recovered = set()
        for s1_id, gt_set in ground_truth.items():
            for r in gt_set & cands_map.get(s1_id, set()):
                curr_recovered.add((s1_id, r))
        unique_recovered = len(curr_recovered - base_recovered)

        all_experiments.append({
            'experiment_id': exp_id,
            'method': f"Individual: {name}",
            'parameters': f"passes={list(pass_set)}, top_k=80",
            'candidate_recall': m['candidate_recall'],
            'full_entity_recall': m['full_entity_recall'],
            'mean_candidates': m['mean_candidates'],
            'median_candidates': m['median_candidates'],
            'p95_candidates': m['p95_candidates'],
            'p99_candidates': m['p99_candidates'],
            'max_candidates': m['max_candidates'],
            'total_candidates': m['total_candidates'],
            'runtime_seconds': m['total_time_s'],
            'peak_memory_mb': m['peak_memory_mb'],
            'unique_true_matches_recovered': unique_recovered,
            'notes': f"Standalone single-pass performance"
        })

    # --- 3. MULTI-PASS CUMULATIVE UNION ---
    print("\n--- Running Multi-Pass Cumulative Union ---")
    cumulative_stages = [
        ('EXP-09', 'Union Pass A (Exact Name)', {'exact_name'}),
        ('EXP-10', 'Union Pass A + B (+ Exact Address)', {'exact_name', 'exact_addr'}),
        ('EXP-11', 'Union Pass A + B + C (+ Char 3-Grams)', {'exact_name', 'exact_addr', 'char_ngram'}),
        ('EXP-12', 'Union Pass A..D (+ Word TF-IDF)', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf'}),
        ('EXP-13', 'Union Pass A..E (+ Rare Tokens)', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens'}),
        ('EXP-14', 'Union Pass A..F (+ Address Postal / Numeric)', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens', 'addr_numeric'}),
        ('EXP-15', 'Full Multi-Pass Search Engine (A through G)', {'exact_name', 'exact_addr', 'char_ngram', 'token_tfidf', 'rare_tokens', 'addr_numeric', 'phonetic'}),
    ]

    prev_recovered = set()
    final_eval_res = None
    final_cands_map = None

    for exp_id, name, pass_set in cumulative_stages:
        print(f"\nEvaluating: {name}...")
        t_q0 = time.time()
        cands_map = {}
        for s1_id, s1_rec in s1_records.items():
            cands = engine.retrieve(s1_rec, top_k=80, active_passes=pass_set)
            cands_map[s1_id] = {cid for cid, _ in cands}
        t_query = time.time() - t_q0

        eval_res = evaluate_candidate_generation(
            cands_map, ground_truth,
            timing_stats={'index_time_s': t_idx_full, 'query_time_s': t_query, 'total_time_s': t_idx_full + t_query},
            peak_ram_mb=peak_idx_mem / (1024 * 1024)
        )
        m = eval_res['metrics']

        curr_recovered = set()
        for s1_id, gt_set in ground_truth.items():
            for r in gt_set & cands_map.get(s1_id, set()):
                curr_recovered.add((s1_id, r))

        marginal_gained = len(curr_recovered - prev_recovered) if prev_recovered else len(curr_recovered)
        prev_recovered = curr_recovered
        final_eval_res = eval_res
        final_cands_map = cands_map

        all_experiments.append({
            'experiment_id': exp_id,
            'method': f"Multi-Pass: {name}",
            'parameters': f"passes={len(pass_set)}, top_k=80",
            'candidate_recall': m['candidate_recall'],
            'full_entity_recall': m['full_entity_recall'],
            'mean_candidates': m['mean_candidates'],
            'median_candidates': m['median_candidates'],
            'p95_candidates': m['p95_candidates'],
            'p99_candidates': m['p99_candidates'],
            'max_candidates': m['max_candidates'],
            'total_candidates': m['total_candidates'],
            'runtime_seconds': m['total_time_s'],
            'peak_memory_mb': m['peak_memory_mb'],
            'unique_true_matches_recovered': marginal_gained,
            'notes': f"Cumulative multi-pass recall"
        })

    # Save to CSV
    exp_df = pd.DataFrame(all_experiments)
    exp_df.to_csv(CSV_TRACKER_PATH, index=False)
    print(f"\nSuccessfully written {len(exp_df)} experiment records to {CSV_TRACKER_PATH}")

    # Save missed matches
    if final_eval_res:
        missed_df = final_eval_res['missed_pairs']
        # Enrich missed matches with names and addresses
        enriched_missed = []
        s2s3_lookup = {r['entity_id']: r for r in s2s3_records}
        for _, row in missed_df.iterrows():
            s1 = s1_records.get(row['s1_id'], {})
            s2s3 = s2s3_lookup.get(row['true_source_id'], {})
            enriched_missed.append({
                's1_id': row['s1_id'],
                'true_source_id': row['true_source_id'],
                's1_name': s1.get('business_name', ''),
                'source_name': s2s3.get('business_name', ''),
                's1_address': s1.get('business_address', ''),
                'source_address': s2s3.get('business_address', ''),
                'country': s1.get('country', ''),
                'noise_type': s2s3.get('noise_type', 'unknown')
            })
        missed_enriched_df = pd.DataFrame(enriched_missed)
        missed_csv_path = os.path.join(EXP_DIR, 'role2_missed_matches.csv')
        missed_enriched_df.to_csv(missed_csv_path, index=False)
        print(f"Saved {len(missed_enriched_df)} missed matches to {missed_csv_path}")

    return exp_df, base_metrics, final_eval_res['metrics']


if __name__ == '__main__':
    run_experiment_suite()
