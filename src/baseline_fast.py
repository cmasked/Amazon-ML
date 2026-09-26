"""
FAST measured baseline: Get a measured F0.5 score within 5 minutes.

Strategy: 
1. Sample 2000 S1 entities
2. Identify which S2/S3 IDs are in their ground truth (positive pairs)
3. Load ONLY those + 10x random S2/S3 negatives 
4. Run full blocking + features + scoring
5. Report measured F0.5

This gives an ACCURATE score measurement fast, because:
- The candidate recall is measurable
- The feature distributions are realistic
- The threshold tuning is valid
"""
import sys, io, os, time, gc, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict, Counter
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import normalize_name, normalize_address, extract_tokens, extract_numbers, remove_legal_suffix
from evaluate import f05_single_entity, evaluate_predictions
from features_enhanced import compute_all_enhanced_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def char_ngrams(text, n=3):
    if len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def containment(s1, s2):
    if not s1:
        return 0.0
    return len(s1 & s2) / len(s1)


def preprocess_record(eid, bname, baddr, country):
    """Preprocess a single record."""
    name_norm = normalize_name(bname)
    addr_norm = normalize_address(baddr)
    name_ns = remove_legal_suffix(name_norm)
    
    return {
        'entity_id': eid,
        'name_norm': name_norm,
        'addr_norm': addr_norm,
        'country': country,
        'name_tokens': frozenset(name_norm.split()),
        'ns_tokens': frozenset(name_ns.split()),
        'addr_tokens': frozenset(addr_norm.split()),
        'name_ngrams': frozenset(char_ngrams(name_norm, 3)),
        'addr_nums': frozenset(extract_numbers(addr_norm)),
    }


def compute_features(s1_rec, s2s3_rec):
    """Compute comprehensive pairwise features."""
    name1 = s1_rec['name_norm']
    name2 = s2s3_rec['name_norm']
    addr1 = s1_rec['addr_norm']
    addr2 = s2s3_rec['addr_norm']
    
    nt1, nt2 = s1_rec['name_tokens'], s2s3_rec['name_tokens']
    ns1, ns2 = s1_rec['ns_tokens'], s2s3_rec['ns_tokens']
    at1, at2 = s1_rec['addr_tokens'], s2s3_rec['addr_tokens']
    ng1, ng2 = s1_rec['name_ngrams'], s2s3_rec['name_ngrams']
    an1, an2 = s1_rec['addr_nums'], s2s3_rec['addr_nums']
    c1, c2 = s1_rec.get('country', ''), s2s3_rec.get('country', '')
    
    feats = {}
    
    # Token-based name features
    feats['name_jaccard'] = jaccard(nt1, nt2)
    feats['name_ns_jaccard'] = jaccard(ns1, ns2)
    feats['name_char_jaccard'] = jaccard(ng1, ng2)
    feats['name_cont_12'] = containment(nt1, nt2)
    feats['name_cont_21'] = containment(nt2, nt1)
    feats['name_exact'] = float(name1 == name2 and name1 != '')
    feats['name_ns_exact'] = float(ns1 == ns2 and len(ns1) > 0)
    feats['name_len_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    feats['name_token_diff'] = abs(len(nt1) - len(nt2))
    
    # Rapidfuzz name features
    feats['name_fuzz_ratio'] = fuzz.ratio(name1, name2) / 100.0
    feats['name_fuzz_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    feats['name_fuzz_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    feats['name_fuzz_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    
    # Token-based address features
    feats['addr_jaccard'] = jaccard(at1, at2)
    feats['addr_cont_12'] = containment(at1, at2)
    feats['addr_cont_21'] = containment(at2, at1)
    feats['addr_num_jaccard'] = jaccard(an1, an2)
    feats['addr_common_num'] = float(bool(an1 and an2 and (an1 & an2)))
    feats['addr_num_contradiction'] = float(bool(an1 and an2 and not (an1 & an2)))
    feats['addr_char_jaccard'] = jaccard(char_ngrams(addr1, 3), char_ngrams(addr2, 3))
    
    # Rapidfuzz address features
    feats['addr_fuzz_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    feats['addr_fuzz_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    feats['addr_fuzz_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    
    # Country
    feats['country_match'] = float(c1 == c2 and c1 != '')
    feats['country_mismatch'] = float(c1 != c2 and c1 != '' and c2 != '')
    
    # Source type
    feats['is_s3'] = float(s2s3_rec['entity_id'].startswith('S3-'))
    
    # Empty indicators
    feats['addr1_empty'] = float(not addr1)
    feats['addr2_empty'] = float(not addr2)
    
    # Combined score
    combined = (
        0.20 * feats['name_fuzz_token_sort'] +
        0.15 * feats['name_ns_jaccard'] +
        0.15 * feats['name_char_jaccard'] +
        0.10 * feats['name_fuzz_ratio'] +
        0.10 * feats['addr_fuzz_token_sort'] +
        0.10 * feats['addr_jaccard'] +
        0.05 * feats['addr_num_jaccard'] +
        0.05 * feats['country_match'] +
        0.05 * feats['name_exact'] +
        0.05 * feats['name_cont_12']
    )
    feats['combined_score'] = combined
    
    return feats


def load_ground_truth(path):
    gt = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            matched = parts[1] if len(parts) > 1 else ''
            gt[s1_id] = set(matched.split(',')) if matched.strip() else set()
    return gt


def load_records_by_ids(path, target_ids):
    """Load specific records from a source file."""
    records = {}
    target = set(target_ids)
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            if eid in target:
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                records[eid] = preprocess_record(eid, bname, baddr, country)
                target.discard(eid)
                if not target:
                    break
    return records


def load_random_negatives(path, n, exclude_ids):
    """Load n random records, excluding specific IDs."""
    # First pass: count lines
    total = 0
    with open(path, encoding='utf-8') as f:
        f.readline()
        for _ in f:
            total += 1
    
    # Select random line indices
    indices = set(random.sample(range(total), min(n, total)))
    
    records = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            if i in indices:
                parts = line.rstrip('\n').split('\t')
                eid = parts[0]
                if eid not in exclude_ids:
                    bname = parts[1] if len(parts) > 1 else ''
                    baddr = parts[2] if len(parts) > 2 else ''
                    country = parts[3] if len(parts) > 3 else ''
                    records[eid] = preprocess_record(eid, bname, baddr, country)
    return records


def run_fast_experiment(val_size=2000, neg_multiplier=20):
    """Get a measured F0.5 score fast."""
    print("=" * 60)
    print(f"FAST EXPERIMENT: {val_size} S1 entities, {neg_multiplier}x negatives")
    print("=" * 60)
    t_start = time.time()
    
    # Load ground truth
    print("\n1. Loading ground truth...", flush=True)
    gt = load_ground_truth(os.path.join(BASE_DIR, 'dataset/train/train_ground_truth.tsv'))
    all_ids = list(gt.keys())
    random.shuffle(all_ids)
    val_ids = all_ids[:val_size]
    val_gt = {sid: gt[sid] for sid in val_ids}
    del gt
    gc.collect()
    
    n_singletons = sum(1 for v in val_gt.values() if len(v) == 0)
    total_match_ids = sum(len(v) for v in val_gt.values())
    print(f"   {val_size} entities, {n_singletons} singletons, {total_match_ids} match IDs", flush=True)
    
    # Identify all positive S2/S3 IDs
    all_positive_ids = set()
    for matches in val_gt.values():
        all_positive_ids.update(matches)
    s2_pos = {mid for mid in all_positive_ids if mid.startswith('S2-')}
    s3_pos = {mid for mid in all_positive_ids if mid.startswith('S3-')}
    print(f"   Positive S2 IDs: {len(s2_pos)}, S3 IDs: {len(s3_pos)}", flush=True)
    
    # Load S1 records
    print("\n2. Loading S1 records...", flush=True)
    s1_recs = load_records_by_ids(
        os.path.join(BASE_DIR, 'dataset/train/train_source1.tsv'),
        val_ids
    )
    print(f"   Loaded {len(s1_recs)} S1 records", flush=True)
    
    # Load positive S2/S3 records
    print("\n3. Loading positive S2/S3 records...", flush=True)
    s2_pos_recs = load_records_by_ids(
        os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv'),
        s2_pos
    )
    s3_pos_recs = load_records_by_ids(
        os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv'),
        s3_pos
    )
    print(f"   Loaded {len(s2_pos_recs)} S2 positives, {len(s3_pos_recs)} S3 positives", flush=True)
    
    # Load random negative S2/S3 records (for realistic scoring)
    n_neg = len(all_positive_ids) * neg_multiplier
    print(f"\n4. Loading ~{n_neg} negative S2/S3 records...", flush=True)
    t0 = time.time()
    s2_neg = load_random_negatives(
        os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv'),
        n_neg // 2, all_positive_ids
    )
    s3_neg = load_random_negatives(
        os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv'),
        n_neg // 2, all_positive_ids
    )
    print(f"   Loaded {len(s2_neg)} S2 negatives, {len(s3_neg)} S3 negatives in {time.time()-t0:.0f}s", flush=True)
    
    # Combine all S2/S3 records
    all_s2s3 = {**s2_pos_recs, **s3_pos_recs, **s2_neg, **s3_neg}
    del s2_pos_recs, s3_pos_recs, s2_neg, s3_neg
    gc.collect()
    print(f"   Total S2/S3 pool: {len(all_s2s3)}", flush=True)
    
    # Build inverted index on the pool
    print("\n5. Building inverted index on pool...", flush=True)
    token_to_ids = defaultdict(set)
    for eid, rec in all_s2s3.items():
        for token in rec['ns_tokens']:
            if len(token) >= 3:
                token_to_ids[token].add(eid)
        for token in rec['addr_tokens']:
            if len(token) >= 5:
                token_to_ids[token].add(eid)
    
    # Prune overly common
    to_del = [t for t, ids in token_to_ids.items() if len(ids) > len(all_s2s3) * 0.3]
    for t in to_del:
        del token_to_ids[t]
    print(f"   Index: {len(token_to_ids)} tokens (pruned {len(to_del)})", flush=True)
    
    # Blocking + Feature computation + Scoring
    print("\n6. Blocking + Features + Scoring...", flush=True)
    t0 = time.time()
    
    predictions = {}
    candidate_sets = {}
    all_features = []
    all_labels = []
    recovered = 0
    total_candidates = 0
    
    for i, s1_id in enumerate(val_ids):
        s1_rec = s1_recs.get(s1_id)
        if not s1_rec:
            predictions[s1_id] = set()
            candidate_sets[s1_id] = set()
            continue
        
        gt_matches = val_gt[s1_id]
        
        # Find candidates via inverted index
        cand_scores = Counter()
        for token in s1_rec['ns_tokens']:
            if token in token_to_ids and len(token) >= 3:
                for cid in token_to_ids[token]:
                    cand_scores[cid] += 2.0
        for token in s1_rec['addr_tokens']:
            if token in token_to_ids and len(token) >= 5:
                for cid in token_to_ids[token]:
                    cand_scores[cid] += 1.0
        
        # Top-k candidates
        top_k = 100
        cands = cand_scores.most_common(top_k)
        cand_ids = {cid for cid, _ in cands}
        
        candidate_sets[s1_id] = cand_ids
        total_candidates += len(cand_ids)
        recovered += len(gt_matches & cand_ids)
        
        # Compute features and score
        matched = set()
        for cand_id, block_score in cands:
            if cand_id not in all_s2s3:
                continue
            
            feats = compute_all_enhanced_features(s1_rec, all_s2s3[cand_id])
            feats['block_score'] = block_score
            all_features.append(feats)
            all_labels.append(1 if cand_id in gt_matches else 0)
            
            if feats['combined_score'] > 0.45:
                matched.add(cand_id)
        
        predictions[s1_id] = matched
        
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{val_size}, {time.time()-t0:.0f}s", flush=True)
    
    print(f"   Done in {time.time()-t0:.0f}s", flush=True)
    
    # Results
    cand_recall = recovered / max(total_match_ids, 1)
    avg_cands = total_candidates / max(val_size, 1)
    
    print(f"\n7. RESULTS", flush=True)
    print(f"   Candidate recall: {cand_recall:.4f} ({recovered}/{total_match_ids})", flush=True)
    print(f"   Avg candidates per S1: {avg_cands:.1f}", flush=True)
    
    result = evaluate_predictions(predictions, val_gt, verbose=True)
    
    # Feature analysis
    labels = np.array(all_labels)
    print(f"\n   Feature matrix: {len(all_features)}x{len(all_features[0]) if all_features else 0}", flush=True)
    print(f"   Positives: {labels.sum():,} ({100*labels.mean():.2f}%)", flush=True)
    
    # Threshold sweep
    print(f"\n8. THRESHOLD SWEEP", flush=True)
    best_f05 = 0
    best_thresh = 0.45
    
    for t in np.arange(0.15, 0.80, 0.025):
        preds_t = {}
        feat_idx = 0
        for s1_id in val_ids:
            s1_rec = s1_recs.get(s1_id)
            if not s1_rec:
                preds_t[s1_id] = set()
                continue
            
            cands = list(candidate_sets.get(s1_id, set()))
            matched_t = set()
            for cand_id in cands:
                if cand_id not in all_s2s3:
                    continue
                if feat_idx < len(all_features):
                    if all_features[feat_idx]['combined_score'] > t:
                        matched_t.add(cand_id)
                    feat_idx += 1
            preds_t[s1_id] = matched_t
        
        f05 = sum(f05_single_entity(preds_t.get(sid, set()), val_gt[sid]) for sid in val_ids) / len(val_ids)
        marker = " <-- BEST" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
        print(f"   threshold={t:.3f}: F0.5={f05:.6f}{marker}", flush=True)
    
    print(f"\n   Best: threshold={best_thresh:.3f}, F0.5={best_f05:.6f}", flush=True)
    
    # Save features for ML training
    import pandas as pd
    feat_df = pd.DataFrame(all_features)
    feat_df['label'] = labels
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    feat_path = os.path.join(BASE_DIR, 'models', 'train_features.pkl')
    feat_df.to_pickle(feat_path)
    print(f"\n   Saved {len(feat_df)} features to {feat_path}", flush=True)
    
    elapsed = time.time() - t_start
    print(f"\n   Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    
    return {
        'macro_f05': best_f05,
        'best_threshold': best_thresh,
        'candidate_recall': cand_recall,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', type=int, default=2000)
    parser.add_argument('--neg-mult', type=int, default=20)
    args = parser.parse_args()
    
    result = run_fast_experiment(val_size=args.val_size, neg_multiplier=args.neg_mult)
    
    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE: F0.5={result['macro_f05']:.6f}, CandRecall={result['candidate_recall']:.4f}, Thresh={result['best_threshold']:.3f}", flush=True)
    print(f"{'='*60}", flush=True)
