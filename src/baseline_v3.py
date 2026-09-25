"""
Baseline v3: Ultra memory-efficient pipeline for 16GB RAM.

Key optimization: Store ONLY the minimal data needed per S2/S3 record.
Instead of storing full frozensets (expensive), store lightweight representations.

Architecture:
1. Stream S2/S3 files to build inverted index (token -> entity_ids)
2. During streaming, also store per-entity: (name_norm_hash, addr_norm_hash, country_code)  
3. For detailed feature computation: re-read only needed records from disk
4. Use a 2-pass approach: blocking first, then selective loading for features

This keeps memory under ~6GB for the full 10M record index.
"""
import sys, io, os, time, gc, struct, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import normalize_name, normalize_address, extract_tokens, extract_numbers, remove_legal_suffix
from evaluate import f05_single_entity, evaluate_predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
np.random.seed(SEED)


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


def load_ground_truth(path):
    """Load ground truth as dict: s1_id -> set of matched IDs."""
    gt = {}
    with open(path, encoding='utf-8') as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            matched = parts[1] if len(parts) > 1 else ''
            if matched.strip():
                gt[s1_id] = set(matched.split(','))
            else:
                gt[s1_id] = set()
    return gt


from search_engine import MultiPassSearchEngine, find_candidates_from_index as search_engine_find_candidates


def build_multipass_index(source_paths, top_k=80):
    """
    Role 2 Optimized Search Engine: Multi-Pass Inverted Index.
    Memory-compact integer indexing, character n-grams, legal suffix,
    exact name/address, rare tokens, and postal code passes.
    """
    engine = MultiPassSearchEngine(default_top_k=top_k)
    all_ids = set()
    total = 0
    for path in source_paths:
        print(f"  Indexing {os.path.basename(path)} into MultiPassSearchEngine...")
        count = 0
        with open(path, encoding='utf-8') as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                all_ids.add(eid)
                engine.add_record(eid, bname, baddr, country)
                count += 1
                total += 1
                if count % 1000000 == 0:
                    print(f"    {count:,} records...")
        print(f"    Done: {count:,} records")
    engine.finalize_index()
    print(f"  MultiPassSearchEngine ready: {total:,} records indexed.")
    return engine, all_ids


def build_inverted_index(source_paths, max_posting_size=50000):
    """
    Legacy inverted index builder maintained for backwards compatibility.
    """
    token_index = defaultdict(set)
    all_ids = set()
    total = 0
    
    for path in source_paths:
        print(f"  Indexing {os.path.basename(path)}...")
        count = 0
        with open(path, encoding='utf-8') as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                
                all_ids.add(eid)
                
                # Normalize and tokenize
                name_norm = normalize_name(bname)
                name_ns = remove_legal_suffix(name_norm)
                
                # Index name tokens
                for token in name_ns.split():
                    if len(token) >= 3:
                        token_index[token].add(eid)
                
                # Also index significant address tokens  
                addr_norm = normalize_address(baddr)
                for token in addr_norm.split():
                    if len(token) >= 5:  # longer threshold for address
                        token_index[token].add(eid)
                
                count += 1
                total += 1
                if count % 1000000 == 0:
                    print(f"    {count:,} records...")
        
        print(f"    Done: {count:,} records")
    
    # Prune overly common tokens
    pruned = 0
    to_del = [t for t, ids in token_index.items() if len(ids) > max_posting_size]
    for t in to_del:
        del token_index[t]
        pruned += 1
    
    print(f"  Total indexed: {total:,} records, {len(token_index):,} tokens (pruned {pruned})")
    return dict(token_index), all_ids


def find_candidates_from_index(name_tokens, addr_tokens, token_index, top_k=100, s1_rec=None):
    """Find top-k candidate entity IDs using inverted index or MultiPassSearchEngine."""
    return search_engine_find_candidates(name_tokens, addr_tokens, token_index, top_k=top_k, s1_rec=s1_rec)



def load_specific_records(source_paths, needed_ids):
    """
    Pass 2: Load only the specific records we need for feature computation.
    Returns dict: entity_id -> preprocessed record dict
    """
    needed = set(needed_ids)
    records = {}
    
    for path in source_paths:
        print(f"  Loading needed records from {os.path.basename(path)}...")
        count = 0
        with open(path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                eid = parts[0]
                if eid not in needed:
                    continue
                
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                
                name_norm = normalize_name(bname)
                addr_norm = normalize_address(baddr)
                name_ns = remove_legal_suffix(name_norm)
                
                records[eid] = {
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
                count += 1
                needed.discard(eid)
                
                if not needed:
                    break
        
        print(f"    Loaded {count:,} records")
        if not needed:
            break
    
    return records


def compute_features(s1_rec, s2s3_rec):
    """Compute all pairwise similarity features."""
    name1 = s1_rec['name_norm']
    name2 = s2s3_rec['name_norm']
    
    nt1 = s1_rec['name_tokens']
    nt2 = s2s3_rec['name_tokens']
    ns1 = s1_rec['ns_tokens']
    ns2 = s2s3_rec['ns_tokens']
    at1 = s1_rec['addr_tokens']
    at2 = s2s3_rec['addr_tokens']
    ng1 = s1_rec['name_ngrams']
    ng2 = s2s3_rec['name_ngrams']
    an1 = s1_rec['addr_nums']
    an2 = s2s3_rec['addr_nums']
    
    c1 = s1_rec.get('country', '')
    c2 = s2s3_rec.get('country', '')
    
    feats = {}
    
    # Name features
    feats['name_jaccard'] = jaccard(nt1, nt2)
    feats['name_ns_jaccard'] = jaccard(ns1, ns2)
    feats['name_char_jaccard'] = jaccard(ng1, ng2)
    feats['name_cont_12'] = containment(nt1, nt2)
    feats['name_cont_21'] = containment(nt2, nt1)
    feats['name_exact'] = float(name1 == name2 and name1 != '')
    feats['name_ns_exact'] = float(ns1 == ns2 and len(ns1) > 0)
    feats['name_len_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    feats['name_token_diff'] = abs(len(nt1) - len(nt2))
    
    # Address features
    feats['addr_jaccard'] = jaccard(at1, at2)
    feats['addr_cont_12'] = containment(at1, at2)
    feats['addr_cont_21'] = containment(at2, at1)
    feats['addr_num_jaccard'] = jaccard(an1, an2)
    feats['addr_common_num'] = float(bool(an1 and an2 and (an1 & an2)))
    feats['addr_num_contradiction'] = float(bool(an1 and an2 and not (an1 & an2)))
    
    # Empty flags
    feats['addr1_empty'] = float(not s1_rec['addr_norm'])
    feats['addr2_empty'] = float(not s2s3_rec['addr_norm'])
    
    # Country
    feats['country_match'] = float(c1 == c2 and c1 != '')
    feats['country_mismatch'] = float(c1 != c2 and c1 != '' and c2 != '')
    
    # Source type
    eid = s2s3_rec['entity_id']
    feats['is_s3'] = float(eid.startswith('S3-'))
    
    # Combined score for rule-based baseline
    combined = (
        0.25 * feats['name_ns_jaccard'] +
        0.20 * feats['name_char_jaccard'] +
        0.15 * feats['name_jaccard'] +
        0.10 * feats['addr_jaccard'] +
        0.10 * feats['addr_num_jaccard'] +
        0.05 * feats['country_match'] +
        0.05 * feats['name_exact'] +
        0.05 * feats['name_cont_12'] +
        0.05 * feats['name_cont_21']
    )
    feats['combined_score'] = combined
    
    return feats


def run_validation(val_size=10000, top_k=80, threshold=0.40, mode='train', use_multipass=True):
    """
    Two-pass validation pipeline:
    Pass 1: Build index + blocking  
    Pass 2: Load candidate records + features + scoring
    """
    print("=" * 60)
    print(f"BASELINE v3 - {mode.upper()} mode, {val_size} S1 entities, top_k={top_k}, multipass={use_multipass}")
    print("=" * 60)
    t_start = time.time()
    
    if mode == 'train':
        s1_path = os.path.join(BASE_DIR, 'dataset/train/train_source1.tsv')
        s2_path = os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv')
        s3_path = os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv')
        gt_path = os.path.join(BASE_DIR, 'dataset/train/train_ground_truth.tsv')
    else:
        s1_path = os.path.join(BASE_DIR, 'dataset/test/test_source1.tsv')
        s2_path = os.path.join(BASE_DIR, 'dataset/test/test_source2.tsv')
        s3_path = os.path.join(BASE_DIR, 'dataset/test/test_source3.tsv')
        gt_path = None
    
    # === PASS 1: Build index ===
    if use_multipass:
        print("\nPASS 1: Building Role 2 MultiPassSearchEngine index...")
        t0 = time.time()
        token_index, all_s2s3_ids = build_multipass_index([s2_path, s3_path], top_k=top_k)
        print(f"  Multi-pass index built in {time.time()-t0:.0f}s")
    else:
        print("\nPASS 1: Building legacy inverted index...")
        t0 = time.time()
        token_index, all_s2s3_ids = build_inverted_index([s2_path, s3_path])
        print(f"  Legacy index built in {time.time()-t0:.0f}s")
    
    # Load ground truth and select validation entities
    if gt_path:
        print("\nLoading ground truth...")
        gt = load_ground_truth(gt_path)
        all_s1_ids = list(gt.keys())
        np.random.shuffle(all_s1_ids)
        val_ids = all_s1_ids[:val_size]
        val_gt = {sid: gt[sid] for sid in val_ids}
        del gt
    else:
        # Test mode: use all S1 entities
        val_ids = []
        with open(s1_path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                val_ids.append(line.split('\t')[0])
        val_gt = None
    
    gc.collect()
    
    # Load S1 records for validation set
    print(f"\nLoading {len(val_ids)} S1 records...")
    s1_recs = {}
    val_id_set = set(val_ids)
    with open(s1_path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            if eid not in val_id_set:
                continue
            bname = parts[1] if len(parts) > 1 else ''
            baddr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''
            
            name_norm = normalize_name(bname)
            addr_norm = normalize_address(baddr)
            name_ns = remove_legal_suffix(name_norm)
            
            s1_recs[eid] = {
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
    print(f"  Loaded {len(s1_recs)} S1 records")
    
    # === BLOCKING: Find candidates for each S1 entity ===
    print(f"\nBlocking: finding top-{top_k} candidates per S1 entity...")
    t0 = time.time()
    candidates_by_s1 = {}  # s1_id -> list of (s2s3_id, block_score)
    needed_s2s3_ids = set()
    
    for i, s1_id in enumerate(val_ids):
        s1_rec = s1_recs[s1_id]
        
        # Get candidates from inverted index
        name_tokens = list(s1_rec['ns_tokens'])  # Use name-without-suffix tokens
        addr_tokens = list(s1_rec['addr_tokens'])
        
        cands = find_candidates_from_index(name_tokens, addr_tokens, token_index, top_k=top_k, s1_rec=s1_rec)
        
        candidates_by_s1[s1_id] = cands
        for cid, _ in cands:
            needed_s2s3_ids.add(cid)
        
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(val_ids)} entities blocked, {len(needed_s2s3_ids):,} unique candidates")
    
    print(f"  Blocking done in {time.time()-t0:.0f}s")
    print(f"  Total unique candidates needed: {len(needed_s2s3_ids):,}")
    
    # Free the large inverted index
    del token_index
    gc.collect()
    
    # Measure candidate recall
    if val_gt:
        total_gt = sum(len(v) for v in val_gt.values())
        recovered = 0
        for s1_id in val_ids:
            gt_matches = val_gt[s1_id]
            cand_ids = {cid for cid, _ in candidates_by_s1.get(s1_id, [])}
            recovered += len(gt_matches & cand_ids)
        cand_recall = recovered / max(total_gt, 1)
        print(f"\n  CANDIDATE RECALL: {cand_recall:.4f} ({recovered:,}/{total_gt:,})")
    
    # === PASS 2: Load needed S2/S3 records, compute features ===
    print(f"\nPASS 2: Loading {len(needed_s2s3_ids):,} candidate records...")
    t0 = time.time()
    s2s3_recs = load_specific_records([s2_path, s3_path], needed_s2s3_ids)
    print(f"  Loaded {len(s2s3_recs):,} records in {time.time()-t0:.0f}s")
    
    # Compute features and score
    print("\nComputing features and scoring...")
    t0 = time.time()
    
    predictions = {}
    all_features = []
    all_labels = []
    candidate_sets = {}
    
    for i, s1_id in enumerate(val_ids):
        s1_rec = s1_recs[s1_id]
        cands = candidates_by_s1.get(s1_id, [])
        gt_matches = val_gt[s1_id] if val_gt else set()
        
        cand_ids_set = set()
        matched = set()
        
        for cand_id, block_score in cands:
            cand_ids_set.add(cand_id)
            
            if cand_id not in s2s3_recs:
                continue
            
            s2s3_rec = s2s3_recs[cand_id]
            feats = compute_features(s1_rec, s2s3_rec)
            feats['block_score'] = block_score
            
            all_features.append(feats)
            if val_gt:
                all_labels.append(1 if cand_id in gt_matches else 0)
            
            if feats['combined_score'] > threshold:
                matched.add(cand_id)
        
        predictions[s1_id] = matched
        candidate_sets[s1_id] = cand_ids_set
        
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(val_ids)} entities scored")
    
    print(f"  Feature computation done in {time.time()-t0:.0f}s")
    print(f"  Feature rows: {len(all_features):,}")
    
    # Evaluate
    if val_gt:
        print("\n" + "=" * 60)
        result = evaluate_predictions(predictions, val_gt, verbose=True)
        
        # Feature analysis
        labels = np.array(all_labels)
        print(f"\n  Positive pairs: {labels.sum():,} ({100*labels.mean():.2f}%)")
        print(f"  Negative pairs: {(~labels.astype(bool)).sum():,}")
        
        # Threshold sweep
        print(f"\n  THRESHOLD SWEEP:")
        best_f05 = 0
        best_thresh = threshold
        
        # Re-compute predictions for each threshold using stored features
        feat_idx_map = {}  # (s1_id, cand_id) -> feat_index
        idx = 0
        for s1_id in val_ids:
            cands = candidates_by_s1.get(s1_id, [])
            for cand_id, _ in cands:
                if cand_id in s2s3_recs:
                    feat_idx_map[(s1_id, cand_id)] = idx
                    idx += 1
        
        for t in np.arange(0.15, 0.75, 0.025):
            preds_t = {}
            for s1_id in val_ids:
                cands = candidates_by_s1.get(s1_id, [])
                matched_t = set()
                for cand_id, _ in cands:
                    key = (s1_id, cand_id)
                    if key in feat_idx_map:
                        fi = feat_idx_map[key]
                        if all_features[fi]['combined_score'] > t:
                            matched_t.add(cand_id)
                preds_t[s1_id] = matched_t
            
            f05 = sum(f05_single_entity(preds_t.get(sid, set()), val_gt[sid]) for sid in val_ids) / len(val_ids)
            marker = " <-- BEST" if f05 > best_f05 else ""
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = t
            print(f"    threshold={t:.3f}: F0.5={f05:.6f}{marker}")
        
        print(f"\n    Best threshold: {best_thresh:.3f}")
        print(f"    Best F0.5: {best_f05:.6f}")
        
        elapsed = time.time() - t_start
        print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        
        # Save features for ML model training
        import pandas as pd
        feat_df = pd.DataFrame(all_features)
        feat_df['label'] = labels
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
        feat_df.to_pickle(os.path.join(BASE_DIR, 'models', 'train_features.pkl'))
        print(f"  Saved {len(feat_df)} feature rows to models/train_features.pkl")
        
        return {
            'macro_f05': best_f05,
            'best_threshold': best_thresh,
            'candidate_recall': cand_recall,
            'n_features': len(all_features),
        }
    else:
        # Test mode: write output files
        return predictions, candidate_sets


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=80)
    parser.add_argument('--threshold', type=float, default=0.40)
    parser.add_argument('--legacy-blocking', action='store_true', help='Use legacy token inverted index instead of MultiPassSearchEngine')
    args = parser.parse_args()
    
    result = run_validation(
        val_size=args.val_size,
        top_k=args.top_k,
        threshold=args.threshold,
        use_multipass=not args.legacy_blocking,
    )
    
    if isinstance(result, dict):
        print(f"\n{'='*60}")
        print(f"FINAL BASELINE RESULTS:")
        print(f"  Macro F0.5: {result['macro_f05']:.6f}")
        print(f"  Candidate Recall: {result['candidate_recall']:.4f}")
        print(f"  Best Threshold: {result['best_threshold']:.3f}")
        print(f"{'='*60}")
