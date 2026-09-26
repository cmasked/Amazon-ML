"""
Inference script to generate submission files on the full test set.
Handles the 16GB RAM limit by streaming and memory-efficient data structures.
"""
import sys, io, os, time, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import normalize_name, normalize_address, remove_legal_suffix, extract_numbers
from features_enhanced import compute_all_enhanced_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Optimal threshold from our validation
THRESHOLD = 0.90

def char_ngrams(text, n=3):
    if len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def preprocess_on_the_fly(eid, bname, baddr, country):
    """Preprocess a record exactly as done during training."""
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


def run_inference():
    print("=" * 60)
    print("RUNNING INFERENCE ON FULL TEST SET")
    print("=" * 60)
    t_start = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_matches = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    out_cands = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    
    s1_path = os.path.join(BASE_DIR, 'dataset/test/test_source1.tsv')
    s2_path = os.path.join(BASE_DIR, 'dataset/test/test_source2.tsv')
    s3_path = os.path.join(BASE_DIR, 'dataset/test/test_source3.tsv')
    
    # 1. Load ML Model
    print("\n1. Loading LightGBM model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    import pickle
    with open(os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    print(f"   Model loaded. Features expected: {len(feature_cols)}")
    
    # 2. Build Inverted Index and load raw S2/S3 into memory efficiently
    # To save memory, we store a flat list of tuples and a dict mapping ID -> index
    print("\n2. Loading S2/S3 and building inverted index...")
    t0 = time.time()
    s2s3_raw = []  # List of (eid, bname, baddr, country)
    id_to_idx = {}
    token_to_idxs = defaultdict(set)
    
    idx = 0
    for path in [s2_path, s3_path]:
        print(f"   Reading {os.path.basename(path)}...")
        with open(path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                
                # Store raw
                s2s3_raw.append((eid, bname, baddr, country))
                id_to_idx[eid] = idx
                
                # Build index
                name_norm = normalize_name(bname)
                name_ns = remove_legal_suffix(name_norm)
                for token in name_ns.split():
                    if len(token) >= 3:
                        token_to_idxs[token].add(idx)
                
                addr_norm = normalize_address(baddr)
                for token in addr_norm.split():
                    if len(token) >= 5:
                        token_to_idxs[token].add(idx)
                
                idx += 1
                if idx % 1000000 == 0:
                    print(f"      {idx:,} records loaded...")

    # Prune overly common tokens to save memory and speed up search
    to_del = [t for t, idxs in token_to_idxs.items() if len(idxs) > 100000]
    for t in to_del:
        del token_to_idxs[t]
    
    print(f"   Loaded {len(s2s3_raw):,} records. Index size: {len(token_to_idxs):,} tokens. Time: {time.time()-t0:.0f}s")
    gc.collect()
    
    # 3. Stream S1, find candidates, compute features, predict
    print("\n3. Streaming S1 and predicting...")
    t0 = time.time()
    
    f_match = open(out_matches, 'w', encoding='utf-8')
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    
    f_cand = open(out_cands, 'w', encoding='utf-8')
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    s1_count = 0
    batch_features = []
    batch_metadata = []  # (s1_id, s2s3_id)
    batch_size = 50000  # Number of feature rows to accumulate before predicting
    
    def process_batch():
        nonlocal batch_features, batch_metadata
        if not batch_features:
            return
            
        # Predict
        df = pd.DataFrame(batch_features, columns=feature_cols)
        preds = model.predict(df.values)
        
        # Group by S1
        predictions = defaultdict(list)
        candidates = defaultdict(list)
        
        for i, (s1_id, s2s3_id) in enumerate(batch_metadata):
            candidates[s1_id].append(s2s3_id)
            if preds[i] > THRESHOLD:
                predictions[s1_id].append(s2s3_id)
        
        # We need to write ALL S1 IDs that were in this batch, even singletons
        # The metadata holds all (s1, cand) pairs. But some S1s might have 0 candidates!
        # We handle that in the main loop instead.
        
        batch_features = []
        batch_metadata = []
        return predictions
    
    # Read S1
    with open(s1_path, encoding='utf-8') as f:
        f.readline()
        
        batch_s1_ids = []
        candidates_by_s1 = {}
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            bname = parts[1] if len(parts) > 1 else ''
            baddr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''
            
            s1_count += 1
            batch_s1_ids.append(s1_id)
            
            s1_rec = preprocess_on_the_fly(s1_id, bname, baddr, country)
            
            # Blocking
            cand_scores = Counter()
            for token in s1_rec['ns_tokens']:
                if token in token_to_idxs and len(token) >= 3:
                    for cid_idx in token_to_idxs[token]:
                        cand_scores[cid_idx] += 2.0
            for token in s1_rec['addr_tokens']:
                if token in token_to_idxs and len(token) >= 5:
                    for cid_idx in token_to_idxs[token]:
                        cand_scores[cid_idx] += 1.0
            
            cands = cand_scores.most_common(100)
            cand_ids = []
            
            for cid_idx, block_score in cands:
                raw = s2s3_raw[cid_idx]
                s2s3_rec = preprocess_on_the_fly(*raw)
                cand_ids.append(s2s3_rec['entity_id'])
                
                feats = compute_all_enhanced_features(s1_rec, s2s3_rec)
                feats['block_score'] = block_score
                
                # Align features to model
                feat_row = [feats.get(c, 0.0) for c in feature_cols]
                
                batch_features.append(feat_row)
                batch_metadata.append((s1_id, s2s3_rec['entity_id']))
            
            candidates_by_s1[s1_id] = cand_ids
            
            # If batch is full, process
            if len(batch_features) >= batch_size:
                preds = process_batch()
                
                for sid in batch_s1_ids:
                    # Write matches
                    matched_str = ','.join(sorted(set(preds.get(sid, []))))
                    f_match.write(f"{sid}\t{matched_str}\n")
                    
                    # Write candidates
                    cand_str = ','.join(sorted(set(candidates_by_s1.get(sid, []))))
                    f_cand.write(f"{sid}\t{cand_str}\n")
                
                batch_s1_ids = []
                candidates_by_s1 = {}
            
            if s1_count % 10000 == 0:
                print(f"      {s1_count:,} S1 entities processed... {time.time()-t0:.0f}s")
        
        # Process remaining
        if batch_features or batch_s1_ids:
            preds = process_batch() or {}
            for sid in batch_s1_ids:
                matched_str = ','.join(sorted(set(preds.get(sid, []))))
                f_match.write(f"{sid}\t{matched_str}\n")
                cand_str = ','.join(sorted(set(candidates_by_s1.get(sid, []))))
                f_cand.write(f"{sid}\t{cand_str}\n")
    
    f_match.close()
    f_cand.close()
    
    print(f"\n   Inference completed in {time.time()-t0:.0f}s")
    print(f"   Outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_inference()
