"""
Memory-efficient baseline pipeline for 16GB RAM systems.

Architecture:
1. Build inverted index of S2/S3 by streaming (stores only normalized name tokens + entity_id)
2. Block S1 entities against index to find candidates  
3. For each S1 entity's candidates, load and compare only those specific S2/S3 records
4. Score and threshold
5. Output matching_results.tsv and candidate_pairs.tsv

Target: Get a measured F0.5 score on validation within 30 minutes.
"""
import sys, io, os, time, gc, json, warnings
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


class StreamingInvertedIndex:
    """
    Build an inverted index from S2/S3 files by streaming.
    Stores: token -> set of entity_ids
    Also stores: entity_id -> (normalized_name, normalized_addr, country)
    
    Memory: ~4GB for 10M records with avg 5 tokens each
    """
    
    def __init__(self):
        self.token_to_ids = defaultdict(set)  # name token -> entity IDs
        self.addr_token_to_ids = defaultdict(set)  # addr token -> entity IDs
        self.records = {}  # entity_id -> (name_norm, addr_norm, country, name_tokens, addr_tokens, name_ngrams, addr_nums, ns_tokens)
        self.n_records = 0
        
    def add_from_file(self, path, max_records=None):
        """Stream a source file and add to index."""
        count = 0
        with open(path, encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                    
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                
                name_norm = normalize_name(bname)
                addr_norm = normalize_address(baddr)
                name_ns = remove_legal_suffix(name_norm)
                
                name_tokens = extract_tokens(name_norm)
                ns_tokens = extract_tokens(name_ns)
                addr_tokens = extract_tokens(addr_norm)
                addr_nums = extract_numbers(addr_norm)
                name_ngrams = char_ngrams(name_norm, 3)
                
                # Store record
                self.records[eid] = (name_norm, addr_norm, country, 
                                     frozenset(name_tokens), frozenset(addr_tokens),
                                     frozenset(name_ngrams), frozenset(addr_nums),
                                     frozenset(ns_tokens))
                
                # Index name tokens (skip very short or very common)
                for token in name_tokens:
                    if len(token) >= 3:
                        self.token_to_ids[token].add(eid)
                
                # Index rare address tokens
                for token in addr_tokens:
                    if len(token) >= 4:
                        self.addr_token_to_ids[token].add(eid)
                
                count += 1
                self.n_records += 1
                
                if count % 500000 == 0:
                    print(f"    Indexed {count:,} records from {os.path.basename(path)}")
                
                if max_records and count >= max_records:
                    break
        
        print(f"    Done: {count:,} records from {os.path.basename(path)}")
        return count
    
    def prune_common_tokens(self, max_posting=100000):
        """Remove overly common tokens that aren't discriminative."""
        to_remove = []
        for token, ids in self.token_to_ids.items():
            if len(ids) > max_posting:
                to_remove.append(token)
        for token in to_remove:
            del self.token_to_ids[token]
        print(f"  Pruned {len(to_remove)} overly common name tokens (>{max_posting} postings)")
        
        to_remove = []
        for token, ids in self.addr_token_to_ids.items():
            if len(ids) > max_posting:
                to_remove.append(token)
        for token in to_remove:
            del self.addr_token_to_ids[token]
        print(f"  Pruned {len(to_remove)} overly common addr tokens")
        print(f"  Index: {len(self.token_to_ids)} name tokens, {len(self.addr_token_to_ids)} addr tokens")
    
    def find_candidates(self, name_tokens, addr_tokens, name_ngrams, top_k=100, min_token_overlap=1):
        """Find candidate matches using the inverted index."""
        cand_scores = Counter()
        
        # Score by shared name tokens (strongest signal)
        for token in name_tokens:
            if token in self.token_to_ids and len(token) >= 3:
                posting = self.token_to_ids[token]
                # IDF-like weighting: rare tokens matter more
                weight = 1.0 / (1.0 + np.log1p(len(posting)))
                for cid in posting:
                    cand_scores[cid] += weight * 2.0  # name tokens weighted 2x
        
        # Score by shared address tokens
        for token in addr_tokens:
            if token in self.addr_token_to_ids and len(token) >= 4:
                posting = self.addr_token_to_ids[token]
                weight = 1.0 / (1.0 + np.log1p(len(posting)))
                for cid in posting:
                    cand_scores[cid] += weight
        
        if not cand_scores:
            return []
        
        # Return top_k by score
        top = cand_scores.most_common(top_k)
        return [(cid, score) for cid, score in top]


def compute_similarity(s1_rec, s2s3_rec_tuple):
    """
    Compute similarity features between S1 record dict and S2S3 record tuple.
    s2s3_rec_tuple: (name_norm, addr_norm, country, name_tokens, addr_tokens, name_ngrams, addr_nums, ns_tokens)
    """
    name1 = s1_rec['name_norm']
    addr1 = s1_rec['addr_norm']
    nt1 = s1_rec['name_tokens']
    at1 = s1_rec['addr_tokens']
    ng1 = s1_rec['name_ngrams']
    an1 = s1_rec['addr_nums']
    ns1 = s1_rec['ns_tokens']
    c1 = s1_rec['country']
    
    name2, addr2, c2, nt2, at2, ng2, an2, ns2 = s2s3_rec_tuple
    
    # Name features
    name_jacc = jaccard(nt1, nt2)
    name_ns_jacc = jaccard(ns1, ns2)
    name_char_jacc = jaccard(ng1, ng2)
    name_cont_12 = containment(nt1, nt2)
    name_cont_21 = containment(nt2, nt1)
    name_exact = 1.0 if name1 == name2 and name1 else 0.0
    name_ns_exact = 1.0 if ns1 == ns2 and ns1 else 0.0
    
    # Length ratio
    name_len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    
    # Address features
    addr_jacc = jaccard(at1, at2)
    addr_cont_12 = containment(at1, at2)
    addr_num_jacc = jaccard(an1, an2)
    addr_common_num = 1.0 if an1 and an2 and (an1 & an2) else 0.0
    
    # Country
    country_match = 1.0 if c1 == c2 and c1 else 0.0
    country_mismatch = 1.0 if c1 != c2 and c1 and c2 else 0.0
    
    # Empty indicators
    addr1_empty = 1.0 if not addr1 else 0.0
    addr2_empty = 1.0 if not addr2 else 0.0
    
    features = {
        'name_jaccard': name_jacc,
        'name_ns_jaccard': name_ns_jacc,
        'name_char_jaccard': name_char_jacc,
        'name_cont_12': name_cont_12,
        'name_cont_21': name_cont_21,
        'name_exact': name_exact,
        'name_ns_exact': name_ns_exact,
        'name_len_ratio': name_len_ratio,
        'addr_jaccard': addr_jacc,
        'addr_cont_12': addr_cont_12,
        'addr_num_jaccard': addr_num_jacc,
        'addr_common_num': addr_common_num,
        'country_match': country_match,
        'country_mismatch': country_mismatch,
        'addr1_empty': addr1_empty,
        'addr2_empty': addr2_empty,
    }
    
    # Combined score for threshold-based baseline
    combined = (
        0.25 * name_ns_jacc +
        0.20 * name_char_jacc +
        0.15 * name_jacc +
        0.10 * addr_jacc +
        0.10 * addr_num_jacc +
        0.05 * country_match +
        0.05 * name_exact +
        0.05 * name_cont_12 +
        0.05 * name_cont_21
    )
    features['combined_score'] = combined
    
    return features


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


def load_s1_records(path, entity_ids=None):
    """Load S1 records, optionally filtered to specific IDs."""
    records = {}
    with open(path, encoding='utf-8') as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            if entity_ids is not None and eid not in entity_ids:
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
                'name_tokens': extract_tokens(name_norm),
                'ns_tokens': extract_tokens(name_ns),
                'addr_tokens': extract_tokens(addr_norm),
                'name_ngrams': char_ngrams(name_norm, 3),
                'addr_nums': extract_numbers(addr_norm),
            }
    return records


def run_validation(val_size=10000, top_k=50, threshold=0.40):
    """Run validation on a subset of training data."""
    print("=" * 60)
    print(f"VALIDATION RUN: {val_size} S1 entities, top_k={top_k}, threshold={threshold}")
    print("=" * 60)
    
    t_start = time.time()
    
    # Load ground truth (lightweight - just IDs)
    print("\n1. Loading ground truth...")
    gt = load_ground_truth(os.path.join(BASE_DIR, 'dataset/train/train_ground_truth.tsv'))
    print(f"   {len(gt):,} S1 entities in ground truth")
    
    # Sample validation entities
    all_ids = list(gt.keys())
    np.random.shuffle(all_ids)
    val_ids = set(all_ids[:val_size])
    val_gt = {sid: gt[sid] for sid in val_ids}
    del gt
    gc.collect()
    
    n_singletons = sum(1 for v in val_gt.values() if len(v) == 0)
    n_matched = val_size - n_singletons
    total_match_ids = sum(len(v) for v in val_gt.values())
    print(f"   Val: {val_size} entities, {n_singletons} singletons ({100*n_singletons/val_size:.1f}%), {total_match_ids} total match IDs")
    
    # Load S1 validation records
    print("\n2. Loading S1 validation records...")
    s1_records = load_s1_records(
        os.path.join(BASE_DIR, 'dataset/train/train_source1.tsv'),
        entity_ids=val_ids
    )
    print(f"   Loaded {len(s1_records)} S1 records")
    
    # Build inverted index from S2 and S3
    print("\n3. Building inverted index from S2+S3...")
    t0 = time.time()
    index = StreamingInvertedIndex()
    index.add_from_file(os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv'))
    index.add_from_file(os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv'))
    index.prune_common_tokens(max_posting=100000)
    print(f"   Index built in {time.time()-t0:.0f}s, {index.n_records:,} records indexed")
    
    # Blocking + scoring
    print(f"\n4. Blocking + scoring (top_k={top_k})...")
    t0 = time.time()
    
    predictions = {}
    candidate_sets = {}
    total_candidates = 0
    recovered = 0
    
    feature_rows = []
    feature_labels = []
    
    count = 0
    for s1_id in val_ids:
        s1_rec = s1_records[s1_id]
        gt_matches = val_gt[s1_id]
        
        # Find candidates
        cands = index.find_candidates(
            s1_rec['name_tokens'],
            s1_rec['addr_tokens'],
            s1_rec['name_ngrams'],
            top_k=top_k
        )
        
        cand_ids = set()
        matched = set()
        
        for cand_id, block_score in cands:
            cand_ids.add(cand_id)
            
            # Get candidate record and compute features
            rec = index.records.get(cand_id)
            if rec is None:
                continue
            
            feats = compute_similarity(s1_rec, rec)
            feats['block_score'] = block_score
            
            feature_rows.append(feats)
            feature_labels.append(1 if cand_id in gt_matches else 0)
            
            if feats['combined_score'] > threshold:
                matched.add(cand_id)
        
        predictions[s1_id] = matched
        candidate_sets[s1_id] = cand_ids
        total_candidates += len(cand_ids)
        recovered += len(gt_matches & cand_ids)
        
        count += 1
        if count % 2000 == 0:
            print(f"      {count}/{val_size} entities, {time.time()-t0:.0f}s")
    
    print(f"   Blocking + scoring done in {time.time()-t0:.0f}s")
    
    # Results
    cand_recall = recovered / max(total_match_ids, 1)
    avg_cands = total_candidates / max(val_size, 1)
    
    print(f"\n5. RESULTS")
    print(f"   Candidate recall: {cand_recall:.4f} ({recovered}/{total_match_ids})")
    print(f"   Avg candidates per S1: {avg_cands:.1f}")
    print(f"   Total candidates: {total_candidates:,}")
    
    result = evaluate_predictions(predictions, val_gt, verbose=True)
    
    # Feature analysis
    import pandas as pd
    if feature_rows:
        feat_df = pd.DataFrame(feature_rows)
        labels = np.array(feature_labels)
        print(f"\n   Feature matrix: {feat_df.shape}")
        print(f"   Positives: {labels.sum():,} ({100*labels.mean():.2f}%)")
        
        print(f"\n   Feature-label correlations:")
        for col in sorted(feat_df.columns):
            corr = feat_df[col].corr(pd.Series(labels))
            if not np.isnan(corr):
                print(f"     {col:25s}: {corr:.4f}")
        
        # Save features for ML model training
        feat_df['label'] = labels
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
        feat_df.to_pickle(os.path.join(BASE_DIR, 'models', 'train_features.pkl'))
        print(f"   Saved features to models/train_features.pkl")
    
    # Threshold sweep
    print(f"\n6. THRESHOLD SWEEP")
    best_f05 = 0
    best_thresh = threshold
    
    for t in np.arange(0.15, 0.70, 0.025):
        preds_t = {}
        feat_idx = 0
        for s1_id in val_ids:
            cands = candidate_sets.get(s1_id, set())
            matched_t = set()
            for cand_id in cands:
                if feat_idx < len(feature_rows):
                    if feature_rows[feat_idx]['combined_score'] > t:
                        matched_t.add(cand_id)
                feat_idx += 1
            preds_t[s1_id] = matched_t
        
        # Compute macro F0.5
        total_f05 = 0
        for s1_id in val_ids:
            total_f05 += f05_single_entity(preds_t.get(s1_id, set()), val_gt[s1_id])
        f05 = total_f05 / val_size
        
        marker = " <-- BEST" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
        print(f"   threshold={t:.3f}: F0.5={f05:.6f}{marker}")
    
    print(f"\n   Best threshold: {best_thresh:.3f}")
    print(f"   Best F0.5: {best_f05:.6f}")
    
    elapsed = time.time() - t_start
    print(f"\n   Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    return {
        'macro_f05': best_f05,
        'best_threshold': best_thresh,
        'candidate_recall': cand_recall,
        'avg_candidates': avg_cands,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.40)
    args = parser.parse_args()
    
    result = run_validation(
        val_size=args.val_size,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL: Macro F0.5 = {result['macro_f05']:.6f}")
    print(f"       Candidate Recall = {result['candidate_recall']:.4f}")
    print(f"       Best Threshold = {result['best_threshold']:.3f}")
    print(f"{'='*60}")
