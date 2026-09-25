"""
Complete end-to-end baseline pipeline for Entity Resolution.
Phase 1: Build on a VALIDATION SUBSET of training data to get measured baseline score.
Then run on full test set.

Strategy:
1. TF-IDF character n-gram blocking for candidate generation
2. Multi-feature similarity scoring
3. Threshold-based matching (tuned on validation F0.5)

Designed for the massive scale: 2.2M S1, 5M S2, 5.3M S3 records.
Uses batch processing and sparse matrices for memory efficiency.
"""
import sys, io, os, time, pickle, gc, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import normalize_name, normalize_address, extract_tokens, extract_numbers, remove_legal_suffix
from evaluate import f05_single_entity, macro_f05, evaluate_predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
np.random.seed(SEED)


def load_data(source1_path, source2_path, source3_path, gt_path=None):
    """Load source files and optionally ground truth."""
    print("Loading data...")
    t0 = time.time()
    s1 = pd.read_csv(source1_path, sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(source2_path, sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(source3_path, sep='\t', dtype=str).fillna('')
    
    gt = None
    if gt_path:
        gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    
    print(f"  Loaded in {time.time()-t0:.1f}s: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")
    return s1, s2, s3, gt


def parse_ground_truth(gt_df):
    """Parse ground truth into dict: s1_id -> set of matched IDs."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))
    return gt_dict


def preprocess_dataframe(df):
    """Add normalized columns to a source dataframe."""
    df = df.copy()
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['name_no_suffix'] = df['name_norm'].apply(remove_legal_suffix)
    df['combined'] = df['name_norm'] + ' ' + df['addr_norm']
    return df


def build_tfidf_index(texts, max_features=200000, ngram_range=(2, 4)):
    """Build TF-IDF vectorizer and transform texts."""
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def batch_tfidf_candidates(query_matrix, index_matrix, top_k=50, batch_size=5000):
    """
    Find top-k candidates for each query using batched cosine similarity.
    Returns list of (query_idx, index_idx, score) tuples.
    """
    n_queries = query_matrix.shape[0]
    all_candidates = []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        
        # Compute cosine similarity for the batch
        sim = cosine_similarity(batch, index_matrix)
        
        # Get top-k for each query in the batch
        for i in range(sim.shape[0]):
            query_idx = start + i
            row = sim[i]
            # Get top-k indices
            if top_k < len(row):
                top_indices = np.argpartition(row, -top_k)[-top_k:]
                top_indices = top_indices[row[top_indices] > 0.01]  # filter near-zero
            else:
                top_indices = np.where(row > 0.01)[0]
            
            for idx in top_indices:
                all_candidates.append((query_idx, idx, row[idx]))
        
        if (start // batch_size) % 20 == 0:
            print(f"    Blocking batch {start//batch_size}: {start}/{n_queries}")
    
    return all_candidates


def compute_token_jaccard(tokens1, tokens2):
    """Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def compute_token_containment(tokens1, tokens2):
    """What fraction of tokens1 are in tokens2."""
    if not tokens1:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1)


def compute_features(s1_row, s2s3_row, tfidf_score=0.0):
    """Compute pairwise features between an S1 record and an S2/S3 record."""
    features = {}
    
    name1 = s1_row['name_norm']
    name2 = s2s3_row['name_norm']
    addr1 = s1_row['addr_norm']
    addr2 = s2s3_row['addr_norm']
    
    name1_ns = s1_row['name_no_suffix']
    name2_ns = s2s3_row['name_no_suffix']
    
    # TF-IDF retrieval score
    features['tfidf_score'] = tfidf_score
    
    # Name features
    name_tokens1 = extract_tokens(name1)
    name_tokens2 = extract_tokens(name2)
    features['name_jaccard'] = compute_token_jaccard(name_tokens1, name_tokens2)
    features['name_containment_12'] = compute_token_containment(name_tokens1, name_tokens2)
    features['name_containment_21'] = compute_token_containment(name_tokens2, name_tokens1)
    
    # Name without suffix
    ns_tokens1 = extract_tokens(name1_ns)
    ns_tokens2 = extract_tokens(name2_ns)
    features['name_ns_jaccard'] = compute_token_jaccard(ns_tokens1, ns_tokens2)
    
    # Exact name match
    features['name_exact'] = 1.0 if name1 == name2 and name1 != '' else 0.0
    features['name_ns_exact'] = 1.0 if name1_ns == name2_ns and name1_ns != '' else 0.0
    
    # Name lengths
    features['name_len_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    features['name_len1'] = len(name1)
    features['name_len2'] = len(name2)
    
    # Address features
    addr_tokens1 = extract_tokens(addr1)
    addr_tokens2 = extract_tokens(addr2)
    features['addr_jaccard'] = compute_token_jaccard(addr_tokens1, addr_tokens2)
    features['addr_containment_12'] = compute_token_containment(addr_tokens1, addr_tokens2)
    features['addr_containment_21'] = compute_token_containment(addr_tokens2, addr_tokens1)
    
    # Address number match
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    features['addr_num_jaccard'] = compute_token_jaccard(nums1, nums2)
    features['addr_has_common_num'] = 1.0 if nums1 & nums2 else 0.0
    
    # Address empty indicators
    features['addr1_empty'] = 1.0 if not addr1 else 0.0
    features['addr2_empty'] = 1.0 if not addr2 else 0.0
    features['both_addr_empty'] = 1.0 if not addr1 and not addr2 else 0.0
    
    # Country match
    c1 = s1_row.get('country', '')
    c2 = s2s3_row.get('country', '')
    features['country_match'] = 1.0 if c1 == c2 and c1 != '' else 0.0
    features['country_mismatch'] = 1.0 if c1 != c2 and c1 != '' and c2 != '' else 0.0
    
    # Source indicator
    eid = s2s3_row['entity_id']
    features['is_s2'] = 1.0 if eid.startswith('S2-') else 0.0
    features['is_s3'] = 1.0 if eid.startswith('S3-') else 0.0
    
    return features


def run_validation_baseline(sample_size=50000):
    """
    Run the baseline on a validation subset of training data.
    Returns the measured macro F0.5 score.
    """
    print("=" * 60)
    print("BASELINE PIPELINE - VALIDATION RUN")
    print("=" * 60)
    
    # Load training data
    s1, s2, s3, gt_df = load_data(
        os.path.join(BASE_DIR, 'dataset/train/train_source1.tsv'),
        os.path.join(BASE_DIR, 'dataset/train/train_source2.tsv'),
        os.path.join(BASE_DIR, 'dataset/train/train_source3.tsv'),
        os.path.join(BASE_DIR, 'dataset/train/train_ground_truth.tsv'),
    )
    
    # Parse ground truth
    print("Parsing ground truth...")
    gt_dict = parse_ground_truth(gt_df)
    
    # Split into train/val by S1 entity IDs
    all_s1_ids = list(gt_dict.keys())
    np.random.shuffle(all_s1_ids)
    
    # Sample a manageable validation set
    val_s1_ids = set(all_s1_ids[:sample_size])
    
    # Get all matched IDs from validation ground truth  
    val_gt = {sid: gt_dict[sid] for sid in val_s1_ids}
    
    # Determine which S2/S3 records could be relevant
    # For validation, we use ALL S2/S3 records (they're the "database" to search)
    val_s1 = s1[s1['entity_id'].isin(val_s1_ids)].copy()
    
    print(f"\nValidation set: {len(val_s1)} S1 entities")
    n_singletons = sum(1 for v in val_gt.values() if len(v) == 0)
    n_matched = sum(1 for v in val_gt.values() if len(v) > 0)
    total_match_ids = sum(len(v) for v in val_gt.values())
    print(f"  Singletons: {n_singletons}, Matched: {n_matched}, Total match IDs: {total_match_ids}")
    
    # Free memory
    del gt_df
    gc.collect()
    
    # Preprocess
    print("\nPreprocessing...")
    t0 = time.time()
    val_s1 = preprocess_dataframe(val_s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    print(f"  Preprocessing done in {time.time()-t0:.1f}s")
    
    # Combine S2 and S3 for blocking
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s2s3)} records")
    
    # Free individual source dataframes
    del s2, s3, s1
    gc.collect()
    
    # Build TF-IDF index on combined text (name + address)
    print("\nBuilding TF-IDF index on S2+S3...")
    t0 = time.time()
    vectorizer, s2s3_tfidf = build_tfidf_index(
        s2s3['combined'].values,
        max_features=200000,
        ngram_range=(2, 4)
    )
    print(f"  TF-IDF index built in {time.time()-t0:.1f}s, shape={s2s3_tfidf.shape}")
    
    # Transform S1 queries
    print("Transforming S1 queries...")
    s1_tfidf = vectorizer.transform(val_s1['combined'].values)
    
    # Blocking: find top-k candidates per S1 entity
    print("\nBlocking: finding candidates...")
    t0 = time.time()
    candidates_raw = batch_tfidf_candidates(
        s1_tfidf, s2s3_tfidf,
        top_k=50,
        batch_size=2000
    )
    print(f"  Blocking done in {time.time()-t0:.1f}s, {len(candidates_raw)} candidate pairs")
    
    # Free TF-IDF matrices
    del s1_tfidf, s2s3_tfidf, vectorizer
    gc.collect()
    
    # Build candidate pair index
    # candidates_raw: list of (s1_idx_in_val, s2s3_idx, tfidf_score)
    s1_id_list = val_s1['entity_id'].values
    s2s3_id_list = s2s3['entity_id'].values
    
    # Group candidates by S1 entity
    candidates_by_s1 = defaultdict(list)
    for s1_idx, s2s3_idx, score in candidates_raw:
        s1_id = s1_id_list[s1_idx]
        s2s3_id = s2s3_id_list[s2s3_idx]
        candidates_by_s1[s1_id].append((s2s3_idx, score))
    
    del candidates_raw
    gc.collect()
    
    # Measure candidate recall
    print("\nCandidate recall analysis...")
    total_gt_matches = 0
    recovered_matches = 0
    candidate_s2s3_ids_by_s1 = {}
    
    for s1_id in val_s1_ids:
        gt_matches = val_gt[s1_id]
        total_gt_matches += len(gt_matches)
        
        cand_indices = candidates_by_s1.get(s1_id, [])
        cand_ids = set(s2s3_id_list[idx] for idx, _ in cand_indices)
        candidate_s2s3_ids_by_s1[s1_id] = cand_ids
        
        recovered_matches += len(gt_matches & cand_ids)
    
    cand_recall = recovered_matches / max(total_gt_matches, 1)
    avg_candidates = np.mean([len(v) for v in candidate_s2s3_ids_by_s1.values()])
    print(f"  Candidate recall: {cand_recall:.4f} ({recovered_matches}/{total_gt_matches})")
    print(f"  Avg candidates per S1: {avg_candidates:.1f}")
    
    # Feature computation and scoring
    print("\nComputing features and scoring...")
    t0 = time.time()
    
    # Build lookup for S2/S3 rows
    s2s3_dict = {}
    for idx, row in s2s3.iterrows():
        s2s3_dict[idx] = row
    
    # Also build entity_id -> row index lookup
    s2s3_id_to_idx = {}
    for idx in range(len(s2s3)):
        s2s3_id_to_idx[s2s3_id_list[idx]] = idx
    
    # For each S1 entity, compute features for all candidates and pick matches
    # Simple threshold-based approach for baseline
    predictions = {}
    all_feature_records = []
    all_labels = []
    
    val_s1_dict = {row['entity_id']: row for _, row in val_s1.iterrows()}
    
    count = 0
    for s1_id in val_s1_ids:
        s1_row = val_s1_dict[s1_id]
        gt_matches = val_gt[s1_id]
        
        cands = candidates_by_s1.get(s1_id, [])
        
        matched_ids = set()
        for s2s3_idx, tfidf_score in cands:
            s2s3_row = s2s3_dict[s2s3_idx]
            s2s3_id = s2s3_row['entity_id']
            
            feats = compute_features(s1_row, s2s3_row, tfidf_score)
            all_feature_records.append(feats)
            all_labels.append(1 if s2s3_id in gt_matches else 0)
            
            # Simple baseline rule: high combined score
            combined_score = (
                0.3 * feats['tfidf_score'] +
                0.3 * feats['name_jaccard'] +
                0.2 * feats['name_ns_jaccard'] + 
                0.1 * feats['addr_jaccard'] +
                0.1 * feats['country_match']
            )
            
            if combined_score > 0.45:
                matched_ids.add(s2s3_id)
        
        predictions[s1_id] = matched_ids
        count += 1
        if count % 10000 == 0:
            print(f"    Scored {count}/{len(val_s1_ids)} entities")
    
    print(f"  Feature computation done in {time.time()-t0:.1f}s")
    
    # Evaluate
    print("\n" + "=" * 60)
    result = evaluate_predictions(predictions, val_gt, verbose=True)
    
    # Feature analysis
    feat_df = pd.DataFrame(all_feature_records)
    labels = np.array(all_labels)
    print(f"\nFeature matrix: {feat_df.shape}")
    print(f"Positive pairs: {labels.sum()} ({100*labels.mean():.2f}%)")
    print(f"Negative pairs: {(1-labels).sum()}")
    
    # Feature correlations with label
    print("\nFeature-label correlations:")
    for col in feat_df.columns:
        corr = feat_df[col].corr(pd.Series(labels))
        if not np.isnan(corr):
            print(f"  {col}: {corr:.4f}")
    
    # Save features for model training
    feat_df['label'] = labels
    feat_path = os.path.join(BASE_DIR, 'models', 'train_features.pkl')
    feat_df.to_pickle(feat_path)
    print(f"\nSaved feature matrix to {feat_path}")
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=50000,
                        help='Number of S1 entities for validation')
    args = parser.parse_args()
    
    result = run_validation_baseline(sample_size=args.sample_size)
    
    print(f"\n{'='*60}")
    print(f"BASELINE MACRO F0.5: {result['macro_f05']:.6f}")
    print(f"{'='*60}")
