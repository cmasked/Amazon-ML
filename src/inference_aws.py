"""
AWS High-Performance Inference Script
Designed to run on an EC2 instance with 64GB+ RAM and multiple CPU cores.
Uses multiprocessing to parallelize the 1.7 million test entities.
"""
import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
import pandas as pd
import lightgbm as lgb
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_engine import MultiPassSearchEngine
from features_enhanced import compute_all_enhanced_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
THRESHOLD = 0.90

# Globals for worker processes
worker_engine = None
worker_s2s3 = None
worker_model = None
worker_feature_cols = None

def init_worker(engine_obj, s2s3_dict, model_path, feature_cols):
    """Initialize the shared objects in each worker process."""
    global worker_engine, worker_s2s3, worker_model, worker_feature_cols
    worker_engine = engine_obj
    worker_s2s3 = s2s3_dict
    worker_feature_cols = feature_cols
    # LightGBM Booster needs to be loaded inside the worker
    worker_model = lgb.Booster(model_file=model_path)


def process_chunk(s1_chunk):
    """Process a chunk of S1 records."""
    matches_out = []
    cands_out = []
    
    for s1_rec in s1_chunk:
        s1_id = s1_rec['entity_id']
        cands = worker_engine.retrieve(s1_rec, top_k=10)
        
        s1_rec_norm = {
            'name_norm': s1_rec['business_name'], 
            'addr_norm': s1_rec['business_address'], 
            'country': s1_rec['country']
        }
        
        features_list = []
        metadata_list = []
        cand_ids = []
        
        for cid, block_score in cands:
            cand_ids.append(cid)
            s2s3_rec = worker_s2s3[cid]
            feats = compute_all_enhanced_features(s1_rec_norm, s2s3_rec)
            feats['block_score'] = block_score
            
            feat_row = [feats.get(c, 0.0) for c in worker_feature_cols]
            features_list.append(feat_row)
            metadata_list.append(cid)
            
        matched_ids = []
        if features_list:
            df = pd.DataFrame(features_list, columns=worker_feature_cols)
            preds = worker_model.predict(df.values)
            for i, p in enumerate(preds):
                if p > THRESHOLD:
                    matched_ids.append(metadata_list[i])
                    
        matched_str = ','.join(sorted(set(matched_ids)))
        cand_str = ','.join(sorted(set(cand_ids)))
        
        matches_out.append(f"{s1_id}\t{matched_str}")
        cands_out.append(f"{s1_id}\t{cand_str}")
        
    return matches_out, cands_out


def run_aws_inference():
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    s1_path = os.path.join(BASE_DIR, 'dataset/test/test_source1.tsv')
    s2_path = os.path.join(BASE_DIR, 'dataset/test/test_source2.tsv')
    s3_path = os.path.join(BASE_DIR, 'dataset/test/test_source3.tsv')
    
    with open(os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
        
    print("\n[AWS] Loading S2/S3 and building Search Engine...")
    t0 = time.time()
    engine = MultiPassSearchEngine()
    s2s3_raw = {}
    
    for path in [s2_path, s3_path]:
        print(f"   Reading {os.path.basename(path)}...")
        with open(path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2: continue
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                
                engine.add_record(eid, bname, baddr, country)
                s2s3_raw[eid] = {'entity_id': eid, 'name_norm': bname, 'addr_norm': baddr, 'country': country}
                
    engine.finalize_index()
    print(f"   Engine built. RAM is heavily utilized. Time: {time.time()-t0:.0f}s")
    
    print("\n[AWS] Reading S1 records for multiprocessing...")
    s1_records = []
    with open(s1_path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_records.append({
                'entity_id': parts[0],
                'business_name': parts[1] if len(parts) > 1 else '',
                'business_address': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            })
            
    print(f"   Loaded {len(s1_records):,} S1 records.")
    
    # Chunking
    num_cores = mp.cpu_count()
    print(f"\n[AWS] Launching {num_cores} parallel CPU workers...")
    
    chunk_size = 5000
    chunks = [s1_records[i:i + chunk_size] for i in range(0, len(s1_records), chunk_size)]
    
    pool = mp.Pool(
        processes=num_cores, 
        initializer=init_worker, 
        initargs=(engine, s2s3_raw, MODEL_PATH, feature_cols)
    )
    
    out_matches = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    out_cands = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    
    f_match = open(out_matches, 'w', encoding='utf-8')
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    
    f_cand = open(out_cands, 'w', encoding='utf-8')
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    t0 = time.time()
    processed = 0
    total = len(s1_records)
    
    for i, (m_out, c_out) in enumerate(pool.imap(process_chunk, chunks)):
        for line in m_out: f_match.write(line + '\n')
        for line in c_out: f_cand.write(line + '\n')
        
        processed += len(chunks[i])
        print(f"   Progress: {processed:,} / {total:,} ({processed/total*100:.1f}%) | Elapsed: {time.time()-t0:.0f}s", flush=True)
        
    pool.close()
    pool.join()
    
    f_match.close()
    f_cand.close()
    print(f"\n[AWS] Inference fully completed in {time.time()-t_start:.0f}s!")

if __name__ == '__main__':
    run_aws_inference()
