"""
Inference Script V2 (Integrated with Role 2's Search Engine)
Generates the final submission on the full test dataset.
"""
import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import lightgbm as lgb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_engine import MultiPassSearchEngine
from features_enhanced import compute_all_enhanced_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
THRESHOLD = 0.90

def run_inference_v2():
    t_start = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_matches = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    out_cands = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    
    s1_path = os.path.join(BASE_DIR, 'dataset/test/test_source1.tsv')
    s2_path = os.path.join(BASE_DIR, 'dataset/test/test_source2.tsv')
    s3_path = os.path.join(BASE_DIR, 'dataset/test/test_source3.tsv')
    
    print("\n1. Loading LightGBM model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    import pickle
    with open(os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    
    print("\n2. Initializing Person 2's Advanced Search Engine...")
    t0 = time.time()
    engine = MultiPassSearchEngine()
    
    s2s3_raw = {}
    
    for path in [s2_path, s3_path]:
        print(f"   Reading {os.path.basename(path)}...")
        with open(path, encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2: continue
                eid = parts[0]
                bname = parts[1] if len(parts) > 1 else ''
                baddr = parts[2] if len(parts) > 2 else ''
                country = parts[3] if len(parts) > 3 else ''
                
                engine.add_record(eid, bname, baddr, country)
                s2s3_raw[eid] = {'entity_id': eid, 'name_norm': bname, 'addr_norm': baddr, 'country': country}
                
                if (i+1) % 2000000 == 0:
                    print(f"      {i+1:,} records loaded...")
                    
    engine.finalize_index()
    print(f"   Engine built in {time.time()-t0:.0f}s")
    gc.collect()
    
    print("\n3. Streaming S1 and predicting...")
    t0 = time.time()
    
    f_match = open(out_matches, 'w', encoding='utf-8')
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    
    f_cand = open(out_cands, 'w', encoding='utf-8')
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    s1_count = 0
    batch_features = []
    batch_metadata = []
    batch_s1_ids = []
    candidates_by_s1 = {}
    batch_size = 50000 
    
    def process_batch():
        nonlocal batch_features, batch_metadata, batch_s1_ids, candidates_by_s1
        if batch_features:
            df = pd.DataFrame(batch_features, columns=feature_cols)
            preds = model.predict(df.values)
            predictions = defaultdict(list)
            for i, (s1_id, s2s3_id) in enumerate(batch_metadata):
                if preds[i] > THRESHOLD:
                    predictions[s1_id].append(s2s3_id)
        else:
            predictions = {}
            
        for sid in batch_s1_ids:
            matched_str = ','.join(sorted(set(predictions.get(sid, []))))
            f_match.write(f"{sid}\t{matched_str}\n")
            cand_str = ','.join(sorted(set(candidates_by_s1.get(sid, []))))
            f_cand.write(f"{sid}\t{cand_str}\n")
            
        batch_features.clear()
        batch_metadata.clear()
        batch_s1_ids.clear()
        candidates_by_s1.clear()
    
    with open(s1_path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            bname = parts[1] if len(parts) > 1 else ''
            baddr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''
            
            s1_rec = {'entity_id': s1_id, 'business_name': bname, 'business_address': baddr, 'country': country}
            cands = engine.retrieve(s1_rec, top_k=10)
            cand_ids = []
            
            s1_rec_norm = {'name_norm': bname, 'addr_norm': baddr, 'country': country}
            
            for cid, block_score in cands:
                cand_ids.append(cid)
                s2s3_rec = s2s3_raw[cid]
                feats = compute_all_enhanced_features(s1_rec_norm, s2s3_rec)
                feats['block_score'] = block_score
                
                feat_row = [feats.get(c, 0.0) for c in feature_cols]
                batch_features.append(feat_row)
                batch_metadata.append((s1_id, cid))
                
            candidates_by_s1[s1_id] = cand_ids
            batch_s1_ids.append(s1_id)
            s1_count += 1
            
            if len(batch_features) >= batch_size:
                process_batch()
                
            if s1_count % 10000 == 0:
                elapsed = time.time()-t0
                rate = s1_count / elapsed
                rem = (1732544 - s1_count) / rate
                print(f"      {s1_count:,} S1 processed | Time remaining: {rem/60:.1f} minutes", flush=True)
                
        process_batch()
    
    f_match.close()
    f_cand.close()
    print(f"\n   Inference completed in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    run_inference_v2()
