"""
Populates dataset/train/ with 10,000 S1 records and ~30,000 S2/S3 candidate records
to allow direct end-to-end execution of baseline_v3.py commands at full 10K scale.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_harness import generate_synthetic_benchmark

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
os.makedirs(TRAIN_DIR, exist_ok=True)

def populate_10k():
    print("Generating 10,000 S1 entities and candidate universe...")
    s1_recs, s2s3_recs, gt = generate_synthetic_benchmark(n_s1=10000, singleton_rate=0.35, random_seed=42)

    s1_path = os.path.join(TRAIN_DIR, 'train_source1.tsv')
    s2_path = os.path.join(TRAIN_DIR, 'train_source2.tsv')
    s3_path = os.path.join(TRAIN_DIR, 'train_source3.tsv')
    gt_path = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

    print(f"Writing {s1_path}...")
    with open(s1_path, 'w', encoding='utf-8') as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for sid, rec in s1_recs.items():
            f.write(f"{sid}\t{rec['business_name']}\t{rec['business_address']}\t{rec['country']}\n")

    print(f"Writing {s2_path} and {s3_path}...")
    with open(s2_path, 'w', encoding='utf-8') as f2, open(s3_path, 'w', encoding='utf-8') as f3:
        f2.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f3.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for rec in s2s3_recs:
            line = f"{rec['entity_id']}\t{rec['business_name']}\t{rec['business_address']}\t{rec['country']}\n"
            if rec['entity_id'].startswith('S2-'):
                f2.write(line)
            else:
                f3.write(line)

    print(f"Writing {gt_path}...")
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, matches in gt.items():
            match_str = ",".join(sorted(matches))
            f.write(f"{sid}\t{match_str}\n")

    print("10K dataset files populated successfully.")

if __name__ == '__main__':
    populate_10k()
