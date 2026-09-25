"""
Generate official reproducible validation splits for Amazon ML Challenge 2026.
Person 4 Deliverable.

Run:
    python experiments/generate_splits.py
"""

import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.split_manager import load_s1_metadata_and_gt, generate_stratified_splits
from src.config import TRAIN_S1, TRAIN_GT


def main():
    print("=" * 65)
    print("   GENERATING REPRODUCIBLE LEAKAGE-FREE VALIDATION SPLITS   ")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "splits")

    entities = load_s1_metadata_and_gt(TRAIN_S1, TRAIN_GT)

    paths = generate_stratified_splits(
        entities=entities,
        holdout_ratio=0.15,      # 15% sealed frozen holdout (~331k entities)
        fast_dev_size=50000,     # 50k stratified sample for sub-minute dev cycles
        n_cv_folds=5,            # 5-fold CV for full training
        random_seed=42,          # Fixed reproducible seed
        output_dir=output_dir,
    )

    print("\nSplits generated successfully in:")
    for k, v in paths.items():
        print(f"  {k:<18}: {v}")


if __name__ == "__main__":
    main()
