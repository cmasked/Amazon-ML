"""
Universal Validation Suite Runner.
Person 4 Deliverable for Amazon ML Challenge 2026.

Run this script to evaluate either:
1. Final matching model outputs (--matching <file.tsv>) -> computes official Macro F0.5
2. Blocking / Candidate generation outputs (--candidate <file.tsv>) -> computes Candidate Recall Ceiling & Reduction Ratio

Supports splits:
- 'fast_dev' (default: 50k stratified entities, takes seconds)
- 'frozen_holdout' (331k sealed entities, used ONLY for final pre-submission check)
- 'full' (all 2.2M training entities)
- 'loco_us' or 'loco_india' (leave-one-country-out stress tests)

Example:
    python experiments/run_validation_suite.py --matching output/matching_results.tsv --split fast_dev
    python experiments/run_validation_suite.py --candidate output/candidate_pairs.tsv --split fast_dev
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

# Ensure src is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (
    evaluate_predictions,
    evaluate_blocking,
    load_id_mapping_tsv,
    print_evaluation_report,
)
from src.split_manager import load_split_ids
from src.config import TRAIN_S1, TRAIN_GT, BASE_DIR


def get_country_mapping(s1_file: str, target_ids: set) -> dict:
    """Load country for target S1 entities."""
    import csv
    country_map = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        id_idx = header.index("entity_id") if header and "entity_id" in header else 0
        c_idx = header.index("country") if header and "country" in header else 3
        for row in reader:
            if not row or len(row) <= id_idx:
                continue
            eid = row[id_idx].strip()
            if eid in target_ids:
                country_map[eid] = row[c_idx].strip() if len(row) > c_idx else "Unknown"
    return country_map


def main():
    parser = argparse.ArgumentParser(description="Run validation suite on predictions or candidate pairs.")
    parser.add_argument("--matching", type=str, default=None, help="Path to matching TSV file")
    parser.add_argument("--candidate", type=str, default=None, help="Path to candidate TSV file")
    parser.add_argument(
        "--split",
        type=str,
        default="fast_dev",
        choices=["fast_dev", "frozen_holdout", "full", "loco_us", "loco_india"],
        help="Evaluation split to evaluate against",
    )
    parser.add_argument("--save-report", type=str, default=None, help="Optional JSON file to save report")
    args = parser.parse_args()

    if not args.matching and not args.candidate:
        parser.error("Specify at least one of --matching or --candidate.")

    splits_dir = os.path.join(BASE_DIR, "splits")

    # 1. Determine evaluation entity IDs
    if args.split == "fast_dev":
        split_path = os.path.join(splits_dir, "fast_dev_50k.json")
        target_ids = load_split_ids(split_path)
        print(f"Loaded 'fast_dev' split ({len(target_ids):,} entities).")
    elif args.split == "frozen_holdout":
        split_path = os.path.join(splits_dir, "frozen_holdout.json")
        target_ids = load_split_ids(split_path)
        print(f"Loaded 'frozen_holdout' split ({len(target_ids):,} entities).")
    elif args.split == "loco_us":
        split_path = os.path.join(splits_dir, "loco_splits.json")
        with open(split_path) as f:
            loco = json.load(f)
        target_ids = set(loco["splits"]["train_India_val_US"]["val_ids"])
        print(f"Loaded 'loco_us' split ({len(target_ids):,} US entities held out).")
    elif args.split == "loco_india":
        split_path = os.path.join(splits_dir, "loco_splits.json")
        with open(split_path) as f:
            loco = json.load(f)
        target_ids = set(loco["splits"]["train_US_val_India"]["val_ids"])
        print(f"Loaded 'loco_india' split ({len(target_ids):,} India entities held out).")
    else:  # full
        target_ids = None
        print("Evaluating on FULL training dataset.")

    # 2. Load ground truth for target entities
    t0 = time.time()
    print("Loading ground truth...")
    all_gt = load_id_mapping_tsv(TRAIN_GT, "source1_entity_id", "matched_entity_ids")

    if target_ids is not None:
        eval_gt = {eid: all_gt.get(eid, set()) for eid in target_ids}
    else:
        eval_gt = all_gt

    print(f"Ground truth prepared for {len(eval_gt):,} entities in {time.time() - t0:.2f}s.")

    # Country mapping
    country_map = get_country_mapping(TRAIN_S1, set(eval_gt.keys()))

    report = {"split": args.split, "num_entities": len(eval_gt)}

    # 3. Evaluate matching if provided
    if args.matching:
        print(f"\nEvaluating matching predictions from {args.matching}...")
        t0 = time.time()
        preds = load_id_mapping_tsv(args.matching, "source1_entity_id", "matched_entity_ids")
        eval_preds = {eid: preds.get(eid, set()) for eid in eval_gt}
        print(f"Predictions loaded in {time.time() - t0:.2f}s.")

        match_metrics = evaluate_predictions(eval_preds, eval_gt, country_map, verbose=True)
        report["matching"] = match_metrics

    # 4. Evaluate candidates if provided
    if args.candidate:
        print(f"\nEvaluating candidate blocking from {args.candidate}...")
        t0 = time.time()
        cands = load_id_mapping_tsv(args.candidate, "source1_entity_id", "candidate_entity_ids")
        eval_cands = {eid: cands.get(eid, set()) for eid in eval_gt}
        print(f"Candidates loaded in {time.time() - t0:.2f}s.")

        block_metrics = evaluate_blocking(eval_gt, eval_cands, verbose=True)
        report["blocking"] = block_metrics

    # Save report if requested
    if args.save_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_report)), exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.save_report}")


if __name__ == "__main__":
    main()
