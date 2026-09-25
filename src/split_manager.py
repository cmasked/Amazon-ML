"""
Leakage-Free Validation Split Generator and Manager.
Person 4 Deliverable for Amazon ML Challenge 2026.

Key Principles:
1. Entity-Level Partitioning: S1 entities are partitioned strictly as discrete atoms.
   Zero overlap between train and validation S1 entities.
2. Stratification: Stratifies across (Country x Match Cardinality) bins:
   - Cardinality bins: singleton (0), 1-match (1), 2-3 matches, 4+ matches
   - Countries: US, India
3. Protocols provided:
   - Frozen Holdout (15% / ~330k entities, sealed for final pre-submission audit)
   - Fast Dev Set (50k entities, stratified for sub-minute iteration by Person 2 & 3)
   - 5-Fold Cross Validation (for robust multi-seed model evaluation)
   - Leave-One-Country-Out (LOCO: Train US / Val India, Train India / Val US) to simulate France!
"""

from __future__ import annotations
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def categorize_cardinality(n_matches: int) -> str:
    """Categorize match count into stratification strata."""
    if n_matches == 0:
        return "singleton"
    elif n_matches == 1:
        return "single"
    elif n_matches <= 3:
        return "multi_small"
    else:
        return "multi_large"


def load_s1_metadata_and_gt(
    s1_path: str,
    gt_path: str,
) -> List[dict]:
    """
    Load S1 metadata and ground truth match counts with minimal memory overhead.
    Returns list of dicts: [{'entity_id': ..., 'country': ..., 'n_matches': ..., 'stratum': ...}]
    """
    print(f"Loading ground truth match counts from {gt_path}...")
    match_counts: Dict[str, int] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                matches = [m.strip() for m in row[1].split(",") if m.strip()]
                match_counts[s1_id] = len(matches)
            else:
                match_counts[s1_id] = 0

    print(f"Loading S1 entities from {s1_path}...")
    entities = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        id_idx = header.index("entity_id") if header and "entity_id" in header else 0
        country_idx = header.index("country") if header and "country" in header else 3

        for row in reader:
            if not row or len(row) <= id_idx:
                continue
            s1_id = row[id_idx].strip()
            country = row[country_idx].strip() if len(row) > country_idx else "Unknown"
            n_matches = match_counts.get(s1_id, 0)
            card_class = categorize_cardinality(n_matches)
            stratum = f"{country}_{card_class}"

            entities.append({
                "entity_id": s1_id,
                "country": country,
                "n_matches": n_matches,
                "stratum": stratum,
            })

    print(f"Total S1 entities indexed: {len(entities):,}")
    return entities


def generate_stratified_splits(
    entities: List[dict],
    holdout_ratio: float = 0.15,
    fast_dev_size: int = 50000,
    n_cv_folds: int = 5,
    random_seed: int = 42,
    output_dir: str = "splits",
) -> Dict[str, str]:
    """
    Generate all required experimental splits:
    1. frozen_holdout.json: sealed holdout set
    2. fast_dev.json: 50k stratified sample for lightning-fast iterations
    3. cv_5folds.json: 5-fold cross validation on dev set (total entities minus holdout)
    4. loco_us_val.json / loco_india_val.json: Leave-One-Country-Out splits
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(random_seed)

    # Group entities by stratum
    strata: Dict[str, List[str]] = defaultdict(list)
    country_groups: Dict[str, List[str]] = defaultdict(list)

    for item in entities:
        eid = item["entity_id"]
        strata[item["stratum"]].append(eid)
        country_groups[item["country"]].append(eid)

    # 1. Stratified Partition: Frozen Holdout vs Dev Pool
    holdout_ids: List[str] = []
    dev_pool_ids: List[str] = []
    dev_pool_by_stratum: Dict[str, List[str]] = defaultdict(list)

    for stratum_name, ids in strata.items():
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n_holdout = int(len(shuffled) * holdout_ratio)
        holdout_part = shuffled[:n_holdout]
        dev_part = shuffled[n_holdout:]

        holdout_ids.extend(holdout_part)
        dev_pool_ids.extend(dev_part)
        dev_pool_by_stratum[stratum_name] = dev_part

    rng.shuffle(holdout_ids)
    rng.shuffle(dev_pool_ids)

    # 2. Fast Dev Set (Stratified sample from dev pool)
    fast_dev_ratio = fast_dev_size / len(dev_pool_ids) if len(dev_pool_ids) > 0 else 0.05
    fast_dev_ids: List[str] = []
    for stratum_name, ids in dev_pool_by_stratum.items():
        n_fast = max(1, int(len(ids) * fast_dev_ratio))
        fast_dev_ids.extend(ids[:n_fast])
    rng.shuffle(fast_dev_ids)
    # Trim to exact requested size
    fast_dev_ids = fast_dev_ids[:fast_dev_size]

    # 3. 5-Fold Cross Validation within Dev Pool
    cv_folds: List[Dict[str, List[str]]] = []
    fold_buckets: List[List[str]] = [[] for _ in range(n_cv_folds)]

    for stratum_name, ids in dev_pool_by_stratum.items():
        for i, eid in enumerate(ids):
            fold_buckets[i % n_cv_folds].append(eid)

    for f_idx in range(n_cv_folds):
        val_f = fold_buckets[f_idx]
        train_f = []
        for other_idx in range(n_cv_folds):
            if other_idx != f_idx:
                train_f.extend(fold_buckets[other_idx])
        cv_folds.append({
            "fold": f_idx,
            "train_size": len(train_f),
            "val_size": len(val_f),
            "val_ids": val_f,
        })

    # 4. Leave-One-Country-Out (LOCO) Splits
    # Split: Train US / Val India, and Train India / Val US
    loco_splits = {
        "train_US_val_India": {
            "train_country": "US",
            "val_country": "India",
            "train_ids": country_groups.get("US", []),
            "val_ids": country_groups.get("India", []),
        },
        "train_India_val_US": {
            "train_country": "India",
            "val_country": "US",
            "train_ids": country_groups.get("India", []),
            "val_ids": country_groups.get("US", []),
        },
    }

    # Save all splits
    paths = {}

    holdout_path = os.path.join(output_dir, "frozen_holdout.json")
    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Sealed frozen holdout set - DO NOT TUNE ON THIS ITERATIVELY",
            "random_seed": random_seed,
            "holdout_ratio": holdout_ratio,
            "total_entities": len(holdout_ids),
            "entity_ids": holdout_ids,
        }, f, indent=2)
    paths["frozen_holdout"] = holdout_path

    fast_dev_path = os.path.join(output_dir, "fast_dev_50k.json")
    with open(fast_dev_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": f"Stratified fast dev set of {len(fast_dev_ids):,} entities for rapid iterations",
            "random_seed": random_seed,
            "total_entities": len(fast_dev_ids),
            "entity_ids": fast_dev_ids,
        }, f, indent=2)
    paths["fast_dev"] = fast_dev_path

    cv_path = os.path.join(output_dir, f"cv_{n_cv_folds}folds.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": f"{n_cv_folds}-fold stratified cross-validation on dev pool",
            "random_seed": random_seed,
            "n_folds": n_cv_folds,
            "dev_pool_size": len(dev_pool_ids),
            "folds": [
                {"fold": cf["fold"], "train_size": cf["train_size"], "val_size": cf["val_size"], "val_ids": cf["val_ids"]}
                for cf in cv_folds
            ],
        }, f)
    paths["cv_folds"] = cv_path

    loco_path = os.path.join(output_dir, "loco_splits.json")
    with open(loco_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Leave-One-Country-Out splits for stress testing unseen country generalization",
            "splits": {
                k: {
                    "train_country": v["train_country"],
                    "val_country": v["val_country"],
                    "train_count": len(v["train_ids"]),
                    "val_count": len(v["val_ids"]),
                    "val_ids": v["val_ids"],
                }
                for k, v in loco_splits.items()
            }
        }, f)
    paths["loco"] = loco_path

    print("\n=== GENERATED LEAKAGE-FREE SPLITS ===")
    print(f"Total Records  : {len(entities):,}")
    print(f"Frozen Holdout : {len(holdout_ids):,} entities ({len(holdout_ids)/len(entities)*100:.1f}%) -> {holdout_path}")
    print(f"Dev Pool       : {len(dev_pool_ids):,} entities ({len(dev_pool_ids)/len(entities)*100:.1f}%)")
    print(f"Fast Dev Sample: {len(fast_dev_ids):,} entities -> {fast_dev_path}")
    print(f"CV Folds       : {n_cv_folds} folds (~{len(dev_pool_ids)//n_cv_folds:,} val entities/fold) -> {cv_path}")
    print(f"LOCO Splits    : US vs India generalization splits -> {loco_path}")

    return paths


def load_split_ids(split_file_path: str) -> Set[str]:
    """Load entity IDs from any generated split JSON file."""
    with open(split_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "entity_ids" in data:
            return set(data["entity_ids"])
        elif "folds" in data:
            # Returns fold 0 val_ids as default if CV file passed
            return set(data["folds"][0]["val_ids"])
        elif "splits" in data:
            first_split = next(iter(data["splits"].values()))
            return set(first_split["val_ids"])
        raise ValueError(f"Unknown split format in {split_file_path}")
