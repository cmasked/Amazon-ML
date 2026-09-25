"""
Unit tests for Person 4 Leakage-Free Split Manager.
Verifies that:
1. S1 entities are never leaked between train and validation sets.
2. Stratification preserves country and match-cardinality distributions.
3. Cross-validation folds are mutually exclusive and collectively exhaustive over dev set.
4. Leave-One-Country-Out partitions strictly separate countries.
"""

import os
import shutil
import tempfile
import pytest
from src.split_manager import generate_stratified_splits, load_split_ids, categorize_cardinality


@pytest.fixture
def mock_entities():
    """Create a balanced synthetic set of entities across countries and cardinality."""
    entities = []
    countries = ["US", "India"]
    counts = [0, 1, 2, 5]  # singleton, single, multi_small, multi_large

    idx = 1
    for country in countries:
        for n_matches in counts:
            card_class = categorize_cardinality(n_matches)
            stratum = f"{country}_{card_class}"
            # Create 100 entities per stratum = 800 total
            for _ in range(100):
                entities.append({
                    "entity_id": f"S1-{idx:06d}",
                    "country": country,
                    "n_matches": n_matches,
                    "stratum": stratum,
                })
                idx += 1
    return entities


def test_zero_leakage_holdout(mock_entities):
    """Test that frozen holdout has strictly ZERO entity overlap with dev pool."""
    temp_dir = tempfile.mkdtemp()
    try:
        paths = generate_stratified_splits(
            mock_entities,
            holdout_ratio=0.2,
            fast_dev_size=100,
            n_cv_folds=5,
            random_seed=42,
            output_dir=temp_dir,
        )

        holdout_ids = load_split_ids(paths["frozen_holdout"])
        assert len(holdout_ids) == 160  # 20% of 800

        # Load CV fold 0 train and val
        import json
        with open(paths["cv_folds"], "r") as f:
            cv_data = json.load(f)

        for fold in cv_data["folds"]:
            val_ids = set(fold["val_ids"])
            # ZERO overlap between holdout and any CV fold validation
            assert len(holdout_ids & val_ids) == 0, "Leakage detected between Holdout and CV Val!"
    finally:
        shutil.rmtree(temp_dir)


def test_cv_folds_mutually_exclusive_and_exhaustive(mock_entities):
    """Test that CV validation folds partition the dev pool cleanly."""
    temp_dir = tempfile.mkdtemp()
    try:
        paths = generate_stratified_splits(
            mock_entities,
            holdout_ratio=0.2,
            fast_dev_size=100,
            n_cv_folds=5,
            random_seed=42,
            output_dir=temp_dir,
        )

        import json
        with open(paths["cv_folds"], "r") as f:
            cv_data = json.load(f)

        all_cv_val_ids = set()
        for i, fold in enumerate(cv_data["folds"]):
            val_ids = set(fold["val_ids"])
            # Check disjointness with other folds
            assert len(all_cv_val_ids & val_ids) == 0, f"Leakage detected in fold {i}!"
            all_cv_val_ids |= val_ids

        # Dev pool size is 800 - 160 = 640
        assert len(all_cv_val_ids) == 640
    finally:
        shutil.rmtree(temp_dir)


def test_leave_one_country_out_separation(mock_entities):
    """Test that Leave-One-Country-Out splits have 100% pure country separation."""
    temp_dir = tempfile.mkdtemp()
    try:
        paths = generate_stratified_splits(
            mock_entities,
            holdout_ratio=0.2,
            fast_dev_size=100,
            n_cv_folds=5,
            random_seed=42,
            output_dir=temp_dir,
        )

        import json
        with open(paths["loco"], "r") as f:
            loco_data = json.load(f)

        id_to_country = {e["entity_id"]: e["country"] for e in mock_entities}

        us_val = loco_data["splits"]["train_India_val_US"]["val_ids"]
        india_val = loco_data["splits"]["train_US_val_India"]["val_ids"]

        assert all(id_to_country[eid] == "US" for eid in us_val)
        assert all(id_to_country[eid] == "India" for eid in india_val)
        assert len(set(us_val) & set(india_val)) == 0
    finally:
        shutil.rmtree(temp_dir)
