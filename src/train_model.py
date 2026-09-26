"""
Train a LightGBM/XGBoost classifier on entity resolution features.
Uses the features saved by baseline_v3.py.
Optimizes for macro F0.5.
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM classifier."""
    # Calculate class weights
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'seed': 42,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost classifier."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'verbosity': 0,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    
    return model


def main():
    # Load features
    feat_path = os.path.join(BASE_DIR, 'models', 'train_features.pkl')
    if not os.path.exists(feat_path):
        print(f"ERROR: Feature file not found at {feat_path}")
        print("Run baseline_v3.py first to generate features.")
        return
    
    print("Loading features...")
    df = pd.read_pickle(feat_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    
    # Separate features and labels
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Positive: {y.sum():,} ({100*y.mean():.2f}%)")
    print(f"  Negative: {(1-y).sum():,.0f}")
    
    # Simple train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # Train LightGBM
    print("\n=== Training LightGBM ===")
    lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_pred_lgbm = lgbm_model.predict(X_val)
    
    print(f"\n  LightGBM predictions: min={y_pred_lgbm.min():.4f}, max={y_pred_lgbm.max():.4f}, mean={y_pred_lgbm.mean():.4f}")
    
    # Feature importance
    importance = lgbm_model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\n  Feature importance (gain):")
    for name, imp in feat_imp[:15]:
        print(f"    {name:25s}: {imp:.1f}")
    
    # Threshold analysis for pairwise predictions
    from sklearn.metrics import precision_score, recall_score, f1_score
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_bin = (y_pred_lgbm > t).astype(int)
        p = precision_score(y_val, y_bin, zero_division=0)
        r = recall_score(y_val, y_bin, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        print(f"  threshold={t:.1f}: P={p:.4f}, R={r:.4f}, F0.5={f05:.4f}")
    
    # Train XGBoost
    print("\n=== Training XGBoost ===")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    dval = xgb.DMatrix(X_val)
    y_pred_xgb = xgb_model.predict(dval)
    
    print(f"\n  XGBoost predictions: min={y_pred_xgb.min():.4f}, max={y_pred_xgb.max():.4f}, mean={y_pred_xgb.mean():.4f}")
    
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_bin = (y_pred_xgb > t).astype(int)
        p = precision_score(y_val, y_bin, zero_division=0)
        r = recall_score(y_val, y_bin, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        print(f"  threshold={t:.1f}: P={p:.4f}, R={r:.4f}, F0.5={f05:.4f}")
    
    # Save models
    model_dir = os.path.join(BASE_DIR, 'models')
    lgbm_model.save_model(os.path.join(model_dir, 'lgbm_model.txt'))
    xgb_model.save_model(os.path.join(model_dir, 'xgb_model.json'))
    
    # Save feature column names
    with open(os.path.join(model_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\nModels saved to {model_dir}/")


if __name__ == '__main__':
    main()
