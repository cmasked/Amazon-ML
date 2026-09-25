"""
Configuration for the entity resolution pipeline.
All paths, thresholds, and parameters in one place.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data files
TRAIN_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

TEST_S1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_S2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_S3 = os.path.join(TEST_DIR, 'test_source3.tsv')

# Output files
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

# Blocking parameters
TFIDF_NGRAM_RANGE = (2, 4)  # Character n-gram range for TF-IDF
TFIDF_TOP_K = 50  # Number of nearest neighbors per query
TFIDF_MAX_FEATURES = 200000

# Feature parameters  
MATCH_THRESHOLD = 0.5  # Will be tuned on validation

# Validation
VAL_FRACTION = 0.15  # Fraction of training S1 entities to hold out
RANDOM_SEED = 42

# Ensure output dirs exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
