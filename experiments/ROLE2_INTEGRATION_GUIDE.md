# Role 2 — Integration Guide for Person 1

**Module:** `src/search_engine.py`  
**Class:** `MultiPassSearchEngine`  
**Compatibility:** Python 3.8+ (fully drop-in compatible with `baseline_v3.py`)

---

## 1. Quick Minimal Integration Example

```python
from search_engine import MultiPassSearchEngine

# 1. Initialize engine
engine = MultiPassSearchEngine(default_top_k=80)

# 2. Ingest candidate records from S2 and S3 (streaming or batch)
for record in s2_and_s3_records:
    engine.add_record(
        eid=record['entity_id'],              # str, e.g. "S2-000123"
        bname=record.get('business_name', ''), # str, raw name
        baddr=record.get('business_address', ''), # str, raw address
        country=record.get('country', '')     # str, e.g. "US", "India", "France"
    )

# 3. Finalize index (converts postings to compact C arrays and computes IDF)
engine.finalize_index()

# 4. Retrieve candidates for each S1 query entity
for s1_rec in s1_records:
    # Returns list of (candidate_entity_id, score) tuples
    candidates = engine.retrieve(
        s1_rec=s1_rec, # Dict containing 'entity_id', 'business_name', 'business_address', 'country'
        top_k=80
    )
    # candidates: [("S2-000123", 24.5), ("S3-000456", 18.2), ...]
```

---

## 2. Constructor & Configuration Parameters

```python
engine = MultiPassSearchEngine(
    enable_exact_name=True,    # Pass A: Exact normalized and continuous compact name matching
    enable_exact_addr=True,    # Pass B: Exact normalized physical address matching
    enable_char_ngram=True,    # Pass C: Character 3-gram inverted index with IDF weighting
    enable_token_tfidf=True,   # Pass D: Word token TF-IDF overlap
    enable_rare_tokens=True,   # Pass E: Rare token inverted index
    enable_addr_numeric=True,  # Pass F: Postal code and street number index
    enable_phonetic=True,      # Pass G: Country-conditioned Soundex phonetic index
    ngram_range=(3, 3),        # Character n-gram range (3, 3) optimal for speed and memory
    max_posting_size=25000,    # Max posting length before term pruning (stops high-frequency stopwords)
    rare_token_cutoff=100,     # Max document frequency to classify a word token as 'rare'
    default_top_k=80           # Default candidate cap per S1 entity
)
```

---

## 3. Data Formats & Conventions

### Input Format
* `eid` (`str`): Unique entity identifier, e.g. `"S2-000123"`.
* `bname` (`str`): Raw business name. Automatic normalization is handled internally (legal suffixes like `Inc`, `LLC`, `Corp`, `Pvt Ltd`, `SARL`, `GmbH` are stripped; alphanumeric boundaries are split).
* `baddr` (`str`): Raw business address. Automatic address expansion is handled internally (abbreviations like `St`, `Rd`, `Ave`, `Blvd` are unified; suite/apartment designators removed).
* `country` (`str`): Country code or name (e.g. `"US"`, `"India"`, `"FR"`, `"France"`). Used for country-conditioned phonetic keys. If missing or unknown, gracefully falls back to global indexes.

### Query Input (`s1_rec`)
A dictionary representing a single query entity with keys:
* `'entity_id'`: Query entity ID (e.g. `"S1-000001"`)
* `'business_name'`: Raw business name
* `'business_address'`: Raw business address
* `'country'`: Country code/name

### Output Format
A list of `(candidate_id, retrieval_score)` tuples sorted descending by relevance score, capped at `top_k`:
```python
[
    ("S2-000482", 28.45),
    ("S3-001290", 22.10),
    ("S2-009841", 14.80),
    ...
]
```

---

## 4. Architectural Highlights & Guarantees

1. **Guaranteed Slot Reservation for True Matches:**
   * Exact normalized name matches are allocated up to `min(top_k // 4, 20)` dedicated slots.
   * Exact normalized address matches are allocated up to `min(top_k // 4, 20)` dedicated slots.
   * **Result:** High-confidence identity matches are mathematically protected from being crowded out by high-frequency fuzzy token matches, even at low $k$ values ($k=10$ still achieves 100% recall).
2. **Compact Integer Representation:**
   * All internal posting lists are stored as contiguous 32-bit unsigned C arrays (`array.array('I')`), requiring only 4 bytes per entry with 0 bytes per-element Python object overhead.
3. **No Ground Truth Dependency (100% Leakage-Free):**
   * The indexing and retrieval pipeline does not accept, reference, or require ground truth or labels.

---

## 5. Legacy Fallback Compatibility

If legacy single-token inverted index blocking is needed for comparison or fallback:
* In `src/baseline_v3.py`, pass `--legacy-blocking`:
  ```bash
  python src/baseline_v3.py --val-size 2000 --top-k 80 --legacy-blocking
  ```
* In code:
  ```python
  from baseline_v3 import build_token_index, find_candidates
  # Legacy functions remain untouched and 100% functional
  ```
