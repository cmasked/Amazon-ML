"""
Enhanced feature engineering using rapidfuzz for edit-distance based features.
These features capture typos, abbreviations, legal suffix normalization,
and character-level similarity that token-based features miss.
"""
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import re

# --- Legal & Business Suffix Lists (US, India, France) ---
# Multi-word suffixes (checked first to ensure precedence over single-word suffixes)
LEGAL_SUFFIXES_MULTIWORD = [
    "PRIVATE LIMITED", "PVT LTD", "CORPORATION", "ENTERPRISES", "COMPANY", "TRADERS"
]

LEGAL_SUFFIXES_STANDARD = [
    "LIMITED", "SARL", "SASU", "EURL", "CORP", "SNC", "SCI", "GIE", "ETS",
    "LLC", "INC", "LTD", "LLP", "SAS"
]

# Short 2-letter ambiguous suffixes requiring conservative length guards (>=3 char prefix)
LEGAL_SUFFIXES_SHORT = ["SA", "CO", "LP"]

# Combine all suffixes sorted by length descending for regex construction
ALL_LEGAL_SUFFIXES = sorted(
    set(LEGAL_SUFFIXES_MULTIWORD + LEGAL_SUFFIXES_STANDARD + LEGAL_SUFFIXES_SHORT),
    key=len,
    reverse=True
)

# Regex pattern matching legal suffixes anchored to string end (no literal '*' used)
LEGAL_SUFFIX_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in ALL_LEGAL_SUFFIXES) + r')$',
    re.IGNORECASE
)


def normalize_legal_suffix(name):
    """
    Normalize business name by iteratively removing legal/company suffixes (US, India, France).
    Returns tuple: (core_name: str, was_suffix_removed: bool)
    """
    if not name:
        return "", False

    # Standardize punctuation & case
    # Remove abbreviation dots (e.g., P.V.T. L.T.D. -> PVT LTD, INC. -> INC)
    cleaned = re.sub(r'\b([A-Za-z])\.', r'\1', name)
    cleaned = re.sub(r'[^\w\s&]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().upper()

    current_name = cleaned
    was_suffix_removed = False

    # Iterative removal loop (max 3 passes for multi-tier suffixes like "Sharma Traders Pvt Ltd")
    for _ in range(3):
        match = LEGAL_SUFFIX_REGEX.search(current_name)
        if not match:
            break

        suffix = match.group(0).upper()
        start_idx = match.start()

        # Candidate core name after stripping suffix
        candidate_core = current_name[:start_idx].strip()
        # Clean potential trailing '&' or 'AND' left behind (e.g. "SMITH & CO" -> "SMITH &" -> "SMITH")
        candidate_core = re.sub(r'\s+(?:&|AND)$', '', candidate_core, flags=re.IGNORECASE).strip()

        # Conservative Guard for short 2-letter suffixes ('SA', 'CO', 'LP')
        if suffix in LEGAL_SUFFIXES_SHORT:
            if len(candidate_core) < 3:
                # Do not strip short suffix if remaining core is too short
                break

        # Guard: ensure we don't reduce core name to empty string
        if len(candidate_core) > 0:
            current_name = candidate_core
            was_suffix_removed = True
        else:
            break

    return current_name, was_suffix_removed


def compute_core_name_features(name1, name2):
    """
    Compute RapidFuzz similarity metrics on legal-suffix stripped core business names.
    """
    core1, removed1 = normalize_legal_suffix(name1)
    core2, removed2 = normalize_legal_suffix(name2)

    feats = {}
    if core1 and core2:
        feats['name_core_fuzz_ratio'] = fuzz.ratio(core1, core2) / 100.0
        feats['name_core_fuzz_partial'] = fuzz.partial_ratio(core1, core2) / 100.0
        feats['name_core_fuzz_token_sort'] = fuzz.token_sort_ratio(core1, core2) / 100.0
        feats['name_core_fuzz_token_set'] = fuzz.token_set_ratio(core1, core2) / 100.0

        max_len = max(len(core1), len(core2))
        feats['name_core_edit_dist_norm'] = 1.0 - (Levenshtein.distance(core1, core2) / max_len)
    else:
        feats['name_core_fuzz_ratio'] = 0.0
        feats['name_core_fuzz_partial'] = 0.0
        feats['name_core_fuzz_token_sort'] = 0.0
        feats['name_core_fuzz_token_set'] = 0.0
        feats['name_core_edit_dist_norm'] = 0.0

    # Explicit boolean flag indicating if a recognized legal suffix was actually removed
    feats['name_has_suffix_removed'] = float(removed1 or removed2)

    return feats


def compute_rapidfuzz_features(name1, name2, addr1, addr2):
    """
    Compute rapidfuzz-based string similarity features.
    All scores normalized to 0-100 range by rapidfuzz, we convert to 0-1.
    Includes core-name features after legal suffix stripping.
    """
    feats = {}

    # Name similarity (edit distance based)
    feats['name_fuzz_ratio'] = fuzz.ratio(name1, name2) / 100.0
    feats['name_fuzz_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    feats['name_fuzz_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    feats['name_fuzz_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    feats['name_fuzz_wratio'] = fuzz.WRatio(name1, name2) / 100.0

    # Address similarity
    feats['addr_fuzz_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    feats['addr_fuzz_partial'] = fuzz.partial_ratio(addr1, addr2) / 100.0
    feats['addr_fuzz_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    feats['addr_fuzz_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0

    # Normalized edit distance for names
    if name1 and name2:
        max_len = max(len(name1), len(name2))
        edit_dist = Levenshtein.distance(name1, name2)
        feats['name_edit_dist_norm'] = 1.0 - (edit_dist / max_len)
    else:
        feats['name_edit_dist_norm'] = 0.0

    # Normalized edit distance for addresses
    if addr1 and addr2:
        max_len = max(len(addr1), len(addr2))
        edit_dist = Levenshtein.distance(addr1, addr2)
        feats['addr_edit_dist_norm'] = 1.0 - (edit_dist / max_len)
    else:
        feats['addr_edit_dist_norm'] = 0.0

    # Add core name features (legal suffix stripped)
    core_feats = compute_core_name_features(name1, name2)
    feats.update(core_feats)

    return feats


def extract_postal_code(address, country=''):
    """Extract postal/PIN code from address."""
    if not address:
        return ''

    # US ZIP codes: 5 digits, optionally followed by -4 digits
    us_zip = re.search(r'\b(\d{5}(?:-\d{4})?)\b', address)
    if us_zip:
        return us_zip.group(1)

    # India PIN codes: 6 digits
    india_pin = re.search(r'\b(\d{6})\b', address)
    if india_pin:
        return india_pin.group(1)

    # France postal codes: 5 digits
    france_postal = re.search(r'\b(\d{5})\b', address)
    if france_postal:
        return france_postal.group(1)

    return ''


def extract_house_number(address):
    """Extract house/building number from address."""
    if not address:
        return ''
    # Common patterns: starts with number, or "No. X", "No X", "#X"
    m = re.match(r'^(\d+[\-/]?\d*)', address)
    if m:
        return m.group(1)
    m = re.search(r'(?:no\.?\s*|#)(\d+)', address, re.IGNORECASE)
    if m:
        return m.group(1)
    return ''


def compute_address_component_features(addr1, addr2):
    """
    Extract and compare specific address components.
    """
    feats = {}

    # Postal code comparison
    pc1 = extract_postal_code(addr1)
    pc2 = extract_postal_code(addr2)
    feats['postal_code_match'] = float(pc1 == pc2 and pc1 != '')
    feats['postal_code_present_both'] = float(pc1 != '' and pc2 != '')
    feats['postal_code_contradiction'] = float(pc1 != '' and pc2 != '' and pc1 != pc2)

    # House number comparison
    hn1 = extract_house_number(addr1)
    hn2 = extract_house_number(addr2)
    feats['house_num_match'] = float(hn1 == hn2 and hn1 != '')
    feats['house_num_contradiction'] = float(hn1 != '' and hn2 != '' and hn1 != hn2)

    return feats


def compute_all_enhanced_features(s1_rec, s2s3_rec):
    """
    Compute the full enhanced feature set combining basic and rapidfuzz features.
    s1_rec and s2s3_rec are dicts with 'name_norm', 'addr_norm', etc.
    """
    from baseline_v3 import compute_features as base_features

    # Get base features
    feats = base_features(s1_rec, s2s3_rec)

    # Add rapidfuzz features
    name1 = s1_rec['name_norm']
    name2 = s2s3_rec.get('name_norm', '')
    addr1 = s1_rec['addr_norm']
    addr2 = s2s3_rec.get('addr_norm', '')

    rf_feats = compute_rapidfuzz_features(name1, name2, addr1, addr2)
    feats.update(rf_feats)

    # Add address component features
    addr_comp_feats = compute_address_component_features(addr1, addr2)
    feats.update(addr_comp_feats)

    return feats
