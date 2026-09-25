"""
Comprehensive text normalization and key extraction utilities for Role 2 Candidate Generation.
Preserves raw attributes while creating standardized, aggressive, and phonetic representations.
"""

import re
import unicodedata
from typing import Set, List, Tuple, Dict, Any

# Multi-word legal suffixes sorted by length descending
MULTIWORD_LEGAL_SUFFIXES = [
    "PRIVATE LIMITED", "PVT LTD", "PVT. LTD.", "PVT. LIMITED", "P. LTD",
    "LIMITED LIABILITY COMPANY", "LIMITED LIABILITY PARTNERSHIP",
    "CORPORATION", "INCORPORATED", "ENTERPRISES", "ENTERPRISE",
    "COMPANY", "TRADERS", "HOLDINGS", "GROUP", "SOLUTIONS", "TECHNOLOGIES",
    "AND SONS", "& SONS", "AND DAUGHTERS", "& DAUGHTERS",
    "AND BROTHERS", "& BROTHERS", "AND COMPANY", "& COMPANY", "AND CO", "& CO",
    "ASSOCIATES", "PARTNERS", "BROTHERS", "SERVICES", "LOGISTICS"
]

# Standard single-word legal suffixes
STANDARD_LEGAL_SUFFIXES = {
    'incorporated': 'inc',
    'corporation': 'corp',
    'company': 'co',
    'limited': 'ltd',
    'private': 'pvt',
    'public': 'pub',
    'llc': 'llc',
    'llp': 'llp',
    'plc': 'plc',
    'ltd': 'ltd',
    'inc': 'inc',
    'corp': 'corp',
    'pvt': 'pvt',
    'co': 'co',
    'bros': 'bros',
    'sons': 'sons',
    # French legal suffixes
    'sarl': 'sarl',
    'sasu': 'sasu',
    'eurl': 'eurl',
    'snc': 'snc',
    'sci': 'sci',
    'gie': 'gie',
    'ets': 'ets',
    'sas': 'sas',
    'sa': 'sa',
    # German/European suffixes
    'gmbh': 'gmbh',
    'ag': 'ag',
}

# Address abbreviations dictionary
ADDRESS_ABBREVIATIONS = {
    'street': 'st',
    'road': 'rd',
    'avenue': 'ave',
    'boulevard': 'blvd',
    'drive': 'dr',
    'lane': 'ln',
    'court': 'ct',
    'circle': 'cir',
    'place': 'pl',
    'highway': 'hwy',
    'parkway': 'pkwy',
    'terrace': 'ter',
    'north': 'n',
    'south': 's',
    'east': 'e',
    'west': 'w',
    'northeast': 'ne',
    'northwest': 'nw',
    'southeast': 'se',
    'southwest': 'sw',
    'apartment': 'apt',
    'suite': 'ste',
    'building': 'bldg',
    'floor': 'fl',
    'room': 'rm',
    'department': 'dept',
    'nagar': 'ngr',
    'marg': 'mrg',
    'district': 'dist',
    'opposite': 'opp',
    'near': 'nr',
}

COMMON_BUSINESS_STOPWORDS = {
    'the', 'of', 'and', '&', 'a', 'an', 'in', 'at', 'on', 'for', 'by'
}


def normalize_unicode(text: str) -> str:
    """Decompose Unicode and strip non-ASCII accents safely."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text))
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_name_standard(name: str) -> str:
    """
    Standard normalization for business names.
    Preserves core entity tokens while standardizing case, ampersand, punctuation, and suffixes.
    Separates concatenated letters and numbers (e.g. 'Construction4' -> 'Construction 4').
    """
    if not name or not isinstance(name, str):
        return ""
        
    text = normalize_unicode(name).lower().strip()
    text = text.replace('&', ' and ')
    
    # Strip dots inside abbreviations (e.g. p.v.t. -> pvt, u.s.a -> usa)
    text = re.sub(r'\b([a-z])\.', r'\1', text)
    
    # Separate letter-digit concatenations (e.g. Redwood-Construction4 -> Redwood-Construction 4)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # Replace hyphens, underscores, slashes with space
    text = text.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    
    # Replace remaining punctuation with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Standardize legal suffix tokens
    tokens = text.split()
    norm_tokens = []
    for token in tokens:
        clean = token.rstrip('.')
        if clean in STANDARD_LEGAL_SUFFIXES:
            norm_tokens.append(STANDARD_LEGAL_SUFFIXES[clean])
        else:
            norm_tokens.append(clean)
            
    text = " ".join(norm_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_legal_suffixes(normalized_name: str) -> Tuple[str, bool]:
    """
    Iteratively strips recognized multi-word and single-word legal suffixes from the end of a name.
    Returns (core_name, was_stripped).
    """
    if not normalized_name:
        return "", False
        
    name = normalized_name.strip()
    was_stripped = False
    
    # Pass 1: Multi-word suffixes
    for suffix in MULTIWORD_LEGAL_SUFFIXES:
        pattern = r'\b' + re.escape(suffix.lower()) + r'$'
        m = re.search(pattern, name)
        if m and m.start() >= 2:
            candidate = name[:m.start()].strip()
            # Clean trailing 'and', '&', or '-'
            candidate = re.sub(r'\s+(?:and|&|-)$', '', candidate).strip()
            if len(candidate) >= 2:
                name = candidate
                was_stripped = True
                break
                
    # Pass 2: Single-word suffixes
    tokens = name.split()
    while tokens and (tokens[-1] in STANDARD_LEGAL_SUFFIXES or tokens[-1] in {'ltd', 'pvt', 'inc', 'corp', 'co', 'llc', 'llp', 'sa', 'sas', 'sarl', 'group', 'services'}):
        if len(tokens) == 1:
            break
        tokens.pop()
        was_stripped = True
        
    core = " ".join(tokens).strip()
    return core, was_stripped


def normalize_name_aggressive(name: str) -> str:
    """
    Aggressive normalization:
    - Strips all legal suffixes
    - Removes common business stopwords
    - Produces continuous alphanumeric core
    """
    norm = normalize_name_standard(name)
    core, _ = strip_legal_suffixes(norm)
    tokens = [t for t in core.split() if t not in COMMON_BUSINESS_STOPWORDS]
    return " ".join(tokens)


def normalize_name_compact(name: str) -> str:
    """
    Alphanumeric continuous representation of aggressive name.
    Catches hyphenation / spacing discrepancies: 'Wal-Mart' vs 'Wal Mart' -> 'walmart'
    """
    agg = normalize_name_aggressive(name)
    return re.sub(r'[^a-z0-9]', '', agg)


def normalize_address_standard(address: str) -> str:
    """
    Standard address normalization:
    - Lowercasing & Unicode decomposition
    - Standardizing street abbreviations (st, rd, ave, etc.)
    - Retaining slashes and hyphens for building/unit numbers
    """
    if not address or not isinstance(address, str):
        return ""
        
    text = normalize_unicode(address).lower().strip()
    
    # Strip dots in abbreviations
    text = re.sub(r'\b([a-z])\.', r'\1', text)
    
    # Separate letter-digit concatenations (e.g. MG Road560001 -> MG Road 560001)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # Retain hyphens and slashes for addresses (e.g. 12-B, 4/12)
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    
    tokens = text.split()
    norm_tokens = []
    for token in tokens:
        clean = token.rstrip('.')
        if clean in ADDRESS_ABBREVIATIONS:
            norm_tokens.append(ADDRESS_ABBREVIATIONS[clean])
        else:
            norm_tokens.append(clean)
            
    text = " ".join(norm_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric sequences (postal codes, house numbers, plot numbers)."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))


def extract_char_ngrams(text: str, n_range: Tuple[int, int] = (3, 3)) -> Set[str]:
    """
    Extract character n-grams from text within specified range.
    Adds boundary markers '#' for start and end of strings.
    """
    if not text:
        return set()
    clean = re.sub(r'\s+', ' ', text.strip())
    padded = f"#{clean}#"
    ngrams = set()
    min_n, max_n = n_range
    for n in range(min_n, max_n + 1):
        if len(padded) >= n:
            for i in range(len(padded) - n + 1):
                ngrams.add(padded[i:i+n])
    return ngrams


def soundex(token: str) -> str:
    """
    Standard Soundex phonetic encoder for English/Latin tokens.
    Guaranteed deterministic, pure python, fast execution.
    """
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    first_letter = token[0]
    
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    encoded = [first_letter]
    prev = mapping.get(first_letter, '0')
    
    for char in token[1:]:
        code = mapping.get(char, '0')
        if code != '0':
            if code != prev:
                encoded.append(code)
            prev = code
        else:
            prev = '0'
            
    digits = "".join(encoded[1:])
    code = (first_letter + digits + "000")[:4]
    return code


def build_record_representations(eid: str, bname: str, baddr: str, country: str) -> Dict[str, Any]:
    """
    Build rich representations for a single record without altering raw values.
    """
    norm_name = normalize_name_standard(bname)
    core_name, suffix_removed = strip_legal_suffixes(norm_name)
    compact_name = normalize_name_compact(bname)
    norm_addr = normalize_address_standard(baddr)
    
    name_tokens = [t for t in norm_name.split() if len(t) >= 2]
    core_tokens = [t for t in core_name.split() if len(t) >= 2]
    addr_tokens = [t for t in norm_addr.split() if len(t) >= 3]
    addr_nums = extract_numbers(norm_addr)
    
    name_ngrams_3 = extract_char_ngrams(core_name or norm_name, (3, 3))
    
    return {
        'entity_id': eid,
        'raw_name': bname or "",
        'raw_address': baddr or "",
        'country': (country or "").strip().upper(),
        'name_norm': norm_name,
        'name_core': core_name,
        'name_compact': compact_name,
        'addr_norm': norm_addr,
        'name_tokens': frozenset(name_tokens),
        'core_tokens': frozenset(core_tokens),
        'addr_tokens': frozenset(addr_tokens),
        'addr_nums': frozenset(addr_nums),
        'name_ngrams_3': frozenset(name_ngrams_3),
        'has_suffix_removed': suffix_removed
    }
