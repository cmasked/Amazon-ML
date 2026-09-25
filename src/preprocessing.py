"""
Text preprocessing and normalization for business names and addresses.
"""
import re
import unicodedata

# Common legal suffix mappings
LEGAL_SUFFIXES = {
    'incorporated': 'inc',
    'corporation': 'corp',
    'company': 'co',
    'limited': 'ltd',
    'private': 'pvt',
    'public': 'pub',
    'llc': 'llc',
    'llp': 'llp',
    'plc': 'plc',
    'l.l.c': 'llc',
    'l.l.c.': 'llc',
    'l.l.p': 'llp',
    'l.l.p.': 'llp',
    'p.l.c': 'plc',
    'inc.': 'inc',
    'corp.': 'corp',
    'co.': 'co',
    'ltd.': 'ltd',
    'pvt.': 'pvt',
    'pvt': 'pvt',
    'ltd': 'ltd',
    'inc': 'inc',
    'corp': 'corp',
    'sarl': 'sarl',
    's.a.r.l': 'sarl',
    's.a.r.l.': 'sarl',
    'sas': 'sas',
    's.a.s': 'sas',
    's.a.s.': 'sas',
    'sa': 'sa',
    's.a': 'sa',
    's.a.': 'sa',
    'gmbh': 'gmbh',
    'ag': 'ag',
    'eurl': 'eurl',
    'e.u.r.l': 'eurl',
}

# Address abbreviations
ADDRESS_ABBREVS = {
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
}


def normalize_unicode(text):
    """Normalize unicode characters to ASCII where possible."""
    if not text:
        return ''
    # Normalize to NFKD form
    text = unicodedata.normalize('NFKD', text)
    return text


def normalize_name(name):
    """Normalize a business name for comparison."""
    if not name or not isinstance(name, str):
        return ''
    
    name = str(name).strip()
    name = normalize_unicode(name)
    name = name.lower()
    
    # Replace & with 'and'
    name = name.replace('&', ' and ')
    
    # Remove punctuation except hyphens
    name = re.sub(r'[^\w\s\-]', ' ', name)
    
    # Normalize legal suffixes
    tokens = name.split()
    normalized_tokens = []
    for token in tokens:
        clean = token.strip().rstrip('.')
        if clean in LEGAL_SUFFIXES:
            normalized_tokens.append(LEGAL_SUFFIXES[clean])
        else:
            normalized_tokens.append(token)
    
    # Rejoin and collapse whitespace
    name = ' '.join(normalized_tokens)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def normalize_address(address):
    """Normalize a business address for comparison."""
    if not address or not isinstance(address, str):
        return ''
    
    address = str(address).strip()
    address = normalize_unicode(address)
    address = address.lower()
    
    # Remove punctuation except hyphens and slashes
    address = re.sub(r'[^\w\s\-/]', ' ', address)
    
    # Normalize address abbreviations
    tokens = address.split()
    normalized_tokens = []
    for token in tokens:
        clean = token.strip().rstrip('.')
        if clean in ADDRESS_ABBREVS:
            normalized_tokens.append(ADDRESS_ABBREVS[clean])
        else:
            normalized_tokens.append(token)
    
    address = ' '.join(normalized_tokens)
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address


def extract_tokens(text):
    """Extract word tokens from normalized text."""
    if not text:
        return set()
    return set(text.split())


def extract_numbers(text):
    """Extract all numeric tokens from text."""
    if not text:
        return set()
    return set(re.findall(r'\d+', text))


def remove_legal_suffix(name):
    """Remove legal suffixes from a business name."""
    if not name:
        return ''
    tokens = name.split()
    # Remove trailing legal suffixes
    while tokens and tokens[-1] in set(LEGAL_SUFFIXES.values()):
        tokens.pop()
    return ' '.join(tokens)
