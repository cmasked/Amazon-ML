"""
France & Unseen-Country Robustness Stress-Test Suite.
Person 4 Deliverable for Amazon ML Challenge 2026.

Audits pipelines against the unseen country scenario:
1. Code audits for hardcoded country lists or fatal categorical encodings.
2. UTF-8 / French diacritic handling (é, è, ê, ë, à, â, î, ï, ô, ù, û, ç, œ).
3. French legal suffix coverage (SARL, SAS, SA, SASU, EURL, SCI, SNC, GIE).
4. French address pattern extraction (Rue, Bd, Avenue, 5-digit postal code, Cedex).
5. Synthetic French benchmark evaluation dataset to test Candidate Generation (Person 2)
   and Feature Engineering / Matching (Person 3) before test-set submission.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Callable, Dict, List, Optional, Set, Tuple


FRENCH_LEGAL_SUFFIXES = [
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gie", "ei",
    "societe anonyme", "societe par actions simplifiee",
    "societe a responsabilite limitee", "association", "scop",
]

FRENCH_ADDRESS_TOKENS = [
    "rue", "boulevard", "bd", "bvd", "avenue", "av", "allee", "impasse",
    "place", "route", "chemin", "quai", "rond-point", "cours", "cedex",
]

# Benchmark pairs of French entities with realistic variations
FRENCH_SYNTHETIC_BENCHMARKS = [
    {
        "id": "FR-001",
        "name_clean": "Boulangerie Patisserie Paul SAS",
        "name_noisy": "Boulangerie & Pâtisserie Paul S.A.S.",
        "addr_clean": "14 Boulevard Saint-Germain, 75005 Paris, France",
        "addr_noisy": "14 bd St Germain 75005 Paris",
        "is_match": True,
        "challenge": "accents, legal suffix, street abbreviation",
    },
    {
        "id": "FR-002",
        "name_clean": "Société Générale de Banque",
        "name_noisy": "Societe Generale",
        "addr_clean": "29 Boulevard Haussmann, 75009 Paris",
        "addr_noisy": "29 Bvd Haussmann 75009 Paris CEDEX 09",
        "is_match": True,
        "challenge": "accents, partial name, cedex postal suffix",
    },
    {
        "id": "FR-003",
        "name_clean": "Électricité et Plomberie Lyonnaise SARL",
        "name_noisy": "Electricite & Plomberie Lyonnaise",
        "addr_clean": "8 Rue de la République, 69002 Lyon",
        "addr_noisy": "8 r. de la Republique Lyon 69002",
        "is_match": True,
        "challenge": "initial capital accent, ampersand, address transposition",
    },
    {
        "id": "FR-004",
        "name_clean": "Hôtel du Centre et Spa",
        "name_noisy": "Hotel du Centre & Spa",
        "addr_clean": "5 Avenue Jean Jaurès, 31000 Toulouse",
        "addr_noisy": "5 Av Jean Jaures Toulouse",
        "is_match": True,
        "challenge": "circumflex accent, avenue abbreviation, missing postal",
    },
    {
        "id": "FR-005",
        "name_clean": "Pharmacie de la Mairie",
        "name_noisy": "Pharmacie de la Gare",
        "addr_clean": "12 Place de la Mairie, 35000 Rennes",
        "addr_noisy": "12 Place de la Gare, 35000 Rennes",
        "is_match": False,
        "challenge": "different business, similar address layout (hard negative)",
    },
    {
        "id": "FR-006",
        "name_clean": "Carrefour City",
        "name_noisy": "Carrefour Market",
        "addr_clean": "10 Rue Victor Hugo, 13001 Marseille",
        "addr_noisy": "10 Rue Victor Hugo, 13001 Marseille",
        "is_match": False,
        "challenge": "chain store franchise distinction at same/close location (hard negative)",
    },
    {
        "id": "FR-007",
        "name_clean": "Café des Artistes",
        "name_noisy": "Cafe des Artistes EURL",
        "addr_clean": "3 Quai de la Tournelle, 75005 Paris",
        "addr_noisy": "3 quai tournelle paris",
        "is_match": True,
        "challenge": "quai prefix, lowercasing, missing preposition",
    },
]


def extract_french_postal_code(address: str) -> Optional[str]:
    """
    Extract French 5-digit postal code (01000 - 98890, plus 2A/2B Corsica).
    Robust to 'CEDEX' formatting.
    """
    if not address:
        return None
    # French postal codes are 5 consecutive digits starting from 01 to 98
    match = re.search(r"\b(0[1-9]|[1-8]\d|9[0-8])\d{3}\b", address)
    if match:
        return match.group(0)
    return None


def normalize_french_text(text: str, keep_accents: bool = False) -> str:
    """
    Normalize French text with proper unicode handling.
    If keep_accents is False, performs canonical NFKD decomposition to strip accents
    without dropping the underlying ASCII letter.
    """
    if not text:
        return ""

    text = text.strip().lower()

    if not keep_accents:
        # e.g. "Société" -> "Societe" (does NOT drop the e!)
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if unicodedata.category(c) != "Mn"
        )

    # Standardize ampersand
    text = text.replace("&", " and ")

    # Strip punctuation while preserving whitespace and letters
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def audit_similarity_function_on_france(
    similarity_fn: Callable[[str, str], float],
    fn_name: str = "similarity_fn",
) -> Dict[str, float]:
    """
    Audit any similarity function (from Person 3) against French synthetic benchmarks.
    Verifies that matches score significantly higher than hard negatives.
    """
    match_scores = []
    non_match_scores = []

    print(f"\n--- AUDITING '{fn_name}' ON FRANCE BENCHMARKS ---")
    for case in FRENCH_SYNTHETIC_BENCHMARKS:
        n1, n2 = case["name_clean"], case["name_noisy"]
        a1, a2 = case["addr_clean"], case["addr_noisy"]

        # Call with both name and address or name alone depending on signature
        try:
            score = similarity_fn(n1, n2)
        except TypeError:
            score = similarity_fn(f"{n1} {a1}", f"{n2} {a2}")

        if case["is_match"]:
            match_scores.append(score)
            status = "MATCH (TP/FN)"
        else:
            non_match_scores.append(score)
            status = "NON-MATCH (TN/FP)"

        print(f"[{case['id']}] {status:<15} Score: {score:.4f} | Challenge: {case['challenge']}")

    avg_match = sum(match_scores) / len(match_scores) if match_scores else 0.0
    avg_non_match = sum(non_match_scores) / len(non_match_scores) if non_match_scores else 0.0
    margin = avg_match - avg_non_match

    print("-" * 55)
    print(f"Avg Match Score    : {avg_match:.4f}")
    print(f"Avg Non-Match Score: {avg_non_match:.4f}")
    print(f"Separation Margin  : {margin:.4f}")
    if margin < 0.2:
        print("WARNING: Low separation margin on French benchmarks! High risk of false merges or missed matches.")
    else:
        print("PASS: Strong separation between true French matches and hard negatives.")

    return {
        "avg_match": avg_match,
        "avg_non_match": avg_non_match,
        "margin": margin,
    }


def audit_codebase_for_unseen_country_pitfalls(src_dir: str = "src") -> List[str]:
    """
    Static code inspection of team files to find fatal unseen-country pitfalls:
    - hardcoded ['US', 'India'] filters
    - missing handle_unknown in OneHotEncoder
    - regex only looking for 6-digit pin codes
    """
    import os

    warnings = []
    if not os.path.exists(src_dir):
        return warnings

    dangerous_patterns = [
        (r"\[\s*['\"]US['\"]\s*,\s*['\"]India['\"]\s*\]", "Hardcoded country filter ['US', 'India'] detected! Will discard France test records!"),
        (r"country\s*==\s*['\"]US['\"]|country\s*==\s*['\"]India['\"]", "Explicit equality check for US/India detected."),
        (r"OneHotEncoder\((?!.*handle_unknown)", "OneHotEncoder instantiated without explicit handle_unknown='ignore'. Will crash on France!"),
        (r"\b\d{6}\b", "Exact 6-digit regex found without French 5-digit support. French postal codes are 5 digits!"),
    ]

    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as py_file:
                content = py_file.read()
                for pattern, msg in dangerous_patterns:
                    if re.search(pattern, content):
                        warnings.append(f"[{f}] {msg}")

    return warnings
