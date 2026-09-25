"""
Unit tests for France & Unseen-Country Stress Test Suite.
Verifies unicode handling, diacritic normalization, French postal extraction,
and audits against unseen country vulnerabilities.
"""

import pytest
from src.france_stress import (
    normalize_french_text,
    extract_french_postal_code,
    audit_similarity_function_on_france,
    audit_codebase_for_unseen_country_pitfalls,
    FRENCH_SYNTHETIC_BENCHMARKS,
)


def test_french_text_normalization_preserves_letters():
    """
    Ensure unicode accents (é, è, ê, ç, à, etc.) are converted to ASCII base characters
    rather than dropped by naive regexes.
    e.g. 'Société' must become 'societe', NOT 'socit'!
    """
    sample = "Société Générale & Pâtisserie Française"
    normalized = normalize_french_text(sample, keep_accents=False)

    assert "societe" in normalized
    assert "generale" in normalized
    assert "patisserie" in normalized
    assert "francaise" in normalized
    # Ampersand handled
    assert " and " in normalized


def test_french_postal_code_extraction():
    """
    French postal codes are 5 digits (01000 - 98890).
    Verify extraction from diverse formats including CEDEX.
    """
    assert extract_french_postal_code("14 Boulevard Saint-Germain, 75005 Paris") == "75005"
    assert extract_french_postal_code("29 Bvd Haussmann 75009 Paris CEDEX 09") == "75009"
    assert extract_french_postal_code("8 Rue de la République, 69002 Lyon") == "69002"
    assert extract_french_postal_code("5 Avenue Jean Jaurès, 31000 Toulouse") == "31000"
    assert extract_french_postal_code("No postal code here") is None


def test_similarity_function_audit_benchmark():
    """
    Verify the auditing tool runs cleanly on a reference string similarity function.
    """
    # Simple character bigram Jaccard similarity
    def char_bigram_jaccard(s1: str, s2: str) -> float:
        s1_norm = normalize_french_text(s1)
        s2_norm = normalize_french_text(s2)
        if not s1_norm or not s2_norm:
            return 0.0
        bg1 = {s1_norm[i:i+2] for i in range(len(s1_norm) - 1)}
        bg2 = {s2_norm[i:i+2] for i in range(len(s2_norm) - 1)}
        if not bg1 or not bg2:
            return 0.0
        return len(bg1 & bg2) / len(bg1 | bg2)

    results = audit_similarity_function_on_france(char_bigram_jaccard, "Char Bigram Jaccard")
    assert results["avg_match"] > results["avg_non_match"]
    assert results["margin"] > 0.1
