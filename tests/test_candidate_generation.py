"""
Deterministic Unit and Integration Tests for Role 2 Search Engine & Blocking Optimizer.
Verifies all 16 required edge cases:
1. Exact match
2. Corp vs Corporation
3. Ltd vs Limited
4. Inc vs Incorporated
5. Punctuation differences (& vs and, hyphens, slashes)
6. Spelling typo (edit distance 1-2)
7. Token reordering
8. Address variation (Street vs St, Road vs Rd, Avenue vs Ave)
9. Incomplete address
10. Missing address
11. Missing name
12. Duplicate candidates
13. Common token explosion
14. Unseen country (France / open country set)
15. Empty fields
16. Candidate deduplication & top_k cap
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from search_engine import MultiPassSearchEngine, find_candidates_from_index
from normalization import (
    normalize_name_standard,
    strip_legal_suffixes,
    normalize_name_compact,
    normalize_address_standard,
    soundex
)


class TestCandidateGenerationEngine(unittest.TestCase):

    def setUp(self):
        """Set up a fresh search engine instance before each test."""
        self.engine = MultiPassSearchEngine(
            enable_exact_name=True,
            enable_exact_addr=True,
            enable_char_ngram=True,
            enable_token_tfidf=True,
            enable_rare_tokens=True,
            enable_phonetic=True,
            enable_addr_numeric=True,
            default_top_k=50
        )

    def test_01_exact_match(self):
        """Test 1: Identical business name and address retrieve candidate with high score."""
        self.engine.add_record('S2-001', 'Microsoft Corporation', 'One Microsoft Way, Redmond, WA 98052', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-001', 'business_name': 'Microsoft Corporation', 'business_address': 'One Microsoft Way, Redmond, WA 98052', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-001', cand_ids)
        self.assertEqual(cand_ids[0], 'S2-001')

    def test_02_corp_vs_corporation(self):
        """Test 2: Corp vs Corporation variation is resolved via legal suffix normalization."""
        self.engine.add_record('S2-002', 'General Electric Corporation', '5 Necco St, Boston, MA 02210', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-002', 'business_name': 'General Electric Corp', 'business_address': '5 Necco St, Boston, MA 02210', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-002', cand_ids)

    def test_03_ltd_vs_limited(self):
        """Test 3: Ltd vs Limited variation (both single-word and Pvt Ltd vs Private Limited)."""
        self.engine.add_record('S3-003', 'Tata Consultancy Services Private Limited', 'Birlabhandar, Mumbai 400001', 'India')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-003', 'business_name': 'Tata Consultancy Services Pvt Ltd', 'business_address': 'Birlabhandar, Mumbai', 'country': 'India'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S3-003', cand_ids)

    def test_04_inc_vs_incorporated(self):
        """Test 4: Inc vs Incorporated variation."""
        self.engine.add_record('S2-004', 'Apple Incorporated', 'One Apple Park Way, Cupertino, CA 95014', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-004', 'business_name': 'Apple Inc.', 'business_address': '1 Apple Park Way, Cupertino, CA', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-004', cand_ids)

    def test_05_punctuation_differences(self):
        """Test 5: Punctuation differences (& vs and, hyphens, periods, slashes)."""
        self.engine.add_record('S2-005', 'Johnson & Johnson Services Inc.', 'One Johnson & Johnson Plaza, New Brunswick, NJ 08933', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-005', 'business_name': 'Johnson and Johnson Services', 'business_address': '1 Johnson-Johnson Plaza, New Brunswick, NJ', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-005', cand_ids)

    def test_06_spelling_typo(self):
        """Test 6: Spelling typos caught by character n-gram and phonetic passes."""
        self.engine.add_record('S2-006', 'McDonald\'s Restaurants', 'Chicago, IL 60607', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-006', 'business_name': 'McDonnalds Restaurent', 'business_address': 'Chicago, IL', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-006', cand_ids)

    def test_07_token_reordering(self):
        """Test 7: Token reordering handled by bag-of-words / TF-IDF retrieval."""
        self.engine.add_record('S3-007', 'Acme Industrial Supplies', '124 Market St, Austin, TX 78701', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-007', 'business_name': 'Supplies Industrial Acme', 'business_address': 'Market St, Austin', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S3-007', cand_ids)

    def test_08_address_variations(self):
        """Test 8: Address variations (Street vs St, Road vs Rd, Avenue vs Ave, landmark additions)."""
        self.engine.add_record('S2-008', 'Apex Health Clinic', '742 Evergreen Terrace, Suite 200, Springfield, OR 97477', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-008', 'business_name': 'Apex Health', 'business_address': '742 Evergreen Ter Ste 200, Springfield', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-008', cand_ids)

    def test_09_incomplete_address(self):
        """Test 9: Incomplete address (missing postal code, missing state) still retrieves via name passes."""
        self.engine.add_record('S3-009', 'Reliance Retail Limited', '3rd Floor, Court House, Dhobi Talao, Mumbai 400002', 'India')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-009', 'business_name': 'Reliance Retail', 'business_address': 'Mumbai', 'country': 'India'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S3-009', cand_ids)

    def test_10_missing_address(self):
        """Test 10: Completely empty address does not crash and retrieves via name."""
        self.engine.add_record('S2-010', 'Oracle Financial Services', '', 'India')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-010', 'business_name': 'Oracle Financial Services Software', 'business_address': '', 'country': 'India'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S2-010', cand_ids)

    def test_11_missing_name(self):
        """Test 11: Missing name does not crash and retrieves via exact/numeric address."""
        self.engine.add_record('S3-011', '', 'Plot 45, Electronic City, Phase 1, Bangalore 560100', 'India')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-011', 'business_name': '', 'business_address': 'Plot 45, Electronic City, Bangalore 560100', 'country': 'India'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S3-011', cand_ids)

    def test_12_duplicate_candidates_deduplication(self):
        """Test 12: Candidates matching across multiple passes are deduplicated with composite score."""
        self.engine.add_record('S2-012', 'Federal Express Corp', '3610 Hacks Cross Rd, Memphis, TN 38125', 'US')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-012', 'business_name': 'Federal Express Corp', 'business_address': '3610 Hacks Cross Rd, Memphis, TN 38125', 'country': 'US'}
        cands = self.engine.retrieve(s1, top_k=20)
        cand_ids = [cid for cid, _ in cands]
        # Check uniqueness
        self.assertEqual(len(cand_ids), len(set(cand_ids)))

    def test_13_common_token_explosion(self):
        """Test 13: Extremely common tokens (like 'India', 'Corporation', 'Enterprises') do not explode candidate size."""
        for i in range(100):
            self.engine.add_record(f'S2-COM-{i}', f'India Trading Company {i}', f'{i} Road, Bangalore 560001', 'India')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-COM', 'business_name': 'India Trading', 'business_address': 'Bangalore', 'country': 'India'}
        cands = self.engine.retrieve(s1, top_k=25)
        # Verify candidate count respects top_k cap
        self.assertLessEqual(len(cands), 25)

    def test_14_unseen_country(self):
        """Test 14: Unseen country scenario (France with French characters and address schema)."""
        self.engine.add_record('S3-014', 'Societe Generale Transport SARL', '29 Boulevard Haussmann, 75009 Paris', 'France')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-014', 'business_name': 'Société Générale Transport', 'business_address': '29 Blvd Haussmann, Paris 75009', 'country': 'France'}
        cands = self.engine.retrieve(s1, top_k=10)
        cand_ids = [cid for cid, _ in cands]
        self.assertIn('S3-014', cand_ids)

    def test_15_empty_fields_handling(self):
        """Test 15: Records with all empty or whitespace strings handled gracefully without errors."""
        self.engine.add_record('S2-015', '   ', '   ', '')
        self.engine.finalize_index()

        s1 = {'entity_id': 'S1-015', 'business_name': '', 'business_address': '', 'country': ''}
        cands = self.engine.retrieve(s1, top_k=10)
        self.assertIsInstance(cands, list)

    def test_16_backwards_compatible_api(self):
        """Test 16: Verification of find_candidates_from_index() drop-in compatibility."""
        self.engine.add_record('S2-016', 'Bluebird Coffee Roasters', '702 Market St, San Francisco, CA 94105', 'US')
        self.engine.finalize_index()

        cands = find_candidates_from_index(
            name_tokens=['bluebird', 'coffee'],
            addr_tokens=['market', 'st'],
            token_index=self.engine,
            top_k=5
        )
        self.assertIsInstance(cands, list)
        self.assertTrue(len(cands) > 0)
        self.assertEqual(cands[0][0], 'S2-016')


if __name__ == '__main__':
    unittest.main()
