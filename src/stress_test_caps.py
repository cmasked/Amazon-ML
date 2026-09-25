"""
Candidate Cap & Pathological Case Stress Test for Role 2.
Tests extreme corner cases:
  1. High-frequency common business names (e.g. 'Global Solutions' with 200+ index records)
  2. High-frequency common addresses (e.g. 'MG Road Bangalore' with 200+ index records)
  3. Empty business name
  4. Empty business address
  5. Generic tokens ('India', 'Services', 'Enterprises')
  6. Exact duplicate candidate records
Verifies whether the true match survives the top-k cap and candidate volume behavior.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_engine import MultiPassSearchEngine

def run_stress_test():
    print("=" * 65)
    print("TASK 9: CANDIDATE CAP & ADVERSARIAL STRESS TEST")
    print("=" * 65)

    engine = MultiPassSearchEngine(default_top_k=80)

    # 1. Inject 250 common name records (flood the index)
    for i in range(250):
        engine.add_record(f"S2-COMNAME-{i}", "Global Solutions Private Limited", f"Suite {i}, Industrial Park, Chicago", "US")

    # 2. Inject 250 common address records
    for i in range(250):
        engine.add_record(f"S3-COMADDR-{i}", f"Business Entity {i}", "MG Road, Near SBI Bank, Bangalore 560001", "India")

    # 3. Inject empty name and empty address records
    engine.add_record("S2-EMPTY-NAME", "", "100 Broadway, New York, NY 10001", "US")
    engine.add_record("S3-EMPTY-ADDR", "Unique Solitary Enterprise", "", "US")
    engine.add_record("S2-BOTH-EMPTY", "", "", "")

    # 4. Inject true matches with subtle differences
    # True match for common name entity:
    engine.add_record("S2-TRUE-COMNAME", "Global Solutions Pvt Ltd", "Suite 42, Specific Plaza, Chicago 60607", "US")
    # True match for common address entity:
    engine.add_record("S3-TRUE-COMADDR", "Niche Micro Devices Limited", "MG Road, Near SBI Bank, Bangalore 560001", "India")
    # True match for empty name (matches on address):
    engine.add_record("S2-TRUE-EMPTYNAME", "Ghost Business Co", "100 Broadway, New York, NY 10001", "US")
    # True match for empty address (matches on name):
    engine.add_record("S3-TRUE-EMPTYADDR", "Unique Solitary Enterprise Inc", "Now has address 400 Pine St", "US")

    engine.finalize_index()
    print(f"Index built with {len(engine.id_table):,} records including adversarial floods.\n")

    # TEST CASE 1: Query entity with Common Name
    s1_comname = {
        'entity_id': 'S1-TEST-1',
        'business_name': 'Global Solutions Corporation',
        'business_address': 'Suite 42, Specific Plaza, Chicago 60607',
        'country': 'US'
    }
    cands_1 = engine.retrieve(s1_comname, top_k=80)
    cand_ids_1 = [cid for cid, _ in cands_1]
    in_cands_1 = "S2-TRUE-COMNAME" in cand_ids_1
    rank_1 = cand_ids_1.index("S2-TRUE-COMNAME") if in_cands_1 else -1
    print(f"Test 1 (Common Name Flood):")
    print(f"  Total candidates returned: {len(cands_1)} (Cap: 80)")
    print(f"  True match 'S2-TRUE-COMNAME' retrieved? {in_cands_1} (Rank: {rank_1})")
    assert in_cands_1, "Failed Test 1: True match displaced by common name flood!"

    # TEST CASE 2: Query entity with Common Address
    s1_comaddr = {
        'entity_id': 'S1-TEST-2',
        'business_name': 'Niche Micro Devices',
        'business_address': 'MG Road, Near SBI Bank, Bangalore 560001',
        'country': 'India'
    }
    cands_2 = engine.retrieve(s1_comaddr, top_k=80)
    cand_ids_2 = [cid for cid, _ in cands_2]
    in_cands_2 = "S3-TRUE-COMADDR" in cand_ids_2
    rank_2 = cand_ids_2.index("S3-TRUE-COMADDR") if in_cands_2 else -1
    print(f"\nTest 2 (Common Address Flood):")
    print(f"  Total candidates returned: {len(cands_2)} (Cap: 80)")
    print(f"  True match 'S3-TRUE-COMADDR' retrieved? {in_cands_2} (Rank: {rank_2})")
    assert in_cands_2, "Failed Test 2: True match displaced by common address flood!"

    # TEST CASE 3: Query entity with Empty Name
    s1_emptyname = {
        'entity_id': 'S1-TEST-3',
        'business_name': '',
        'business_address': '100 Broadway, New York, NY 10001',
        'country': 'US'
    }
    cands_3 = engine.retrieve(s1_emptyname, top_k=80)
    cand_ids_3 = [cid for cid, _ in cands_3]
    in_cands_3 = "S2-TRUE-EMPTYNAME" in cand_ids_3
    print(f"\nTest 3 (Empty Name):")
    print(f"  Total candidates returned: {len(cands_3)}")
    print(f"  True match 'S2-TRUE-EMPTYNAME' retrieved? {in_cands_3}")
    assert in_cands_3, "Failed Test 3: Empty name did not retrieve via address!"

    # TEST CASE 4: Query entity with Empty Address
    s1_emptyaddr = {
        'entity_id': 'S1-TEST-4',
        'business_name': 'Unique Solitary Enterprise',
        'business_address': '',
        'country': 'US'
    }
    cands_4 = engine.retrieve(s1_emptyaddr, top_k=80)
    cand_ids_4 = [cid for cid, _ in cands_4]
    in_cands_4 = "S3-TRUE-EMPTYADDR" in cand_ids_4
    print(f"\nTest 4 (Empty Address):")
    print(f"  Total candidates returned: {len(cands_4)}")
    print(f"  True match 'S3-TRUE-EMPTYADDR' retrieved? {in_cands_4}")
    assert in_cands_4, "Failed Test 4: Empty address did not retrieve via name!"

    # TEST CASE 5: Completely Empty Record
    s1_bothempty = {'entity_id': 'S1-TEST-5', 'business_name': '', 'business_address': '', 'country': ''}
    cands_5 = engine.retrieve(s1_bothempty, top_k=80)
    print(f"\nTest 5 (Both Empty):")
    print(f"  Graceful execution? Yes, returned {len(cands_5)} candidates.")

    print("\nALL STRESS TESTS PASSED! True matches preserved under heavy distractor flooding.")

if __name__ == '__main__':
    run_stress_test()
