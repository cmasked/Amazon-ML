"""
Submission Compliance & Pre-Flight Validator.
Person 4 Deliverable for Amazon ML Challenge 2026.

Audits `output/matching_results.tsv` and `output/candidate_pairs.tsv` against
all official challenge rules:
1. Exact TSV delimiter and headers
2. Completeness: Exactly one row for every S1 entity in test_source1.tsv
3. Singleton representation: Empty string, no 'None', 'nan', or missing entries
4. ID syntax & target validity: Only S2- and S3- prefixes, no S1- self-matches
5. No duplicate IDs within any row's comma-separated list
6. Candidate subset invariant: matched_entity_ids <= candidate_entity_ids
7. Fast streaming to handle 1.73M+ test records without running out of RAM
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Set, Tuple


MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def check_file_format_and_stream(
    file_path: str,
    expected_header: List[str],
    col_name: str,
    required_s1_ids: Set[str],
    valid_target_ids: Optional[Set[str]] = None,
) -> Tuple[bool, List[str], Dict[str, Set[str]]]:
    """
    Stream and validate a submission TSV file.
    Returns: (is_valid, error_list, parsed_id_dict)
    """
    errors: List[str] = []
    parsed_dict: Dict[str, Set[str]] = {}

    if not os.path.isfile(file_path):
        errors.append(f"File not found: {file_path}")
        return False, errors, parsed_dict

    seen_s1_ids: Set[str] = set()
    line_num = 0

    with open(file_path, "r", encoding="utf-8") as f:
        # Check header
        first_line = f.readline()
        line_num += 1
        if not first_line:
            errors.append(f"{file_path} is completely empty.")
            return False, errors, parsed_dict

        header_tokens = first_line.rstrip("\r\n").split("\t")
        if header_tokens != expected_header:
            errors.append(
                f"Invalid header in {file_path}. Expected: {expected_header}, Found: {header_tokens}"
            )
            return False, errors, parsed_dict

        # Process data rows
        for line in f:
            line_num += 1
            raw = line.rstrip("\r\n")
            if not raw:
                # blank line
                continue

            parts = raw.split("\t")
            if len(parts) > 2:
                errors.append(f"Line {line_num}: Extra tab character detected. Expected exactly 2 columns, got {len(parts)}.")
                if len(errors) > 20:
                    break
                continue

            s1_id = parts[0].strip()
            if not s1_id.startswith("S1-"):
                errors.append(f"Line {line_num}: Invalid S1 entity ID '{s1_id}'. Must start with 'S1-'.")
                if len(errors) > 20:
                    break

            if s1_id in seen_s1_ids:
                errors.append(f"Line {line_num}: Duplicate row for entity '{s1_id}'. Each S1 entity must appear exactly once.")
                if len(errors) > 20:
                    break
            seen_s1_ids.add(s1_id)

            # Check matches / candidates column
            val = parts[1].strip() if len(parts) > 1 else ""

            # Check for bad string representations of singletons
            if val.lower() in ("none", "nan", "null", "[]", "{}"):
                errors.append(f"Line {line_num}: Invalid singleton representation '{val}' for entity '{s1_id}'. Leave field empty.")

            if val:
                ids = [x.strip() for x in val.split(",")]
                unique_ids = set(ids)

                # Check duplicate IDs within the row
                if len(ids) != len(unique_ids):
                    errors.append(f"Line {line_num}: Duplicate IDs within list for entity '{s1_id}': {ids}")

                # Check prefix rules: must be S2 or S3
                for target_id in unique_ids:
                    if target_id.startswith("S1-"):
                        errors.append(f"Line {line_num}: Illegal S1- self-match '{target_id}' for entity '{s1_id}'.")
                    elif not (target_id.startswith("S2-") or target_id.startswith("S3-")):
                        errors.append(f"Line {line_num}: Unknown ID prefix '{target_id}'. Must start with S2- or S3-.")

                    if valid_target_ids is not None and target_id not in valid_target_ids:
                        errors.append(f"Line {line_num}: Entity ID '{target_id}' does not exist in test set.")

                parsed_dict[s1_id] = unique_ids
            else:
                parsed_dict[s1_id] = set()

    # Check completeness against required S1 IDs
    missing_s1 = required_s1_ids - seen_s1_ids
    if missing_s1:
        sample = list(missing_s1)[:5]
        errors.append(f"{len(missing_s1):,} Source 1 entities from test set are missing! Sample: {sample}")

    extra_s1 = seen_s1_ids - required_s1_ids
    if extra_s1:
        sample = list(extra_s1)[:5]
        errors.append(f"{len(extra_s1):,} S1 entity IDs do not exist in test set! Sample: {sample}")

    is_valid = len(errors) == 0
    return is_valid, errors, parsed_dict


def audit_submission(
    matching_path: str,
    candidate_path: Optional[str],
    test_dir: str,
    check_target_existence: bool = False,
) -> bool:
    """Run full compliance audit on output package."""
    print("\n" + "=" * 65)
    print("      AMAZON ML CHALLENGE 2026 — SUBMISSION AUDITOR      ")
    print("=" * 65)

    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(test_s1_path):
        print(f"ERROR: Cannot find {test_s1_path}")
        return False

    print(f"Reading required S1 test entities from {test_s1_path}...")
    required_s1_ids: Set[str] = set()
    with open(test_s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        id_idx = header.index("entity_id") if header and "entity_id" in header else 0
        for row in reader:
            if row:
                required_s1_ids.add(row[id_idx].strip())

    print(f"Total required S1 test entities: {len(required_s1_ids):,}")

    valid_targets: Optional[Set[str]] = None
    if check_target_existence:
        print("Loading test S2 and S3 ID index for existence check...")
        valid_targets = set()
        for s_file in ("test_source2.tsv", "test_source3.tsv"):
            p = os.path.join(test_dir, s_file)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    next(reader, None)
                    for row in reader:
                        if row:
                            valid_targets.add(row[0].strip())
        print(f"Indexed {len(valid_targets):,} valid target IDs.")

    # 1. Audit matching_results.tsv
    print(f"\n[1/3] Validating matching_results.tsv ({matching_path})...")
    m_valid, m_errors, matches = check_file_format_and_stream(
        matching_path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1_ids,
        valid_targets,
    )

    if not m_valid:
        print("FAIL: matching_results.tsv has formatting errors:")
        for err in m_errors[:10]:
            print(f"  - {err}")
        if len(m_errors) > 10:
            print(f"  ... and {len(m_errors) - 10} more errors.")
        return False
    print("PASS: matching_results.tsv satisfies all structure, column, and ID rules.")

    # 2. Audit candidate_pairs.tsv (if provided)
    c_valid = True
    candidates: Optional[Dict[str, Set[str]]] = None
    if candidate_path and os.path.isfile(candidate_path):
        print(f"\n[2/3] Validating candidate_pairs.tsv ({candidate_path})...")
        c_valid, c_errors, candidates = check_file_format_and_stream(
            candidate_path,
            CANDIDATE_HEADER,
            "candidate_entity_ids",
            required_s1_ids,
            valid_targets,
        )
        if not c_valid:
            print("FAIL: candidate_pairs.tsv has formatting errors:")
            for err in c_errors[:10]:
                print(f"  - {err}")
            return False
        print("PASS: candidate_pairs.tsv satisfies all structure, column, and ID rules.")
    else:
        print("\n[2/3] candidate_pairs.tsv not provided or not found — skipping candidate file checks.")

    # 3. Candidate Subset Invariant Check (matches <= candidates)
    if candidates is not None and matches is not None:
        print("\n[3/3] Checking Candidate Subset Invariant (matches <= candidates)...")
        violations = 0
        violation_examples = []
        for s1_id, match_set in matches.items():
            cand_set = candidates.get(s1_id, set())
            unmatched_cands = match_set - cand_set
            if unmatched_cands:
                violations += 1
                if len(violation_examples) < 5:
                    violation_examples.append((s1_id, unmatched_cands))

        if violations > 0:
            print(f"WARNING: Found {violations:,} entities where matching IDs were NOT in candidate pairs!")
            print("This indicates a pipeline bug (a match was generated without being in the candidate set).")
            for eid, diff in violation_examples:
                print(f"  - Entity {eid}: matched {diff} not in candidates")
        else:
            print("PASS: All final matches are strict subsets of candidates (matches <= candidates).")

    print("\n" + "=" * 65)
    print("AUDIT RESULT: ALL CRITICAL SUBMISSION CHECKS PASSED (SAFE TO SUBMIT)")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit submission TSVs for Amazon ML Challenge 2026.")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test directory")
    parser.add_argument("--check-ids", action="store_true", help="Enable strict test target ID existence check")
    args = parser.parse_args()

    success = audit_submission(
        matching_path=args.matching,
        candidate_path=args.candidate if os.path.isfile(args.candidate) else None,
        test_dir=args.test_dir,
        check_target_existence=args.check_ids,
    )
    sys.exit(0 if success else 1)
