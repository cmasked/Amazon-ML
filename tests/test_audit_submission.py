"""
Unit tests for Person 4 Submission Auditor.
Tests compliance against all competition submission rules.
"""

import os
import shutil
import tempfile
import pytest
from src.audit_submission import check_file_format_and_stream, MATCHING_HEADER, CANDIDATE_HEADER


@pytest.fixture
def temp_workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_valid_matching_file(temp_workspace):
    """Test correctly formatted matching TSV passes all checks."""
    path = os.path.join(temp_workspace, "matching_results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-101,S3-201\n")
        f.write("S1-002\t\n")  # singleton correctly empty

    required_s1 = {"S1-001", "S1-002"}
    is_valid, errors, parsed = check_file_format_and_stream(
        path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1,
    )
    assert is_valid is True
    assert len(errors) == 0
    assert parsed["S1-001"] == {"S2-101", "S3-201"}
    assert parsed["S1-002"] == set()


def test_reject_s1_self_match(temp_workspace):
    """Test that self-matching to S1 is strictly rejected."""
    path = os.path.join(temp_workspace, "matching_results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS1-002\n")  # Illegal S1 self match!

    required_s1 = {"S1-001"}
    is_valid, errors, _ = check_file_format_and_stream(
        path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1,
    )
    assert is_valid is False
    assert any("Illegal S1-" in e for e in errors)


def test_reject_duplicate_ids_in_row(temp_workspace):
    """Test that duplicate IDs in list are rejected."""
    path = os.path.join(temp_workspace, "matching_results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-100,S2-100\n")  # duplicate!

    required_s1 = {"S1-001"}
    is_valid, errors, _ = check_file_format_and_stream(
        path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1,
    )
    assert is_valid is False
    assert any("Duplicate IDs" in e for e in errors)


def test_reject_missing_s1_entities(temp_workspace):
    """Test that missing required S1 entities causes rejection."""
    path = os.path.join(temp_workspace, "matching_results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS2-100\n")

    required_s1 = {"S1-001", "S1-002", "S1-003"}  # S1-002 and S1-003 missing!
    is_valid, errors, _ = check_file_format_and_stream(
        path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1,
    )
    assert is_valid is False
    assert any("missing" in e.lower() for e in errors)


def test_reject_bad_singleton_strings(temp_workspace):
    """Test that strings like 'None' or 'NaN' in singleton rows are flagged."""
    path = os.path.join(temp_workspace, "matching_results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tnan\n")

    required_s1 = {"S1-001"}
    is_valid, errors, _ = check_file_format_and_stream(
        path,
        MATCHING_HEADER,
        "matched_entity_ids",
        required_s1,
    )
    assert is_valid is False
    assert any("singleton" in e.lower() for e in errors)
