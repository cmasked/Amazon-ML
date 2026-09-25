"""
Write submission files: matching_results.tsv and candidate_pairs.tsv
"""
import os


def write_matching_results(predictions, output_path, all_s1_ids=None):
    """
    Write matching_results.tsv.
    
    Args:
        predictions: dict mapping s1_id -> set of matched entity IDs
        output_path: path to write the file
        all_s1_ids: optional list of ALL S1 entity IDs to ensure every one has a row
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine which S1 IDs to write
    if all_s1_ids:
        s1_ids = sorted(all_s1_ids)
    else:
        s1_ids = sorted(predictions.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1_ids:
            matched = predictions.get(s1_id, set())
            matched_str = ','.join(sorted(matched)) if matched else ''
            f.write(f'{s1_id}\t{matched_str}\n')
    
    print(f"  Written {len(s1_ids)} rows to {output_path}")
    n_empty = sum(1 for sid in s1_ids if not predictions.get(sid, set()))
    n_nonempty = len(s1_ids) - n_empty
    print(f"  Empty (singletons): {n_empty}, Non-empty: {n_nonempty}")


def write_candidate_pairs(candidates, output_path, all_s1_ids=None):
    """
    Write candidate_pairs.tsv.
    
    Args:
        candidates: dict mapping s1_id -> set of candidate entity IDs
        output_path: path to write the file
        all_s1_ids: optional list of ALL S1 entity IDs
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if all_s1_ids:
        s1_ids = sorted(all_s1_ids)
    else:
        s1_ids = sorted(candidates.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in s1_ids:
            cands = candidates.get(s1_id, set())
            cands_str = ','.join(sorted(cands)) if cands else ''
            f.write(f'{s1_id}\t{cands_str}\n')
    
    print(f"  Written {len(s1_ids)} rows to {output_path}")
