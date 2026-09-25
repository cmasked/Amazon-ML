"""
Official Candidate Generation / Blocking Evaluator for Role 2.
Implements the exact candidate recall metrics, per-entity analysis,
and candidate-distribution statistics specified by the competition and team standards.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Any, Optional


def evaluate_candidate_generation(
    candidates_by_s1: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    timing_stats: Optional[Dict[str, float]] = None,
    peak_ram_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Comprehensive candidate generation evaluation.
    
    Args:
        candidates_by_s1: dict mapping s1_id -> set of candidate entity IDs
        ground_truth: dict mapping s1_id -> set of ground truth entity IDs
        timing_stats: optional dict with 'index_time_s', 'query_time_s', 'total_time_s'
        peak_ram_mb: optional peak RAM in MB
        
    Returns:
        Structured evaluation metrics dict.
    """
    total_entities = len(ground_truth)
    total_true_matches = 0
    total_retrieved_true_matches = 0
    
    per_entity_recalls = []
    candidate_counts = []
    
    n_singletons = 0
    singletons_empty_cands = 0
    
    n_multi_match_entities = 0
    multi_match_retrieved = 0
    multi_match_total = 0
    
    n_full_recall = 0      # 100% recall
    n_partial_recall = 0   # 0% < recall < 100%
    n_zero_recall = 0      # 0% recall
    
    missed_records = []
    per_entity_records = []
    
    for s1_id, gt_set in ground_truth.items():
        cands_set = candidates_by_s1.get(s1_id, set())
        n_cands = len(cands_set)
        candidate_counts.append(n_cands)
        n_gt = len(gt_set)
        
        if n_gt == 0:
            # Singleton entity (no true matches)
            n_singletons += 1
            if n_cands == 0:
                singletons_empty_cands += 1
            per_entity_records.append({
                's1_id': s1_id,
                'true_matches': 0,
                'retrieved_matches': 0,
                'candidate_count': n_cands,
                'candidate_recall': 1.0 if n_cands == 0 else 0.0,
                'category': 'singleton'
            })
            continue
            
        # Entity with true matches
        total_true_matches += n_gt
        retrieved = len(cands_set & gt_set)
        total_retrieved_true_matches += retrieved
        recall = retrieved / n_gt
        per_entity_recalls.append(recall)
        
        if n_gt > 1:
            n_multi_match_entities += 1
            multi_match_retrieved += retrieved
            multi_match_total += n_gt
            
        if recall == 1.0:
            n_full_recall += 1
            cat = 'full_recall'
        elif recall > 0.0:
            n_partial_recall += 1
            cat = 'partial_recall'
        else:
            n_zero_recall += 1
            cat = 'zero_recall'
            
        # Log missed matches for hard-case mining
        missed = gt_set - cands_set
        for m_id in missed:
            missed_records.append({
                's1_id': s1_id,
                'true_source_id': m_id,
            })
            
        per_entity_records.append({
            's1_id': s1_id,
            'true_matches': n_gt,
            'retrieved_matches': retrieved,
            'candidate_count': n_cands,
            'candidate_recall': recall,
            'category': cat
        })
        
    n_matched_entities = len(per_entity_recalls)
    aggregate_recall = (
        total_retrieved_true_matches / total_true_matches 
        if total_true_matches > 0 else 0.0
    )
    macro_recall = (
        float(np.mean(per_entity_recalls)) 
        if per_entity_recalls else 0.0
    )
    full_entity_recall = (
        n_full_recall / n_matched_entities 
        if n_matched_entities > 0 else 0.0
    )
    partial_recall_rate = (
        n_partial_recall / n_matched_entities 
        if n_matched_entities > 0 else 0.0
    )
    zero_recall_rate = (
        n_zero_recall / n_matched_entities 
        if n_matched_entities > 0 else 0.0
    )
    multi_match_recall = (
        multi_match_retrieved / multi_match_total 
        if multi_match_total > 0 else 0.0
    )
    
    # Candidate statistics
    c_counts = np.array(candidate_counts) if candidate_counts else np.array([0])
    mean_cands = float(np.mean(c_counts))
    median_cands = float(np.median(c_counts))
    p90_cands = float(np.percentile(c_counts, 90))
    p95_cands = float(np.percentile(c_counts, 95))
    p99_cands = float(np.percentile(c_counts, 99))
    max_cands = int(np.max(c_counts))
    total_candidate_pairs = int(np.sum(c_counts))
    
    metrics = {
        'total_entities': total_entities,
        'matched_entities': n_matched_entities,
        'singletons': n_singletons,
        'multi_match_entities': n_multi_match_entities,
        'total_true_matches': total_true_matches,
        'total_retrieved_true_matches': total_retrieved_true_matches,
        # Primary Recall Metrics
        'candidate_recall': aggregate_recall,
        'macro_candidate_recall': macro_recall,
        'full_entity_recall': full_entity_recall,
        'partial_recall_rate': partial_recall_rate,
        'zero_recall_rate': zero_recall_rate,
        'multi_match_recall': multi_match_recall,
        # Counts
        'n_full_recall': n_full_recall,
        'n_partial_recall': n_partial_recall,
        'n_zero_recall': n_zero_recall,
        # Candidate count distribution
        'mean_candidates': mean_cands,
        'median_candidates': median_cands,
        'p90_candidates': p90_cands,
        'p95_candidates': p95_cands,
        'p99_candidates': p99_cands,
        'max_candidates': max_cands,
        'total_candidates': total_candidate_pairs,
    }
    
    if timing_stats:
        metrics.update(timing_stats)
    if peak_ram_mb is not None:
        metrics['peak_memory_mb'] = peak_ram_mb
        
    return {
        'metrics': metrics,
        'missed_pairs': pd.DataFrame(missed_records),
        'per_entity_df': pd.DataFrame(per_entity_records)
    }


def format_evaluation_summary(metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics as a clean markdown report block."""
    lines = [
        "### Candidate Generation Evaluation Results",
        f"- **Aggregate Candidate Recall:** {metrics['candidate_recall']:.4%} ({metrics['total_retrieved_true_matches']:,} / {metrics['total_true_matches']:,} true matches)",
        f"- **Macro Candidate Recall:** {metrics['macro_candidate_recall']:.4%}",
        f"- **Full Entity Recall (100% retrieved):** {metrics['full_entity_recall']:.4%} ({metrics['n_full_recall']:,} entities)",
        f"- **Partial Recall (0% < R < 100%):** {metrics['partial_recall_rate']:.4%} ({metrics['n_partial_recall']:,} entities)",
        f"- **Zero Recall (0% retrieved):** {metrics['zero_recall_rate']:.4%} ({metrics['n_zero_recall']:,} entities)",
        f"- **Multi-Match Recall:** {metrics['multi_match_recall']:.4%}",
        "",
        "#### Candidate Distribution:",
        f"- **Mean Candidates / Entity:** {metrics['mean_candidates']:.2f}",
        f"- **Median Candidates:** {metrics['median_candidates']:.1f}",
        f"- **P90 Candidates:** {metrics['p90_candidates']:.1f}",
        f"- **P95 Candidates:** {metrics['p95_candidates']:.1f}",
        f"- **P99 Candidates:** {metrics['p99_candidates']:.1f}",
        f"- **Max Candidates:** {metrics['max_candidates']:,}",
        f"- **Total Candidate Pairs:** {metrics['total_candidates']:,}",
    ]
    if 'index_time_s' in metrics:
        lines.append(f"- **Index Construction Time:** {metrics['index_time_s']:.2f}s")
    if 'query_time_s' in metrics:
        lines.append(f"- **Query Time:** {metrics['query_time_s']:.2f}s")
    if 'total_time_s' in metrics:
        lines.append(f"- **Total Retrieval Time:** {metrics['total_time_s']:.2f}s")
    if 'peak_memory_mb' in metrics:
        lines.append(f"- **Peak RAM:** {metrics['peak_memory_mb']:.1f} MB")
        
    return "\n".join(lines)
