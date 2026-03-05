import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def grn_precision_recall(
    source: List[str],
    target: str,
    grn_data: pd.DataFrame
) -> Dict:
    """Calculate precision-recall curve and related metrics using GRN for the source genes.

    Args:
        source: Predicted regulator genes in order (ranked by confidence if available).For unranked predictions, order is arbitrary.
        target: Target gene name (for validation/reference, grn_data already filtered).
        grn_data: Ground truth regulators for this target. Must contain columns: ['regulator', 'target', 'score'].  Already filtered to contain only rows where target matches.

    Returns:
        A dictionary containing:
            - max_k: Maximum k value (length of predictions).
            - values: List of (precision, recall) tuples for k=1,2,...,max_k.
            - ap: Average Precision score.
            - precisions_at_tp: List of (k, precision) tuples at each true positive.
            - k_for_50_percent_recall: k value needed for 50% recall (or None).
            - k_for_90_percent_recall: k value needed for 90% recall (or None).
            - num_true_regulators: Total true regulators in ground truth.
            - num_predictions: Total predictions (len(source)).

    """
    
    # Handle edge case: no predictions
    if len(source) == 0:
        raise ValueError("source is empty")
    
    # Handle edge case: no true regulators
    if len(grn_data) == 0:
        raise ValueError("grn_data is empty")
    
    # Get set of true regulators
    true_regulators = set(grn_data['regulator'].values)
    num_true = len(true_regulators)
    
    # Remove duplicates from source while preserving order
    # seen = set()
    # source_unique = []
    # for gene in source:
    #     if gene not in seen:
    #         seen.add(gene)
    #         source_unique.append(gene)
    # source = source_unique
    
    # Initialize metric tracking variables
    pr_values = []
    precisions_at_tp = []
    k_for_50_recall = None
    k_for_90_recall = None
    
    # Calculate precision and recall at each k
    for k in range(1, len(source) + 1):
        # Get top-k predictions
        top_k = source[:k]
        
        # Count true positives
        tp = len([g for g in top_k if g in true_regulators])
        
        # Calculate precision and recall
        precision = tp / k if k > 0 else 0.0
        recall = tp / num_true if num_true > 0 else 0.0
        
        pr_values.append((precision, recall))
        
        # Record precision at true positive positions 
        if source[k-1] in true_regulators:
            precisions_at_tp.append((k, precision))
        
        # Record k values for recall thresholds
        if k_for_50_recall is None and recall >= 0.5:
            k_for_50_recall = k
        if k_for_90_recall is None and recall >= 0.9:
            k_for_90_recall = k
    
    # Calculate Average Precision (AP)
    # AP is the sum of precisions at each true positive position divided by total true positives

    
    return {
        'values': pr_values,
        'precisions_at_tp': precisions_at_tp,
        'k_for_50_percent_recall': k_for_50_recall,
        'k_for_90_percent_recall': k_for_90_recall,
        'num_true_regulators': num_true,
        'num_predictions': len(source)
    }


def grn_set_metrics(
    source: List[str],
    target: str,
    grn_data: pd.DataFrame
) -> Dict:
    """Calculate set-based metrics treating predictions as a single set.

    Args:
        source: Predicted regulator genes 
        target: Target gene name (for validation/reference).
        grn_data: Ground truth regulators for this target.
            Must contain columns: ['regulator', 'target', 'score'].
        beta: Beta parameter for F-beta score. Defaults to 1.0.
            - beta=1.0 gives F1 score (equal weight to precision/recall)
            - beta>1.0 weights recall higher
            - beta<1.0 weights precision higher

    Returns:
        A dictionary containing:
            - precision: Precision score (TP / (TP + FP)).
            - recall: Recall score (TP / (TP + FN)).
            - f1: F1 score (harmonic mean of precision and recall).
            - jaccard: Jaccard index (intersection over union).
            - tp: True positives count.
            - fp: False positives count.
            - fn: False negatives count.
            - num_predicted: Number of predicted regulators.
            - num_true: Number of true regulators.
            - overlap: List of genes in both predicted and true sets.

    """
    
    # Remove duplicates from source
    source_set = set(source)
    
    # Get true regulators
    true_regulators = set(grn_data['regulator'].values)
    
    # Calculate set intersections and differences
    tp_set = source_set & true_regulators  # Intersection (correctly predicted)
    fp_set = source_set - true_regulators  # Predicted but not true
    fn_set = true_regulators - source_set  # True but not predicted
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    num_predicted = len(source_set)
    num_true = len(true_regulators)
    
    # Calculate precision and recall
    precision = tp / num_predicted if num_predicted > 0 else 0.0
    recall = tp / num_true if num_true > 0 else 0.0
    
    # Calculate F1 score (harmonic mean)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # Calculate Jaccard index (intersection over union)
    union = source_set | true_regulators
    jaccard = len(tp_set) / len(union) if len(union) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }


def _get_GRN_for_target(
    dataset: Dict,
    grn_key: str,
    target: str
) -> pd.DataFrame:
    """Retrieve and filter GRN data for a specific target gene.

    Args:
        dataset: Dictionary of loaded GRN datasets.
        grn_key: Key identifying which GRN to use (e.g. 'collectri').
        target: Target gene name to filter for.

    Returns:
        DataFrame with columns ['regulator', 'target', 'score'],
        filtered to rows where target matches.
    """
    if dataset is None:
        raise ValueError("dataset is None")
    if grn_key is None:
        raise ValueError("no grn_key provided")
    if target is None:
        raise ValueError("no target provided")

    if grn_key == "collectri":
        grn = dataset["collectri"].copy()
        grn = grn.rename(columns={"source": "regulator"})
        grn = grn[grn["target"] == target][["regulator", "target", "weight"]].rename(
            columns={"weight": "score"}
        )
    else:
        raise ValueError(f"invalid grn_key: '{grn_key}'")

    if grn.empty:
        raise ValueError(f"no GRN entries found for target '{target}' in '{grn_key}'")

    return grn


GRN_METHODS = {
    # 'grn_precision_recall_collectri': {
    #     'func': lambda sources, target, datasets=None, **kwargs:
    #        grn_precision_recall(
    #             sources,
    #             target,
    #             _get_GRN_for_target(datasets, 'collectri', target)
    #         ),
    #      "type":"json",
    #     "cols":["grn_precision_recall_collectri"],
    #     'datasets': ['collectri']
    # },
    'grn_collectri': {
        'func': lambda sources, target, datasets=None, **kwargs:
            grn_set_metrics(
                sources,
                target,
                _get_GRN_for_target(datasets, 'collectri', target)
            ),
        "type":"df_columns",
        "cols":["grn_collectri_precision","grn_collectri_recall","grn_collectri_f1","grn_collectri_jaccard"],
        'datasets': ['collectri']
    },
}
