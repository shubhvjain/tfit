import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

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

    Returns:
        A dictionary containing:
            - precision: Precision score (TP / (TP + FP)).
            - recall: Recall score (TP / (TP + FN)).
            - jaccard: Jaccard index (intersection over union).
    """
    if grn_data.empty:
        # return 0 for all scores
        return {
        'grn_collectri_precision': 0.0,
        'grn_collectri_recall': 0.0,
        'grn_collectri_jaccard': 0.0
    }

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
    # if precision + recall > 0:
    #     f1 = 2 * (precision * recall) / (precision + recall)
    # else:
    #     f1 = 0.0
        
    # Calculate Jaccard index (intersection over union)
    union = source_set | true_regulators
    jaccard = len(tp_set) / len(union) if len(union) > 0 else 0.0
    
    return {
        'grn_collectri_precision': precision,
        'grn_collectri_recall': recall,
        'grn_collectri_jaccard': jaccard
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

    # if grn.empty:
    #     raise ValueError(f"no GRN entries found for target '{target}' in '{grn_key}'")
    
    return grn


GRN_METHODS = {
    'grn_collectri': {
        'func': lambda sources, target, datasets=None, **kwargs:
            grn_set_metrics(
                sources,
                target,
                _get_GRN_for_target(datasets, 'collectri', target)
            ),
        "type":"df_columns",
        "cols":["grn_collectri_precision","grn_collectri_recall","grn_collectri_jaccard"],
        'datasets': ['collectri']
    },
}
