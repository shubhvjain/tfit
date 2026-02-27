import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import hypergeom
from tfitpy.utils import generate_tf_pairs



def get_targets(grn_graph: nx.DiGraph, node: str) -> set:
    """Get all direct targets (out-neighbors) of a regulator."""
    if node not in grn_graph:
        return set()
    return set(grn_graph.successors(node))


def hypergeom_overlap_pvalue(N: int, N1: int, N2: int, c: int) -> float:
    """Compute upper-tail hypergeometric P-value: P(X >= c)."""
    if c == 0 or c > min(N1, N2):
        return 1.0
    return float(hypergeom.sf(c - 1, N, N1, N2))


def target_overlap_score_for_pair(
    reg1: str, reg2: str, 
    grn_graph: nx.DiGraph, 
    background_size: int
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute hypergeometric score for target overlap between two regulators.
    
    Returns:
        S (float): -log10(P) where P is hypergeometric P-value
        info (dict): counts and P-value for debugging
    """
    targets1 = get_targets(grn_graph, reg1)
    targets2 = get_targets(grn_graph, reg2)
    
    N1 = len(targets1)
    N2 = len(targets2)
    
    if N1 == 0 or N2 == 0:
        return 0.0, {"reg1": reg1, "reg2": reg2, "N1": N1, "N2": N2, "c": 0, "pvalue": 1.0}
    
    common_targets = targets1 & targets2
    c = len(common_targets)
    
    if c == 0:
        return 0.0, {"reg1": reg1, "reg2": reg2, "N1": N1, "N2": N2, "c": 0, "pvalue": 1.0}
    
    P = hypergeom_overlap_pvalue(background_size, N1, N2, c)
    
    # Score = -log10(P), handle P=0
    S = float("inf") if P <= 0 else -np.log10(P)
    
    return float(S), {
        "reg1": reg1, "reg2": reg2, 
        "N1": N1, "N2": N2, "c": c, 
        "pvalue": P, "score": S
    }


def target_overlap_score(   
    sources: list,
    target: str,
    background_size: int = None,
    aggregation_method: str = "mean",
    datasets = None,
    pairs = None,
    **args
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Compute target overlap significance score for all regulator pairs in module.
    
    Matches structure of hypergeom_index_score() exactly.
    
    Args:
        config: Database config (passed through)
        module: Dict with "gene_cluster" key containing regulator list
        grn_graph: Pre-loaded GRN DiGraph (auto-loads if None)
        aggregation_method: "mean", "median", or "max"
        background_size: Universe size (auto-detects if None)
    
    Returns:
        final_score, pairs_df, metadata (exact same format as hypergeom_index_score)
    """

    if datasets is None:
        raise ValueError("datasets cache is required. Create cache with load_datasets() first.")
    
    if "collectri" not in datasets:
        raise ValueError("Dataset dependency missing")
    
    grn_graph = datasets["collectri"]

    
    if background_size is None:
        background_size = len(grn_graph.nodes())
    
    if pairs is None:
        pairs = generate_tf_pairs(sources)
    
    pair_results = []
    
    for reg1, reg2 in pairs:
        score, info = target_overlap_score_for_pair(
            reg1, reg2, grn_graph, background_size
        )
        row = {
            "tf1": info["reg1"],     
            "tf2": info["reg2"],
            "target_overlap_score": score,
            "N1": info["N1"], 
            "N2": info["N2"],  
            "c": info["c"],   
            "pvalue": info["pvalue"]
        }
        pair_results.append(row)
    
    pairs_df = pd.DataFrame(pair_results)
    
    # Aggregate scores (ignore inf/-inf) - exact same logic as hypergeom_index_score
    valid_scores = pairs_df["target_overlap_score"].replace([np.inf, -np.inf], np.nan)
    valid_scores = valid_scores.dropna()
    
    if len(valid_scores) == 0:
        final_score = 0.0
    else:
        if aggregation_method == "mean":
            final_score = float(np.mean(valid_scores))
        elif aggregation_method == "median":
            final_score = float(np.median(valid_scores))
        elif aggregation_method == "max":
            final_score = float(np.max(valid_scores))
        else:
            final_score = float(np.mean(valid_scores))
    
    metadata = {
        "input_module": sources,
        "total_pairs": len(pairs),
        "valid_pairs": int(len(valid_scores)),
        "aggregation_method": aggregation_method,
        "background_size": background_size,
        "all_aggregations": {
            "mean": float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "median": float(np.median(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "max": float(np.max(valid_scores)) if len(valid_scores) > 0 else 0.0,
        },
    }
    
    return final_score, pairs_df


GRN_METHODS = {
    'grn_target_overlap_score': {
        'func': target_overlap_score,
        'datasets': ['collectri']
    }
}