"""Protein-Protein Interaction (PPI) related indices.

The core methods accept a NetworkX graph as the PPI input, which enables the
use of multiple PPI network sources. Three sources are currently supported:

- HIPPIE PPI (:cite:t:`hippie`)
- StringDB PPI (:cite:t:`string`)
- BioGRID Database (:cite:t:`biogrid`)

Three wrapper functions are provided that accept a shared dataset cache.
The cache is described in the load_datasets() documentation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Union, Callable, Tuple
from tfitpy.utils import generate_tf_pairs
import pandas as pd
from scipy.stats import hypergeom

# -----
# Index 1 : PPI Shared Partners Index
# ------


def _get_ppi_partners(ppi_graph: nx.Graph, node: str) -> set:
    """Returns the set of direct PPI partners (neighbors) of a node."""
    if node not in ppi_graph:
        return set()
    return set(ppi_graph.neighbors(node))


def _hypergeometric_pvalue(N: int, N1: int, N2: int, c: int) -> float:
    """Computes the upper-tail hypergeometric p-value for partner overlap.

    Calculates P(X >= c) where X ~ Hypergeometric(N, N1, N2).

    Args:
        N: Total population size (number of nodes in the PPI network).
        N1: Number of interaction partners for the first TF.
        N2: Number of interaction partners for the second TF.
        c: Observed number of shared partners.

    Returns:
        The upper-tail p-value as a float in the range [0, 1]. Returns 1.0
        if c exceeds the maximum possible overlap (min(N1, N2)).
    """
    if c > min(N1, N2):
        return 1.0
    return float(hypergeom.sf(c - 1, N, N1, N2))


def shared_partners_pairwise(tf1: str, tf2: str,
                             ppi_graph: nx.Graph,
                             background_size: int) -> Tuple[float, float, int]:
    """ Computes the hypergeometric shared-partners score for a single TF pair.

    For a given pair of transcription factors, retrieves their respective PPI
    partner sets, computes the overlap, and returns a significance score
    S = -log10(P) where P is the upper-tail hypergeometric p-value. Based on :cite:t:`indices_2014`.

    Args:
        tf1: identifier of TF1.
        tf2: identifier of TF2.
        ppi_graph: An undirected NetworkX graph representing the PPI network.
        background_size: Total number of proteins used as the population size for the hypergeometric test. Typically the number of nodes in the PPI graph. 

    Returns:
        A tuple (S,p,c) where:
            S (float): The significance score -log10(P). Returns 0.0 if either TF has no partners or if there is no overlap. Returns inf if P rounds to zero.
            p (float): The p-value 
            c (int): The number of common partners

    """
    partners1 = _get_ppi_partners(ppi_graph, tf1)
    partners2 = _get_ppi_partners(ppi_graph, tf2)

    N1 = len(partners1)
    N2 = len(partners2)
    common_partners = partners1 & partners2
    c = len(common_partners)

    if N1 == 0 or N2 == 0 or c == 0:
        return 0.0, 1.0, 0

    P = _hypergeometric_pvalue(background_size, N1, N2, c)
    S = float("inf") if P <= 0.0 else -np.log10(P)
    return float(S), P, c


def shared_partners(
    sources: list,
    ppi_network: nx.Graph = None,
    pairs=None
) -> Tuple[float, pd.DataFrame]:
    """Computes the shared PPI partners score for a TF regulatory module.

    For each TF pair derived from sources, computes the hypergeometric shared-partners score and aggregates all pairwise scores into a single module-level index using by taking the mean. Based on :cite:t:`indices_2014`.

    Args:
        sources: List of source TF identifiers in the regulatory module. 
        ppi_network: An undirected NetworkX graph representing the PPI
            network. Must be provided.

    Returns:
        A tuple (final_score, pairs_df) where:
            final_score (float): The mean hypergeometric score across all valid
                TF pairs. Returns 0.0 if no valid scores exist.
            pairs_df (DataFrame): A DataFrame with one row per TF pair,
                containing columns: tf1, tf2, score, p_value , common_partners

    Raises:
        ValueError: If ppi_network is None.
    """
    if ppi_network is None:
        raise ValueError("No graph provided")

    if pairs is None:
        pairs = generate_tf_pairs(sources)

    pair_results = []
    background_size = len(ppi_network.nodes())

    for tf1, tf2 in pairs:
        score, p_value, common_partner_count = shared_partners_pairwise(
            tf1=tf1,
            tf2=tf2,
            ppi_graph=ppi_network,
            background_size=background_size,
        )
        row = {
            "tf1": tf1,
            "tf2": tf2,
            "score": score,
            "p_value": p_value,
            "common_partners": common_partner_count
        }
        pair_results.append(row)

    pairs_df = pd.DataFrame(pair_results)

    valid_scores = pairs_df["score"].replace(
        [np.inf, -np.inf], np.nan)
    valid_scores = valid_scores.dropna()

    if len(valid_scores) == 0:
        final_score = 0.0
    else:
        # aggregate via mean for the whole group
        final_score = float(np.mean(valid_scores))

    return final_score, pairs_df


def _hypergeom_wrapper(db_key, sources, datasets, pairs=None):
    """Shared logic for hypergeometric wrappers."""
    if datasets is None:
        raise ValueError(
            "datasets cache is required. Create cache with load_datasets() first.")
    if db_key not in datasets:
        raise ValueError(f"Dataset dependency missing: '{db_key}'")
    return shared_partners(
        sources=sources,
        ppi_network=datasets[db_key],
        pairs=pairs
    )


def shared_partners_hippie(
        sources: list,
        datasets=None,
        pairs=None,
        **kwargs
):
    """PPI Shared Partner score using the HIPPIE PPI network."""
    return _hypergeom_wrapper("hippie", sources, datasets, pairs)


def shared_partners_stringdb(
        sources: list,
        datasets=None,
        pairs=None,
        **kwargs
):
    """PPI Shared Partner score using the STRING PPI network."""
    return _hypergeom_wrapper("stringdb", sources, datasets, pairs)


def shared_partners_biogrid(
        sources: list,
        datasets=None,
        pairs=None,
        **kwargs
):
    """PPI Shared Partner score using the BioGRID PPI network."""
    return _hypergeom_wrapper("biogrid", sources, datasets, pairs)


# =========|
# Index 2
# Shortest path score
# =========|

def shortest_path_pairwise(tf1: str, tf2: str, ppi_graph: nx.Graph) -> Tuple[float, float]:
    """Compute proximity score from shortest path length."""
    if tf1 not in ppi_graph or tf2 not in ppi_graph:
        return 0.0, float('inf')
    try:
        length = nx.shortest_path_length(ppi_graph, tf1, tf2)
        S = 1.0 / length if length > 0 else 1.0
        return S, length
    except nx.NetworkXNoPath:
        return 0.0, float('inf')


def shortest_path_score(
        sources: list,
        ppi_network: nx.Graph = None,
        pairs=None
) -> tuple:
    """Computes the aggregate shortest-path score for a TF regulatory module.

    For each TF pair derived from sources, computes the shortest-path  score and aggregates all pairwise scores into a single module-level index by taking the mean.  Based on :cite:t:`indices_2014`

    Args:
        sources: List of source TF identifiers in the regulatory module.
        ppi_network: An undirected NetworkX graph representing the PPI
            network. Must be provided.
        pairs: Optional precomputed list of (tf1, tf2) tuples. If None, all
            unique pairs are generated from sources via generate_tf_pairs().

    Returns:
        A tuple (final_score, pairs_df) where:
            final_score (float): The mean proximity score across all TF pairs. Returns 0.0 if no valid scores exist.
            pairs_df (DataFrame): A DataFrame with one row per TF pair, containing columns: tf1, tf2, score, path_length.

    Raises:
        ValueError: If ppi_network is None.
    """
    if ppi_network is None:
        raise ValueError("No graph provided")

    if pairs is None:
        pairs = generate_tf_pairs(sources)

    pair_results = []

    for tf1, tf2 in pairs:
        proximity, path_length = shortest_path_pairwise(tf1, tf2, ppi_network)
        pair_results.append({
            'tf1': tf1,
            'tf2': tf2,
            'score': proximity,
            'path_length': path_length
        })

    pairs_df = pd.DataFrame(pair_results)
    valid_scores = pairs_df['score'].replace(
        [np.inf, -np.inf], np.nan).dropna()
    final_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
    return final_score, pairs_df


# =========|
# Optimized: single-pass all PPI scores
# =========|
 
_PPI_DB_KEYS = ["hippie", "stringdb", "biogrid"]
 
def _ppi_single_pass(
    pairs: list,
    graph: nx.Graph,
    background_size: int,
) -> Tuple[float, float]:
    """Single pass over pairs computing both shortest-path and shared-partners scores.
 
    For each pair, computes:
      - shortest path proximity score: 1/length (0.0 if no path)
      - shared partners hypergeometric score: -log10(p) (0.0 if no overlap)
 
    Uses a row-level neighbor cache so each TF's neighbor set is fetched
    only once per call, regardless of how many pairs it appears in.
 
    Args:
        pairs: List of (tf1, tf2) tuples.
        graph: NetworkX PPI graph for this database.
        background_size: Number of nodes in graph, used as hypergeometric N.
 
    Returns:
        Tuple of (shortest_path_score, shared_partners_score) — both are
        means across valid pairs, 0.0 if no valid scores exist.
    """
    # Row-level neighbor cache: avoids recomputing set(G.neighbors(tf))
    # for TFs that appear in multiple pairs within this row.
    neighbor_cache: dict = {}
 
    path_scores = []
    partner_scores = []
 
    for tf1, tf2 in pairs:
 
        # --- shortest path ---
        if tf1 not in graph or tf2 not in graph:
            path_scores.append(0.0)
        else:
            try:
                length = nx.shortest_path_length(graph, tf1, tf2)
                path_scores.append(1.0 / length if length > 0 else 1.0)
            except nx.NetworkXNoPath:
                path_scores.append(0.0)
 
        # --- shared partners (neighbor sets cached at row level) ---
        if tf1 not in neighbor_cache:
            neighbor_cache[tf1] = (
                set(graph.neighbors(tf1)) if tf1 in graph else set()
            )
        if tf2 not in neighbor_cache:
            neighbor_cache[tf2] = (
                set(graph.neighbors(tf2)) if tf2 in graph else set()
            )
 
        n1 = neighbor_cache[tf1]
        n2 = neighbor_cache[tf2]
        N1, N2 = len(n1), len(n2)
        c = len(n1 & n2)
 
        if N1 == 0 or N2 == 0 or c == 0:
            partner_scores.append(0.0)
        else:
            P = _hypergeometric_pvalue(background_size, N1, N2, c)
            S = float("inf") if P <= 0.0 else -np.log10(P)
            partner_scores.append(S)
 
    # Aggregate: mean over valid (finite, non-nan) scores
    def _safe_mean(values: list) -> float:
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) > 0 else 0.0
 
    return _safe_mean(path_scores), _safe_mean(partner_scores)
 

def _ppi_scores_from_cache(sources: list, pairs: list, cache: pd.DataFrame) -> dict:
    """Extract PPI scores from precomputed cache and aggregate."""
    from tfitpy.utils import generate_tf_pairs
    
    if pairs is None:
        pairs = generate_tf_pairs(sources)
    
    # Create a set for faster lookup
    pair_set = {tuple(sorted([g1, g2])) for g1, g2 in pairs}
    
    # Filter cache to matching pairs
    # Cache has pairs in alphabetical order (gene1, gene2)
    mask = cache.apply(
        lambda row: (row['gene1'], row['gene2']) in pair_set,
        axis=1
    )
    relevant_rows = cache[mask]
    
    if len(relevant_rows) == 0:
        # No matching pairs found - return zeros
        return {
            "shortest_PPI_path_score_hippie": 0.0,
            "shortest_PPI_path_score_stringdb": 0.0,
            "shortest_PPI_path_score_biogrid": 0.0,
            "shared_PPI_partners_score_hippie": 0.0,
            "shared_PPI_partners_score_stringdb": 0.0,
            "shared_PPI_partners_score_biogrid": 0.0,
        }
    
    # Aggregate scores (mean, ignoring inf/nan)
    results = {}
    
    for db_key in ["hippie", "stringdb", "biogrid"]:
        path_col = f'shortest_PPI_path_score_{db_key}'
        partner_col = f'shared_PPI_partners_score_{db_key}'
        
        # Get valid scores
        path_scores = relevant_rows[path_col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        partner_scores = relevant_rows[partner_col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        
        # Compute means
        results[path_col] = round(
            float(path_scores.mean()) if len(path_scores) > 0 else 0.0,
            5
        )
        results[partner_col] = round(
            float(partner_scores.mean()) if len(partner_scores) > 0 else 0.0,
            5
        )
    
    return results
 
def ppi_all_scores(
    sources: list,
    datasets: dict = None,
    pairs: list = None,
    **kwargs,
) -> dict:
    """Compute all 6 PPI scores in a single pass per database.
 
    For each of the three PPI databases (hippie, stringdb, biogrid), makes
    one pass over all TF pairs to compute both the shortest-path proximity
    score and the shared-partners hypergeometric score simultaneously.
 
    This is equivalent to calling shortest_path_score and shared_partners
    separately for each database, but with half the graph traversals.
 
    Args:
        sources: List of source TF identifiers in the regulatory module.
        datasets: Dataset cache dict containing 'hippie', 'stringdb', 'biogrid'
                  NetworkX graphs. Must be provided.
        pairs: Optional precomputed list of (tf1, tf2) tuples. If None,
               generated from sources via generate_tf_pairs().
 
    Returns:
        Dict with 6 keys:
            shortest_PPI_path_score_hippie
            shortest_PPI_path_score_stringdb
            shortest_PPI_path_score_biogrid
            shared_PPI_partners_score_hippie
            shared_PPI_partners_score_stringdb
            shared_PPI_partners_score_biogrid
 
    Raises:
        ValueError: If datasets is None or any required db key is missing.
    """
        # Check if we have the cache
    if datasets is not None and 'pairwise_score_cache' in datasets:
        # Fast path: use cache
        cache = datasets['pairwise_score_cache']
        #print("using fastcache")
        return _ppi_scores_from_cache(sources, pairs, cache)
    
    if datasets is None:
        raise ValueError(
            "datasets cache is required. Create cache with load_datasets() first.")
 
    missing = [k for k in _PPI_DB_KEYS if k not in datasets]
    if missing:
        raise ValueError(f"Dataset dependencies missing: {missing}")
 
    if pairs is None:
        pairs = generate_tf_pairs(sources)
 
    results = {}
 
    for db_key in _PPI_DB_KEYS:
        graph = datasets[db_key]
        background_size = len(graph.nodes())
 
        path_score, partner_score = _ppi_single_pass(pairs, graph, background_size)
 
        results[f"shortest_PPI_path_score_{db_key}"] = round(path_score, 5)
        results[f"shared_PPI_partners_score_{db_key}"] = round(partner_score, 5)
 
    return results
 
 
PPI_METHODS = {
    "ppi": {
        "func": ppi_all_scores,
        "datasets": ["pairwise_score_cache"],
        "type": "df_columns",
        "cols": [
            "shortest_PPI_path_score_hippie",
            "shortest_PPI_path_score_stringdb",
            "shortest_PPI_path_score_biogrid",
            "shared_PPI_partners_score_hippie",
            "shared_PPI_partners_score_stringdb",
            "shared_PPI_partners_score_biogrid",
        ],
    },
}