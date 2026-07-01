"""Protein-Protein Interaction (PPI) related indices.
"""

import networkx as nx
import numpy as np
from tfitpy.utils import generate_tf_pairs
import pandas as pd
from scipy.stats import hypergeom
from tfitpy.utils import ORGANISM_METADATA, INDICES_DATA
import igraph as ig

# -----
# Index 1 : PPI Shared Partners Index
# ------


def _get_ppi_partners(ppi_graph, node: str) -> set:
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


def shared_partners_pairwise(tf1, tf2, ppi_graph, background_size):
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


def shared_partners_score(sources, ppi_network=None, pairs=None):
    """Computes the shared PPI partners score for a TF regulatory module.

    For each TF pair derived from sources, computes the hypergeometric
    shared-partners score and aggregates all pairwise scores into a single
    module-level index using the mean.
    """
    if ppi_network is None:
        raise ValueError("No graph provided")

    if pairs is None:
        pairs = generate_tf_pairs(sources)

    background_size = len(ppi_network.nodes())
    scores = []

    for tf1, tf2 in pairs:
        score, p_value, common_partner_count = shared_partners_pairwise(
            tf1=tf1,
            tf2=tf2,
            ppi_graph=ppi_network,
            background_size=background_size,
        )
        scores.append(score)

    # Convert to numpy array to clean inf / nan efficiently
    scores_arr = np.array(scores, dtype=float)
    finite_mask = np.isfinite(scores_arr)
    cleaned_scores = scores_arr[finite_mask]

    if cleaned_scores.size == 0:
        return 0.0

    return float(cleaned_scores.mean())


# =========|
# Index 2
# Shortest path score
# =========|

def shortest_path_pairwise(tf1, tf2, ppi_graph):
    """Compute proximity score from shortest path length."""
    if tf1 not in ppi_graph or tf2 not in ppi_graph:
        return 0.0, float('inf')
    try:
        length = nx.shortest_path_length(ppi_graph, tf1, tf2)
        S = 1.0 / length if length > 0 else 1.0
        return S, length
    except nx.NetworkXNoPath:
        return 0.0, float('inf')

def shortest_path_score(sources, ppi_network=None):
    """Computes the aggregate shortest-path score for a TF regulatory module."""
    if ppi_network is None:
        raise ValueError("No graph provided")

    pairs = generate_tf_pairs(sources)

    scores = []

    for tf1, tf2 in pairs:
        proximity, path_length = shortest_path_pairwise(tf1, tf2, ppi_network)
        if np.isfinite(proximity):
            scores.append(proximity)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


# =========|
# Index 3
# PPI induced subgraph scores 
# =========|

def induced_network_metrics(sources,target, ppi, null_graphs, mapping):
    """
    sources: List of gene symbols
    target: target gene
    ppi: Original aligned igraph object
    null_graphs: List of  pre-built igraph objects
    mapping: Symbol -> Int dictionary
    """
    # 1. Map symbols to indices
    m_indices = [mapping[g] for g in sources if g in mapping]
    n_m = len(m_indices)
    target_idx = mapping.get(target)
        
    if n_m < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    possible_edges = (n_m * (n_m - 1)) / 2

    # --- 2. Observed Metrics ---
    sub_obs = ppi.induced_subgraph(m_indices)
    e_obs = sub_obs.ecount()
    
    obs_density = e_obs / possible_edges if possible_edges > 0 else 0.0
    obs_lcc_count = sub_obs.connected_components().giant().vcount() if e_obs > 0 else 1
    obs_lcc_ratio = obs_lcc_count / n_m

    obs_tc_count = 0
    if target_idx is not None:
        # Get neighbors of target and find intersection with module indices
        target_neighbors = set(ppi.neighbors(target_idx))
        obs_tc_count = len(target_neighbors.intersection(m_indices))
    
    obs_tc_ratio = obs_tc_count / n_m

    # --- 3. Null Distribution ---
    density_hits = 0
    lcc_hits = 0
    tc_hits = 0
    R = len(null_graphs)
    
    for i, g_null in enumerate(null_graphs):
        sub_null = g_null.induced_subgraph(m_indices)
        e_null = sub_null.ecount()
        
        # Null Density
        null_density = e_null / possible_edges if possible_edges > 0 else 0.0
        if null_density >= obs_density:
            density_hits += 1
            
        # Null LCC Ratio
        null_lcc_count = sub_null.connected_components().giant().vcount() if e_null > 0 else 1
        null_lcc_ratio = null_lcc_count / n_m
        
        if null_lcc_ratio >= obs_lcc_ratio:
            lcc_hits += 1

        if target_idx is not None:        
            null_target_neighbors = set(g_null.neighbors(target_idx))
            null_tc_count = len(null_target_neighbors.intersection(m_indices))
            if (null_tc_count / n_m) >= obs_tc_ratio:
                tc_hits += 1

    # --- 4. Final Scores (-ln(p)) ---
    p_density = (density_hits + 1) / (R + 1)
    density_score = max(0.0, -np.log(p_density))
    
    p_lcc = (lcc_hits + 1) / (R + 1)
    lcc_score = max(0.0, -np.log(p_lcc))

    if target_idx is not None:
        p_tc = (tc_hits + 1) / (R + 1)
        target_connectivity_score = max(-np.log(p_tc), 0.0)
    else:
        # target not in PPI: define as zero enrichment
        target_connectivity_score = 0.0

    return obs_density, density_score, obs_lcc_ratio, lcc_score, obs_tc_ratio, target_connectivity_score


def get_ppi_interactions(sources,target,G,evidence_type="ppi"):
    """
    returns dataframe of edges retrieved from the ppi network for the given source and target gene
    """
    gene_set = set(sources+[target])
    # Keep only nodes present in gene_set, then extract induced subgraph
    nodes_in_graph = gene_set & set(G.nodes())
    subgraph = G.subgraph(nodes_in_graph)
    # Convert subgraph edges to DataFrame
    rows = [{"node1": u, "node2": v, **attrs} for u, v, attrs in subgraph.edges(data=True)]
    df = pd.DataFrame(rows)
    df["evidence_type"]=evidence_type
    return df


# =========
#  optimized, main methods 
# =========

def _clean_value(v, ndigits=5):
    # Convert numpy scalar to Python scalar
    if isinstance(v, np.generic):
        v = v.item()
    # Only round numeric types
    if isinstance(v, (int, float)):
        return round(v, ndigits)
    return v

def _ppi_scores_from_cache(sources, pairs: list, cache: pd.DataFrame,ppi_keys,ppi_cache_indices) -> dict:
    """Extract PPI scores from precomputed cache and aggregate."""
    from tfitpy.utils import generate_tf_pairs

    if pairs is None:
        pairs = generate_tf_pairs(sources)

    # Convert to sorted tuples for index lookup
    pair_tuples = [tuple(sorted([g1, g2])) for g1, g2 in pairs]

    # Fast index-based lookup using .loc with list of tuples
    try:
        relevant_rows = cache.loc[pair_tuples]
    except KeyError:
        # Some pairs not in cache - filter to existing ones
        existing_pairs = [p for p in pair_tuples if p in cache.index]
        if not existing_pairs:
            results = {}
            for p in ppi_cache_indices:
                for k in ppi_keys:
                    results[f"{p}_{k}"] = 0.0
            return results
        relevant_rows = cache.loc[existing_pairs]

    # Aggregate scores (mean, ignoring inf/nan)
    results = {}

    for db_key in ppi_keys:
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



def PPI_SCORES(sources,target,dataset_cache,organism="human",use_pairwise_cache=True,pairs=None):
    """
    compute all PPI score for the given source module and target
    """
    ppi_keys = ORGANISM_METADATA[organism]["ppi_keys"]
    pair_cache_key = ORGANISM_METADATA[organism]["pair_cache"]

    ppi_cache_indices = INDICES_DATA["ppi_cached_indices"]
    
    results = {}
    source_pairs = generate_tf_pairs(sources) if pairs is None else pairs

    if use_pairwise_cache :
        if dataset_cache[pair_cache_key] is  None:
            raise ValueError("No cache data provided")
        pairwise_cache = dataset_cache[pair_cache_key]
        cache_results = _ppi_scores_from_cache(sources,source_pairs,pairwise_cache,ppi_keys,ppi_cache_indices)
        results = {**cache_results}
    else:
        # compute all needed ppi indices 
        for p in ppi_keys:
            g = dataset_cache[p]
            s1 = shared_partners_score(sources,g,source_pairs) 
            results[f"shared_PPI_partners_score_{p}"]= s1
            s2 = shortest_path_score(sources,g)
            results[f"shortest_PPI_path_score_{p}"]= s2
    
    evidence_edges = []
    for p in ppi_keys:
        g = dataset_cache[p]
        g_ig = ig.Graph.from_networkx(g)
        # network analysis 
        n,m = dataset_cache[f"{p}_null"]
        obs_density, density_score, obs_lcc_ratio, lcc_score, obs_tc_ratio, target_connectivity_score = induced_network_metrics(sources,target,g_ig,n,m)
        results[f"density_{p}"] = obs_density
        results[f"density_score_{p}"] = density_score
        results[f"lcc_{p}"] = obs_lcc_ratio
        results[f"lcc_score_{p}"] = lcc_score
        results[f"tc_{p}"] = obs_tc_ratio
        results[f"tc_score_{p}"] = target_connectivity_score
        # ppi evidence 
        edges = get_ppi_interactions(sources,target,g,f"{p}")
        evidence_edges.append(edges)

    edges = pd.concat(evidence_edges,ignore_index=True)
    results = {k: _clean_value(v) for k, v in results.items()}
    return results, edges


def _ppi_single_pass(
    pairs: list,
    graph: nx.Graph,
    background_size: int,
):
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

    # print(pairs)
    # print(graph)
    # print(list(graph.nodes)[:10])
    for tf1, tf2 in pairs:

        # --- shortest path ---
        if tf1 not in graph or tf2 not in graph:
            path_scores.append(0.0)
        else:
            try:
                length = nx.shortest_path_length(graph, tf1, tf2)
                #print(length)
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
            # print("none")
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

    # print(path_scores)
    # print(partner_scores)
    return _safe_mean(path_scores), _safe_mean(partner_scores)


_PPI_DB_KEYS = ["hippie", "stringdb", "biogrid"]
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
        # print("using fastcache")
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

        path_score, partner_score = _ppi_single_pass(
            pairs, graph, background_size)

        results[f"shortest_PPI_path_score_{db_key}"] = round(path_score, 5)
        results[f"shared_PPI_partners_score_{db_key}"] = round(
            partner_score, 5)

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
