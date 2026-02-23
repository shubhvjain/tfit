import networkx as nx
import numpy as np
from typing import Dict, List, Any, Union, Callable, Tuple
from tfitpy.utils import generate_tf_pairs
import pandas as pd
from scipy.stats import hypergeom




# =========|
# Index 2 |
# =========|

def shortest_path_proximity(tf1: str, tf2: str, ppi_graph: nx.Graph) -> Tuple[float, float]:
    """Compute proximity score from shortest path length."""
    if tf1 not in ppi_graph or tf2 not in ppi_graph:
        return 0.0, float('inf')
    try:
        length = nx.shortest_path_length(ppi_graph, tf1, tf2)
        proximity = 1.0 / length if length > 0 else 1.0
        return proximity, length
    except nx.NetworkXNoPath:
        return 0.0, float('inf')


def shortest_path_score(
        sources: list,
        target: str,
        ppi_network: nx.Graph = None,
        aggregation_method: str = "mean",
        pairs=None
) -> tuple:
    """
    For all pairs of TF in the given module, compute shortest path score between the 2 TFs.

    :param sources: List of TF gene symbols
    :param target: Target gene symbol
    :param ppi_network: Pre-loaded PPI network
    :param aggregation_method: "mean", "median", "max"
    :param pairs: Optional pre-computed TF pairs
    :return: tuple(final_score, pairs_df, metadata)
    """
    if ppi_network is None:
        raise ValueError("No graph provided")

    if pairs is None:
        pairs = generate_tf_pairs(sources)
    pair_results = []

    for tf1, tf2 in pairs:
        proximity, path_length = shortest_path_proximity(tf1, tf2, ppi_network)
        pair_results.append({
            'tf1': tf1,
            'tf2': tf2,
            'proximity_score': proximity,
            'path_length': path_length
        })

    pairs_df = pd.DataFrame(pair_results)

    valid_proximities = pairs_df['proximity_score'].replace(0, np.nan).dropna()
    methods = {
        "mean": np.mean(valid_proximities) if len(valid_proximities) > 0 else 0.0,
        "median": np.median(valid_proximities) if len(valid_proximities) > 0 else 0.0,
        "max": np.max(valid_proximities) if len(valid_proximities) > 0 else 0.0
    }

    final_score = methods.get(aggregation_method, np.mean(valid_proximities))

    metadata = {
        "input_module": sources,
        "total_pairs": len(pairs),
        "valid_pairs": len(valid_proximities),
        "aggregation_method": aggregation_method,
        "all_aggregations": {k: float(v) for k, v in methods.items()}
    }

    return final_score, pairs_df, metadata


def _shortest_path_wrapper(db_key: str, sources, target, aggregation_method, datasets, pairs):
    """Shared logic for all per-database shortest-path wrappers."""
    if datasets is None:
        raise ValueError("datasets cache is required. Create cache with load_datasets() first.")
    if db_key not in datasets:
        raise ValueError(f"Dataset dependency missing: '{db_key}'")
    return shortest_path_score(
        sources=sources,
        target=target,
        aggregation_method=aggregation_method,
        ppi_network=datasets[db_key],
        pairs=pairs,
    )

def shortest_path_score_hippie(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Shortest path proximity score using the HIPPIE PPI network."""
    return _shortest_path_wrapper("hippie", sources, target, aggregation_method, datasets, pairs)


def shortest_path_score_stringdb(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Shortest path proximity score using the STRING PPI network."""
    return _shortest_path_wrapper("stringdb", sources, target, aggregation_method, datasets, pairs)


def shortest_path_score_biogrid(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Shortest path proximity score using the BioGRID PPI network."""
    return _shortest_path_wrapper("biogrid", sources, target, aggregation_method, datasets, pairs)




# =========|
# Index 1 |
# =========|



def _hypergeom_wrapper(db_key: str, sources, target, aggregation_method, datasets, pairs):
    """Shared logic for all per-database hypergeometric wrappers."""
    if datasets is None:
        raise ValueError("datasets cache is required. Create cache with load_datasets() first.")
    if db_key not in datasets:
        raise ValueError(f"Dataset dependency missing: '{db_key}'")
    return hypergeom_index_score(
        sources=sources,
        target=target,
        aggregation_method=aggregation_method,
        ppi_network=datasets[db_key],
        pairs=pairs,
    )



def get_ppi_partners(ppi_graph: nx.Graph, node: str) -> set:
    """
    Return the set of direct PPI partners (neighbors) of a node in the PPI graph.
    """
    if node not in ppi_graph:
        return set()
    # neighbors() returns an iterator of adjacent nodes
    return set(ppi_graph.neighbors(node))


def hypergeom_overlap_pvalue(N: int, N1: int, N2: int, c: int) -> float:
    """
    Compute upper-tail hypergeometric P-value:
        P = P(X >= c), X ~ Hypergeom(N, N1, N2)
    """
    # hypergeom.sf(k-1, M, K, n) = P(X >= k)
    # M=N (population size), K=N1 (number of "success" states),
    # n=N2 (number of draws)
    if c > min(N1, N2):
        return 1.0
    return float(hypergeom.sf(c - 1, N, N1, N2))

def hypergeom_score_for_pair(tf1: str, tf2: str,
                             ppi_graph: nx.Graph,
                             background_size: int) -> Tuple[float, Dict[str, Any]]:
    """
    Compute hypergeometric distribution score S for a TF pair.

    Returns:
        S (float): -log10(P) where P is hypergeometric P-value
        info (dict): detailed counts and P for debugging / downstream use
    """
    partners1 = get_ppi_partners(ppi_graph, tf1)
    partners2 = get_ppi_partners(ppi_graph, tf2)

    N1 = len(partners1)
    N2 = len(partners2)
    if N1 == 0 or N2 == 0:
        return 0.0, {
            "tf1": tf1,
            "tf2": tf2,
            "N1": N1,
            "N2": N2,
            "c": 0,
            "pvalue": 1.0,
        }

    common_partners = partners1 & partners2
    c = len(common_partners)

    if c == 0:
        # No overlap -> non-significant
        return 0.0, {
            "tf1": tf1,
            "tf2": tf2,
            "N1": N1,
            "N2": N2,
            "c": 0,
            "pvalue": 1.0,
        }

    P = hypergeom_overlap_pvalue(background_size, N1, N2, c)

    # Avoid log10(0)
    if P <= 0.0:
        S = float("inf")
    else:
        S = -np.log10(P)

    info = {
        "tf1": tf1,
        "tf2": tf2,
        "N1": N1,
        "N2": N2,
        "c": c,
        "pvalue": P,
        "score": S,
    }
    return float(S), info


def hypergeom_index_score(
    sources:list,
    target:list,
    ppi_network: nx.Graph = None,
    aggregation_method: str = "mean",
    background_size: int = None,
    pairs=None
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    TF-based performance index 1 (hypergeometric distribution score).

    For each TF pair, compute -log10(P) where P is the hypergeometric
    P-value of the overlap of their PPI partners, then aggregate over
    all pairs in the module.

    Returns:
        final_score, pairs_df, metadata
    """
    if ppi_network is None:
        raise ValueError("No graph provided")

    if pairs is None:
        pairs = generate_tf_pairs(sources)
    
    pair_results = []

    if background_size is None:
        background_size = len(ppi_network.nodes())

    for tf1, tf2 in pairs:
        score, info = hypergeom_score_for_pair(
            tf1=tf1,
            tf2=tf2,
            ppi_graph=ppi_network,
            background_size=background_size,
        )
        row = {
            "tf1": tf1,
            "tf2": tf2,
            "hypergeom_score": score,
            "N1": info["N1"],
            "N2": info["N2"],
            "c": info["c"],
            "pvalue": info["pvalue"],
        }
        pair_results.append(row)

    pairs_df = pd.DataFrame(pair_results)

    valid_scores = pairs_df["hypergeom_score"].replace(
        [np.inf, -np.inf], np.nan)
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

    return final_score, pairs_df, metadata


def hypergeometic_index_score_hippie(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Hypergeometric overlap score using the HIPPIE PPI network."""
    return _hypergeom_wrapper("hippie", sources, target, aggregation_method, datasets, pairs)


def hypergeometic_index_score_stringdb(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Hypergeometric overlap score using the STRING PPI network."""
    return _hypergeom_wrapper("stringdb", sources, target, aggregation_method, datasets, pairs)


def hypergeometic_index_score_biogrid(
        sources: list,
        target: str,
        aggregation_method: str = "mean",
        datasets=None,
        pairs=None,
        **args
):
    """Hypergeometric overlap score using the BioGRID PPI network."""
    return _hypergeom_wrapper("biogrid", sources, target, aggregation_method, datasets, pairs)




PPI_METHODS = {
    'shortest_path_score_hippie': {
        'func': shortest_path_score_hippie,
        'datasets': ['hippie']
    },

    'shortest_path_score_stringdb': {
        'func': shortest_path_score_stringdb,
        'datasets': ['stringdb'],
    },
    'shortest_path_score_biogrid': {
        'func': shortest_path_score_biogrid,
        'datasets': ['biogrid'],
    },

    'hypergeometric_index_score_hippie': {
        'func': hypergeometic_index_score_hippie,
        'datasets': ['hippie']
    },
    
    'hypergeometric_index_score_stringdb': {
        'func': hypergeometic_index_score_stringdb,
        'datasets': ['stringdb'],
    },
    'hypergeometric_index_score_biogrid': {
        'func': hypergeometic_index_score_biogrid,
        'datasets': ['biogrid'],
    },
}
