"""Gene Ontology (GO) functional similarity indices.

The core methods accept a GODag and gene2go dict as inputs, enabling reuse
across any organism or annotation source. One source is currently supported:

- Gene Ontology Annotation (GOA) human (:cite:t:`go_2023`)

Three semantic similarity methods are implemented using Best-Match Average (BMA):

- Lin similarity 
- Resnik similarity 
- Jiang-Conrath similarity 
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable
from goatools.obo_parser import GODag
from goatools.semantic import (
    TermCounts, lin_sim, resnik_sim,
    get_info_content, deepest_common_ancestor
)

from tfitpy.utils import generate_tf_pairs


def _jc_sim(term1: str, term2: str, godag: GODag, termcounts: TermCounts) -> Optional[float]:
    """Jiang-Conrath similarity between two GO terms.

    JC distance = IC(t1) + IC(t2) - 2 * IC(MICA)
    JC sim      = 1 / (1 + JC_distance)  → bounded in (0, 1]
    """
    if term1 not in godag or term2 not in godag:
        return None
    ic1 = get_info_content(term1, termcounts)
    ic2 = get_info_content(term2, termcounts)
    if ic1 is None or ic2 is None:
        return None
    mica = deepest_common_ancestor([term1, term2], godag)
    if mica is None:
        return None
    ic_mica = get_info_content(mica, termcounts)
    if ic_mica is None:
        return None
    return 1.0 / (1.0 + ic1 + ic2 - 2.0 * ic_mica)


_SIM_FUNCS = {
    "lin": lin_sim,
    "resnik": resnik_sim,
    "jc": _jc_sim,
}


def _gene_sim_bma(
    gene1: str,
    gene2: str,
    gene2go: dict,
    godag: GODag,
    termcounts: TermCounts,
    sim_func: Callable,
) -> float:
    terms1 = list(gene2go.get(gene1, set()))
    terms2 = list(gene2go.get(gene2, set()))

    if not terms1 or not terms2:
        return 0.0

    best_matches = []

    for t1 in terms1:
        row = [s for t2 in terms2
               if t1 in godag and t2 in godag
               and godag[t1].namespace == godag[t2].namespace
               if (s := sim_func(t1, t2, godag, termcounts)) is not None
               and not np.isnan(s)]
        if row:
            best_matches.append(max(row))

    for t2 in terms2:
        col = [s for t1 in terms1
               if t1 in godag and t2 in godag
               and godag[t1].namespace == godag[t2].namespace
               if (s := sim_func(t2, t1, godag, termcounts)) is not None
               and not np.isnan(s)]
        if col:
            best_matches.append(max(col))

    return float(np.mean(best_matches)) if best_matches else 0.0


def _gene_sim_bma_with_terms(
    terms1: list,
    terms2: list,
    godag: GODag,
    termcounts: TermCounts,
    sim_func: Callable,
) -> float:
    """BMA similarity using pre-fetched term lists.

    Same logic as _gene_sim_bma but accepts terms directly instead of
    looking them up from gene2go — allows the caller to cache term lookups
    across multiple sim_func calls for the same gene pair.

    Args:
        terms1: GO term IDs for gene1 (already fetched from gene2go).
        terms2: GO term IDs for gene2 (already fetched from gene2go).
        godag: Loaded GODag object.
        termcounts: Pre-computed TermCounts object.
        sim_func: Term-level similarity function (lin_sim, resnik_sim, _jc_sim).

    Returns:
        BMA similarity score as float. 0.0 if no valid scores exist.
    """
    if not terms1 or not terms2:
        return 0.0

    best_matches = []

    for t1 in terms1:
        row = [s for t2 in terms2
               if t1 in godag and t2 in godag
               and godag[t1].namespace == godag[t2].namespace
               if (s := sim_func(t1, t2, godag, termcounts)) is not None
               and not np.isnan(s)]
        if row:
            best_matches.append(max(row))

    for t2 in terms2:
        col = [s for t1 in terms1
               if t1 in godag and t2 in godag
               and godag[t1].namespace == godag[t2].namespace
               if (s := sim_func(t2, t1, godag, termcounts)) is not None
               and not np.isnan(s)]
        if col:
            best_matches.append(max(col))

    return float(np.mean(best_matches)) if best_matches else 0.0


def similarity_score_pairwise(
    gene1: str,
    gene2: str,
    method: str,
    godag: GODag,
    gene2go: dict,
    termcounts: TermCounts,
) -> float:
    """Compute GO semantic similarity for a single gene pair.

    Args:
        gene1: First gene identifier (HGNC symbol).
        gene2: Second gene identifier (HGNC symbol).
        method: Similarity method — one of 'lin', 'resnik', 'jc'.
        godag: Loaded GODag object.
        gene2go: Mapping of gene symbol → set of GO term IDs.
        termcounts: Pre-computed TermCounts object for IC calculation.

    Returns:
        BMA similarity score as a float in [0, 1]. 0.0 if either gene
        has no annotations or no valid term-level scores exist.

    Raises:
        ValueError: If method is not one of 'lin', 'resnik', 'jc'.
    """
    if method not in _SIM_FUNCS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(_SIM_FUNCS)}")
    return _gene_sim_bma(gene1, gene2, gene2go, godag, termcounts, _SIM_FUNCS[method])


def similarity_score(
    sources: list,
    method: str,
    godag: GODag = None,
    gene2go: dict = None,
    termcounts: TermCounts = None,
    pairs=None,
) -> Tuple[float, pd.DataFrame]:
    """Compute GO semantic similarity for a gene module and aggregate by mean.

    For each pair derived from sources, computes BMA similarity using the
    specified method and aggregates into a single module-level score.

    Args:
        sources: List of gene identifiers in the regulatory module.
        method: Similarity method — one of 'lin', 'resnik', 'jc'.
        godag: Loaded GODag object. Must be provided.
        gene2go: Mapping of gene symbol → set of GO term IDs. Must be provided.
        termcounts: Pre-computed TermCounts. Computed from gene2go if None.
        pairs: Optional precomputed list of (g1, g2) tuples. If None, all
            unique pairs are generated from sources via generate_tf_pairs().

    Returns:
        A tuple (final_score, pairs_df) where:
            final_score (float): Mean similarity across all pairs. 0.0 if none.
            pairs_df (DataFrame): One row per pair with columns:
                tf1, tf2, score, n_terms_tf1, n_terms_tf2.

    Raises:
        ValueError: If godag or gene2go is None, or method is invalid.
    """
    if godag is None or gene2go is None:
        raise ValueError("godag and gene2go are required.")
    if method not in _SIM_FUNCS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(_SIM_FUNCS)}")
    if termcounts is None:
        termcounts = TermCounts(godag, gene2go)
    if pairs is None:
        pairs = generate_tf_pairs(sources)

    pair_results = []
    for g1, g2 in pairs:
        score = similarity_score_pairwise(
            g1, g2, method, godag, gene2go, termcounts)
        pair_results.append({
            "tf1": g1,
            "tf2": g2,
            "score": score,
            "n_terms_tf1": len(gene2go.get(g1, set())),
            "n_terms_tf2": len(gene2go.get(g2, set())),
        })

    pairs_df = pd.DataFrame(pair_results)
    valid_scores = pairs_df["score"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    final_score = float(np.mean(valid_scores)) if len(
        valid_scores) > 0 else 0.0
    return final_score, pairs_df


def _similarity_score_all(
    sources: list,
    godag: GODag,
    gene2go: dict,
    termcounts: TermCounts,
    pairs=None,
) -> dict:
    """Compute all three GO similarity methods and return as a flat dict."""
    results = {}
    for method in ("lin", "resnik", "jc"):
        score, _ = similarity_score(
            sources=sources,
            method=method,
            godag=godag,
            gene2go=gene2go,
            termcounts=termcounts,
            pairs=pairs,
        )
        results[f"goa_similarity_{method}"] = score
    return results


# =========|
# Optimized: single-pass all GO scores
# =========|

def _go_scores_from_cache(sources: list, pairs: list, cache: pd.DataFrame) -> dict:
    """Extract GO scores from precomputed cache and aggregate."""
    
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
            "goa_similarity_lin": 0.0,
            "goa_similarity_resnik": 0.0,
            "goa_similarity_jc": 0.0,
        }
    
    # Aggregate scores (mean, ignoring inf/nan)
    results = {}
    
    for method in ["lin", "resnik", "jc"]:
        col = f'goa_similarity_{method}'
        
        # Get valid scores
        scores = relevant_rows[col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        
        # Compute mean
        results[col] = round(
            float(scores.mean()) if len(scores) > 0 else 0.0,
            5
        )
    
    return results

def go_all_scores(
    sources: list,
    datasets: dict = None,
    pairs: list = None,
    **kwargs,
) -> dict:
    """Compute all 3 GO similarity scores in a single pass over pairs.

    For each TF pair, fetches GO term lists once and computes lin, resnik,
    and jc similarity in sequence — avoiding 3 separate pair loops and
    repeated gene2go lookups. Uses a row-level terms cache so each gene's
    GO terms are fetched only once regardless of how many pairs it appears in.

    TermCounts is built once per call rather than once per method.

    Args:
        sources: List of gene identifiers in the regulatory module.
        datasets: Dataset cache dict containing 'go' with keys:
                  'godag', 'gene2go'. Must be provided.
        pairs: Optional precomputed list of (g1, g2) tuples. If None,
               generated from sources via generate_tf_pairs().

    Returns:
        Dict with 3 keys:
            goa_similarity_lin
            goa_similarity_resnik
            goa_similarity_jc

    Raises:
        ValueError: If datasets is None or 'go' key is missing.
    """

        # Check if we have the cache
    if datasets is not None and 'pairwise_score_cache' in datasets:
        # Fast path: use cache
        cache = datasets['pairwise_score_cache']
        #print("using fastcache for GO")
        return _go_scores_from_cache(sources, pairs, cache)
        
    if datasets is None:
        raise ValueError(
            "datasets cache is required. Create cache with load_datasets() first.")
    if "go" not in datasets:
        raise ValueError("Dataset dependency missing: 'go'")

    godag = datasets["go"]["godag"]
    gene2go = datasets["go"]["gene2go"]

    # TermCounts built once per row call, not once per method
    termcounts = TermCounts(godag, gene2go)

    if pairs is None:
        pairs = generate_tf_pairs(sources)

    # Row-level terms cache: gene2go.get(gene) fetched once per gene,
    # reused across all pairs and all 3 methods that gene appears in.
    terms_cache: dict = {}

    lin_scores = []
    resnik_scores = []
    jc_scores = []

    for gene1, gene2 in pairs:
        if gene1 not in terms_cache:
            terms_cache[gene1] = list(gene2go.get(gene1, set()))
        if gene2 not in terms_cache:
            terms_cache[gene2] = list(gene2go.get(gene2, set()))

        terms1 = terms_cache[gene1]
        terms2 = terms_cache[gene2]

        # Each method gets pre-fetched terms — no redundant gene2go lookups
        lin_scores.append(
            _gene_sim_bma_with_terms(terms1, terms2, godag, termcounts, lin_sim))
        resnik_scores.append(
            _gene_sim_bma_with_terms(terms1, terms2, godag, termcounts, resnik_sim))
        jc_scores.append(
            _gene_sim_bma_with_terms(terms1, terms2, godag, termcounts, _jc_sim))

    def _safe_mean(values: list) -> float:
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    return {
        "goa_similarity_lin":    round(_safe_mean(lin_scores), 5),
        "goa_similarity_resnik": round(_safe_mean(resnik_scores), 5),
        "goa_similarity_jc":     round(_safe_mean(jc_scores), 5),
    }


GO_METHODS = {
    "goa_similarity": {
        "func": go_all_scores,
        "type": "df_columns",
        "cols": ["goa_similarity_lin", "goa_similarity_resnik", "goa_similarity_jc"],
        "datasets": ["pairwise_score_cache"],
    },
}