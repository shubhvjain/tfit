"""
given gene expression data these methods help compute a cache of pairwise  distance correlation 

"""

import numpy as np
import pandas as pd
import dcor
from joblib import Parallel, delayed

def _compute_source_row(i, expr, n):
    xi = expr[:, i]
    vals = []

    for j in range(i + 1, n):
        xj = expr[:, j]
        dc = dcor.distance_correlation(xi, xj, method="mergesort")
        vals.append((i, j, dc))

    return vals


def generate_source_dcorr_cache(gene_expression, source_list, n_jobs=-1):
    sources = [g for g in source_list if g in gene_expression.columns]
    expr = np.ascontiguousarray(gene_expression[sources].to_numpy())

    n = len(sources)
    dmat = np.zeros((n, n), dtype=float)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_source_row)(i, expr, n)
        for i in range(n - 1)
    )

    for row_results in results:
        for i, j, dc in row_results:
            dmat[i, j] = dc
            dmat[j, i] = dc

    np.fill_diagonal(dmat, 1.0)

    source_matrix = pd.DataFrame(dmat, index=sources, columns=sources)
    return source_matrix


def _compute_target_dcorr(source_values, target_values):
    return dcor.distance_correlation(source_values, target_values, method="mergesort")


def generate_target_dcorr_cache(gene_expression, source_list, targets, n_jobs=-1):
    """
    Compute distance correlation for all sources vs one or multiple targets.

    targets can be:
      - a single column name (str), or
      - a list/tuple of column names.

    Returns:
      DataFrame: index = sources, columns = targets
    """
    # normalize targets to a list
    if isinstance(targets, (str, int)):
        targets = [targets]

    # keep only those targets present in the expression
    targets = [t for t in targets if t in gene_expression.columns]
    if not targets:
        raise ValueError("no valid targets in gene_expression")

    # filter sources
    sources = [g for g in source_list if g in gene_expression.columns]
    if not sources:
        raise ValueError("no valid sources in gene_expression")

    # expression matrix: samples × sources
    expr = np.ascontiguousarray(gene_expression[sources].to_numpy())

    # target matrix: samples × targets
    target_expr = np.ascontiguousarray(gene_expression[targets].to_numpy())

    # we will compute dcorr for every (source_idx, target_idx) pair
    tasks = [
        (i, j)
        for i in range(len(sources))
        for j in range(len(targets))
    ]

    def _task(idx_pair):
        i, j = idx_pair
        return (
            i,
            j,
            _compute_target_dcorr(expr[:, i], target_expr[:, j])
        )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_task)(pair) for pair in tasks
    )

    # fill result matrix
    dmat = np.zeros((len(sources), len(targets)), dtype=float)
    for i, j, dc in results:
        dmat[i, j] = dc

    target_matrix = pd.DataFrame(dmat, index=sources, columns=targets)
    return target_matrix

# def generate_source_dcorr_cache(gene_expression, source_list,target):
#     sources = [g for g in source_list if g in gene_expression.columns]
#     expr = gene_expression[sources].to_numpy()

#     n = len(sources)
#     dmat = np.zeros((n, n), dtype=float)

#     for i in range(n):
#         xi = expr[:, i]
#         for j in range(i + 1, n):
#             xj = expr[:, j]
#             dc = dcor.distance_correlation(xi, xj, method="mergesort")
#             dmat[i, j] = dc
#             dmat[j, i] = dc

#     np.fill_diagonal(dmat, 1.0)

#     source_matrix = pd.DataFrame(dmat, index=sources, columns=sources)
#     return source_matrix

# def generate_target_dcorr_cache(gene_expression, source_list, target):
#     sources = [g for g in source_list if g in gene_expression.columns]

#     if target not in gene_expression.columns:
#         raise ValueError("target not in gene_expression")

#     target_values = gene_expression[target].to_numpy()
#     values = []

#     for source in sources:
#         source_values = gene_expression[source].to_numpy()
#         dc = dcor.distance_correlation(source_values, target_values, method="mergesort")
#         values.append(dc)

#     target_matrix = pd.DataFrame(values, index=sources, columns=[target])
#     return target_matrix