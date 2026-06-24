import numpy as np
import pandas as pd
import dcor
from joblib import Parallel, delayed
import os
import json
from tfitpy.utils import ORGANISM_METADATA, INDICES_DATA
from tfitpy.datasets.cache_dcorr import generate_source_dcorr_cache,generate_target_dcorr_cache


# def DCORR_SCORES(sources, target, dataset_cache, organism="human", use_pair_cache=False, n_permutations=500):
#     """
#     Compute observed source-source and source-target dCor means,
#     plus permutation-based scores, using cached distance correlation matrices.
#     """

#     gene_expression = dataset_cache["gene_expression"]
#     source_list = dataset_cache["source_list"]

#     source_cache = dataset_cache["dcorr_source_cache"]
#     target_cache = dataset_cache["dcorr_target_cache"]

#     if source_cache is None: 
#         source_cache = generate_source_dcorr_cache(
#             gene_expression, source_list
#         )
    
#     if target_cache is None: 
#         target_cache = generate_target_dcorr_cache(
#             gene_expression, source_list, target
#         )

#     sources = [g for g in sources if g in source_cache.index]
#     n = len(sources)

#     if n < 2 or target not in target_cache.columns:
#         return {
#             "dCor_sources_obs": 0.0,
#             "dCor_target_obs": 0.0,
#             "dCor_sources_score": 0.0,
#             "dCor_target_score": 0.0
#         }, None

#     sub_source = source_cache.loc[sources, sources]
#     sub_target = target_cache.loc[sources, target]

#     obs_sources = sub_source.values[np.triu_indices(n, k=1)].mean()
#     obs_target = sub_target.values.mean()

#     R =  n_permutations 

#     better_source = 0
#     better_target = 0

#     all_sources = [g for g in source_list if g in source_cache.index and g != target]
#     all_sources = list(dict.fromkeys(all_sources))

#     if len(all_sources) < n:
#         return {
#             "dCor_sources_obs": obs_sources,
#             "dCor_target_obs": obs_target,
#             "dCor_sources_score": 0.0,
#             "dCor_target_score": 0.0,
#             "error": "Not enough sources in background pool",
#         }, None

#     all_sources_arr = np.array(all_sources)

#     for _ in range(R):
#         perm_sources = np.random.choice(all_sources_arr, size=n, replace=False)

#         perm_sub_source = source_cache.loc[perm_sources, perm_sources]
#         perm_sources_mean = perm_sub_source.values[
#             np.triu_indices(n, k=1)
#         ].mean()

#         perm_sub_target = target_cache.loc[perm_sources, target]
#         perm_target_mean = perm_sub_target.values.mean()

#         if perm_sources_mean >= obs_sources:
#             better_source += 1
#         if perm_target_mean >= obs_target:
#             better_target += 1

#     p_source = (better_source + 1) / (R + 1)
#     p_target = (better_target + 1) / (R + 1)

#     dc2 = -np.log(p_source)
#     dc1 = -np.log(p_target)

#     heatmap_genes = sources + [target]
#     heatmap_matrix = pd.DataFrame(
#         1.0,
#         index=heatmap_genes,
#         columns=heatmap_genes
#     )

#     heatmap_matrix.loc[sources, sources] = source_cache.loc[sources, sources].values
#     heatmap_matrix.loc[sources, target] = target_cache.loc[sources, target].values
#     heatmap_matrix.loc[target, sources] = target_cache.loc[sources, target].values

#     return {
#         "dCor_sources_obs": float(round(obs_sources, 5)),
#         "dCor_target_obs": float(round(obs_target, 5)),
#         "dCor_sources_score": float(round(dc1, 5)),
#         "dCor_target_score": float(round(dc2, 5))
#     }, heatmap_matrix


def source_source_dcorr(sources, source_cache, source_list, n_permutations=500):
    # restrict to genes present in the cache
    sources = [g for g in sources if g in source_cache.index]
    n = len(sources)

    if n < 2:
        return 0.0, 0.0, sources, None  # obs, score, filtered_sources, error

    sub_source = source_cache.loc[sources, sources]
    obs_sources = sub_source.values[np.triu_indices(n, k=1)].mean()

    all_sources = [g for g in source_list if g in source_cache.index]
    all_sources = list(dict.fromkeys(all_sources))

    if len(all_sources) < n:
        return float(round(obs_sources, 5)), 0.0, sources, "Not enough sources in background pool"

    all_sources_arr = np.array(all_sources)
    better_source = 0

    for _ in range(n_permutations):
        perm_sources = np.random.choice(all_sources_arr, size=n, replace=False)
        perm_sub_source = source_cache.loc[perm_sources, perm_sources]
        perm_sources_mean = perm_sub_source.values[
            np.triu_indices(n, k=1)
        ].mean()

        if perm_sources_mean >= obs_sources:
            better_source += 1

    p_source = (better_source + 1) / (n_permutations + 1)
    dc_source_score = -np.log(p_source)

    return (
        float(round(obs_sources, 5)),
        float(round(dc_source_score, 5)),
        sources,
        None
    )


def target_source_dcorr(sources, target, target_cache, source_list, n_permutations=500):
    if target is None:
        return 0.0, 0.0, None  # obs_target, score_target, error

    if target not in target_cache.columns:
        return 0.0, 0.0, f"{target} not in target_cache"

    sources = [g for g in sources if g in target_cache.index]
    n = len(sources)

    if n == 0:
        return 0.0, 0.0, None

    sub_target = target_cache.loc[sources, target]
    obs_target = sub_target.values.mean()

    all_sources = [g for g in source_list if g in target_cache.index and g != target]
    all_sources = list(dict.fromkeys(all_sources))

    if len(all_sources) < n:
        return float(round(obs_target, 5)), 0.0, "Not enough sources in background pool"

    all_sources_arr = np.array(all_sources)
    better_target = 0

    for _ in range(n_permutations):
        perm_sources = np.random.choice(all_sources_arr, size=n, replace=False)
        perm_sub_target = target_cache.loc[perm_sources, target]
        perm_target_mean = perm_sub_target.values.mean()

        if perm_target_mean >= obs_target:
            better_target += 1

    p_target = (better_target + 1) / (n_permutations + 1)
    dc_target_score = -np.log(p_target)

    return (
        float(round(obs_target, 5)),
        float(round(dc_target_score, 5)),
        None
    )


def DCORR_SCORES(
    sources,
    dataset_cache,
    target=None,
    organism="human",
    use_pair_cache=False,
    n_permutations=500
):
    gene_expression = dataset_cache["gene_expression"]

    source_list_key  = ORGANISM_METADATA[organism]["source_background_list"]
    source_list = dataset_cache[source_list_key]

    source_cache = dataset_cache.get("dcorr_source_cache",None)
    target_cache = dataset_cache.get("dcorr_target_cache",None)

    if source_cache is None:
        source_cache = generate_source_dcorr_cache(
            gene_expression,
            source_list
        )
        dataset_cache["dcorr_source_cache"] = source_cache

    if target is not None and target_cache is None:
        target_cache = generate_target_dcorr_cache(
            gene_expression,
            source_list,
            target
        )
        dataset_cache["dcorr_target_cache"] = target_cache

    # 1) source–source
    dcor_src_obs, dcor_src_score, filtered_sources, src_err = source_source_dcorr(
        sources=sources,
        source_cache=source_cache,
        source_list=source_list,
        n_permutations=n_permutations
    )

    # 2) source–target (optional)
    dcor_tgt_obs = 0.0
    dcor_tgt_score = 0.0
    tgt_err = None

    if target is not None and target_cache is not None and len(filtered_sources) > 0:
        dcor_tgt_obs, dcor_tgt_score, tgt_err = target_source_dcorr(
            sources=filtered_sources,
            target=target,
            target_cache=target_cache,
            source_list=source_list,
            n_permutations=n_permutations
        )

    # 3) heatmap
    heatmap_matrix = None
    if len(filtered_sources) >= 2:
        # enforce uniqueness to avoid inflated loc-slices
        filtered_sources = list(dict.fromkeys(filtered_sources))

        heatmap_genes = filtered_sources.copy()
        if target is not None and target_cache is not None and target in target_cache.columns:
            heatmap_genes = filtered_sources + [target]

        heatmap_matrix = pd.DataFrame(
            1.0,
            index=heatmap_genes,
            columns=heatmap_genes
        )

        src_block = source_cache.loc[filtered_sources, filtered_sources]
        heatmap_matrix.loc[filtered_sources, filtered_sources] = src_block.values

        if target is not None and target_cache is not None and target in target_cache.columns:
            tgt_vals = target_cache.loc[filtered_sources, target].values
            heatmap_matrix.loc[filtered_sources, target] = tgt_vals
            heatmap_matrix.loc[target, filtered_sources] = tgt_vals

    result = {
        "dCor_sources_obs": dcor_src_obs,
        "dCor_target_obs": dcor_tgt_obs,
        "dCor_sources_score": dcor_src_score,
        "dCor_target_score": dcor_tgt_score
    }

    if src_err is not None:
        result["source_error"] = src_err

    if tgt_err is not None:
        result["target_error"] = tgt_err

    return result, heatmap_matrix