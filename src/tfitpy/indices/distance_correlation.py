import numpy as np
import pandas as pd
import dcor
from joblib import Parallel, delayed
import os
import json
from tfitpy.utils import ORGANISM_METADATA, INDICES_DATA
from tfitpy.datasets.cache_dcorr import generate_source_dcorr_cache, generate_target_dcorr_cache
from scipy.stats import spearmanr


def compute_D_from_source_cache(source_cache, percentile=95):
    """
    source_cache: square DataFrame of pairwise dCor values (source_list x source_list)
    Returns D = the percentile threshold from the upper triangle (excluding diagonal)
    """
    vals = source_cache.values
    n = vals.shape[0]

    if n < 2:
        return 0.0

    upper_vals = vals[np.triu_indices(n, k=1)]
    if upper_vals.size == 0:
        return 0.0

    return float(np.percentile(upper_vals, percentile))

def expression_coherence_from_cache(genes, cache, D):
    genes = [g for g in genes if g in cache.index and g in cache.columns]
    n = len(genes)

    if n < 2:
        return 0.0

    sub = cache.loc[genes, genes].values
    pairwise = sub[np.triu_indices(n, k=1)]

    if pairwise.size == 0:
        return 0.0

    return float(np.mean(pairwise > D))


def build_heatmap_matrix(sources, target, source_cache, target_cache, gene_expression):
    valid_sources = [g for g in sources if g in source_cache.index]
    rows = []

    # source-source pairs
    for i, g1 in enumerate(valid_sources):
        for g2 in valid_sources[i + 1:]:
            rows.append({
                "gene1": g1,
                "gene2": g2,
                "dCor": source_cache.loc[g1, g2],
                "rho": None,  # direction not meaningful for source-source
            })

    # source-target pairs
    if target is not None and target_cache is not None and target in target_cache.columns:
        for g in valid_sources:
            if g not in target_cache.index:
                continue
            dcor_val = target_cache.loc[g, target]
            rho_val, _ = spearmanr(gene_expression[g].values, gene_expression[target].values)
            rows.append({
                "gene1": g,
                "gene2": target,
                "dCor": dcor_val,
                "rho": float(round(rho_val, 5)),
            })

    return pd.DataFrame(rows)


def source_source_dcorr(sources, source_cache, source_list, n_permutations=100):
    sources = [g for g in sources if g in source_cache.index]
    n = len(sources)

    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    sub_source = source_cache.loc[sources, sources]
    obs_sources = sub_source.values[np.triu_indices(n, k=1)].mean()

    all_sources = [g for g in source_list if g in source_cache.index and g not in sources]
    all_sources = list(dict.fromkeys(all_sources))

    D = compute_D_from_source_cache(source_cache)
    obs_ec = expression_coherence_from_cache(sources, source_cache, D)

    if len(all_sources) < n:
        return (
            float(round(obs_sources, 5)),
            0.0,
            0.0,
            float(round(obs_ec, 5)),
            0.0,
            0.0,
        )

    all_sources_arr = np.array(all_sources)
    better_source = 0
    better_ec = 0
    perm_means = np.empty(n_permutations)
    perm_ec = np.empty(n_permutations)

    for i in range(n_permutations):
        perm_sources = np.random.choice(all_sources_arr, size=n, replace=False)
        perm_sub_source = source_cache.loc[perm_sources, perm_sources]
        perm_sources_mean = perm_sub_source.values[np.triu_indices(n, k=1)].mean()
        perm_means[i] = perm_sources_mean

        if perm_sources_mean >= obs_sources:
            better_source += 1

        # reuse the same permuted draw for the EC null distribution
        perm_ec_val = expression_coherence_from_cache(list(perm_sources), source_cache, D)
        perm_ec[i] = perm_ec_val

        if perm_ec_val >= obs_ec:
            better_ec += 1

    p_source = (better_source + 1) / (n_permutations + 1)
    dc_source_score = -np.log(p_source)

    null_std = perm_means.std(ddof=1)
    z_source = (obs_sources - perm_means.mean()) / null_std if null_std > 0 else 0.0

    p_ec = (better_ec + 1) / (n_permutations + 1)
    dc_ec_score = -np.log(p_ec)

    ec_null_std = perm_ec.std(ddof=1)
    z_ec = (obs_ec - perm_ec.mean()) / ec_null_std if ec_null_std > 0 else 0.0

    return (
        float(round(obs_sources, 5)),
        float(round(dc_source_score, 5)),
        float(round(z_source, 5)),
        float(round(obs_ec, 5)),
        float(round(dc_ec_score, 5)),
        float(round(z_ec, 5)),
    )


def target_source_dcorr(sources, target, target_cache, source_list, n_permutations=100):
    if target is None:
        return 0.0, 0.0, 0.0, None

    if target not in target_cache.columns:
        return 0.0, 0.0, 0.0, f"{target} not in target_cache"

    sources = [g for g in sources if g in target_cache.index]
    n = len(sources)

    if n == 0:
        return 0.0, 0.0, 0.0, None

    sub_target = target_cache.loc[sources, target]
    obs_target = sub_target.values.mean()

    all_sources = [g for g in source_list if g in target_cache.index and g != target]
    all_sources = list(dict.fromkeys(all_sources))

    if len(all_sources) < n:
        return float(round(obs_target, 5)), 0.0, 0.0, None

    all_sources_arr = np.array(all_sources)
    better_target = 0
    perm_means = np.empty(n_permutations)

    for i in range(n_permutations):
        perm_sources = np.random.choice(all_sources_arr, size=n, replace=False)
        perm_sub_target = target_cache.loc[perm_sources, target]
        perm_target_mean = perm_sub_target.values.mean()
        perm_means[i] = perm_target_mean

        if perm_target_mean >= obs_target:
            better_target += 1

    p_target = (better_target + 1) / (n_permutations + 1)
    dc_target_score = -np.log(p_target)

    null_std = perm_means.std(ddof=1)
    z_target = (obs_target - perm_means.mean()) / null_std if null_std > 0 else 0.0

    return (
        float(round(obs_target, 5)),
        float(round(dc_target_score, 5)),
        float(round(z_target, 5)),
        None,
    )

def DCORR_SCORES(
    sources,
    dataset_cache,
    target=None,
    organism="human",
    n_permutations=200,
):
    gene_expression = dataset_cache["gene_expression"]

    source_list_key = ORGANISM_METADATA[organism]["source_background_list"]
    source_list = dataset_cache[source_list_key]

    source_cache = dataset_cache.get("dcorr_source_cache")
    target_cache = dataset_cache.get("dcorr_target_cache")

    if source_cache is None:
        source_cache = generate_source_dcorr_cache(gene_expression, source_list)
        dataset_cache["dcorr_source_cache"] = source_cache

    if target is not None and target_cache is None:
        target_cache = generate_target_dcorr_cache(
            gene_expression,
            source_list,
            target,
        )
        dataset_cache["dcorr_target_cache"] = target_cache

    if "dcorr_D_threshold" not in dataset_cache:
        dataset_cache["dcorr_D_threshold"] = compute_D_from_source_cache(source_cache)

    dcor_src_obs, dcor_src_score, z_src, src_ec, src_ec_score, z_ec = source_source_dcorr(
        sources=sources,
        source_cache=source_cache,
        source_list=source_list,
        n_permutations=n_permutations,
    )

    dcor_tgt_obs = 0.0
    dcor_tgt_score = 0.0
    z_tgt = 0.0
    tgt_err = None

    if target is not None and target_cache is not None:
        dcor_tgt_obs, dcor_tgt_score, z_tgt, tgt_err = target_source_dcorr(
            sources=sources,
            target=target,
            target_cache=target_cache,
            source_list=source_list,
            n_permutations=n_permutations,
        )

    heatmap_matrix = None
    if len(sources) >= 2:
        heatmap_matrix = build_heatmap_matrix(sources, target, source_cache, target_cache, gene_expression)

    result = {
        "dCor_sources": dcor_src_obs,
        "dCor_target": dcor_tgt_obs,
        "dCor_sources_score": dcor_src_score,
        "dCor_target_score": dcor_tgt_score,
        "dCor_sources_z": z_src,
        "dCor_target_z": z_tgt,
        "dCor_EC": src_ec,
        "dCor_EC_score": src_ec_score,
        "dCor_EC_z": z_ec,
    }

    if tgt_err is not None:
        result["target_error"] = tgt_err

    return result, heatmap_matrix