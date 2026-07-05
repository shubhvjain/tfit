import numpy as np
import pandas as pd
import dcor
from joblib import Parallel, delayed


def _centered_all(expr, chunk=500):
    """expr: (n_samples, n_genes) float array.
    Returns (n_genes, n_samples, n_samples) double-centered distance matrices."""
    n_samples, n_genes = expr.shape
    out = np.empty((n_genes, n_samples, n_samples), dtype=np.float32)
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        block = expr[:, start:end]                          # (n_samples, c)
        d = np.abs(block[:, None, :] - block[None, :, :])    # (n_samples, n_samples, c)
        d = np.moveaxis(d, 2, 0)                              # (c, n_samples, n_samples)
        row_mean = d.mean(axis=2, keepdims=True)
        col_mean = d.mean(axis=1, keepdims=True)
        grand_mean = d.mean(axis=(1, 2), keepdims=True)
        out[start:end] = d - row_mean - col_mean + grand_mean
    return out


def _dcor_matrix_from_centered(centered):
    # centered: (G, n_samples, n_samples)
    G = centered.shape[0]
    flat = centered.reshape(G, -1)              # (G, n*n)
    n2 = flat.shape[1]

    gram = (flat @ flat.T) / n2                  # (G, G)
    dvar = np.diag(gram)
    denom = np.sqrt(np.outer(dvar, dvar))

    with np.errstate(invalid="ignore", divide="ignore"):
        dcorr = np.sqrt(np.clip(gram / denom, 0, None))
    np.fill_diagonal(dcorr, 1.0)
    return dcorr



def generate_source_dcorr_cache(gene_expression, source_list, n_jobs=-1, chunk=500):
    #print(source_list)
    sources = [g for g in source_list if g in gene_expression.columns]
    expr = np.ascontiguousarray(gene_expression[sources].to_numpy(), dtype=np.float64)

    centered = _centered_all(expr, chunk=chunk)          # (n, n_samples, n_samples)
    dmat = _dcor_matrix_from_centered(centered)

    return pd.DataFrame(dmat, index=sources, columns=sources)


def generate_target_dcorr_cache(gene_expression, source_list, targets, n_jobs=-1, chunk=500):
    if isinstance(targets, (str, int)):
        targets = [targets]

    targets = [t for t in targets if t in gene_expression.columns]
    if not targets:
        raise ValueError("no valid targets in gene_expression")

    sources = [g for g in source_list if g in gene_expression.columns]
    if not sources:
        raise ValueError("no valid sources in gene_expression")

    expr = np.ascontiguousarray(gene_expression[sources].to_numpy(), dtype=np.float64)
    target_expr = np.ascontiguousarray(gene_expression[targets].to_numpy(), dtype=np.float64)

    source_centered = _centered_all(expr, chunk=chunk)          # (n_src, s, s)
    target_centered = _centered_all(target_expr, chunk=chunk)   # (n_tgt, s, s)

    n_samples = expr.shape[0]
    n2 = n_samples * n_samples

    src_flat = source_centered.reshape(len(sources), -1)
    tgt_flat = target_centered.reshape(len(targets), -1)

    gram = (src_flat @ tgt_flat.T) / n2                          # (n_src, n_tgt)
    dvar_s = (src_flat * src_flat).mean(axis=1)
    dvar_t = (tgt_flat * tgt_flat).mean(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        dmat = np.sqrt(np.clip(gram / np.sqrt(np.outer(dvar_s, dvar_t)), 0, None))

    return pd.DataFrame(dmat, index=sources, columns=targets)