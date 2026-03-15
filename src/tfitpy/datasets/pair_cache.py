"""
Build a cache of pair wise scores
"""
from pathlib import Path
from tfitpy.datasets.tf import load_tflist
import time
from itertools import combinations
import pandas as pd
from joblib import Parallel, delayed

from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.datasets.go import  GO_DATASET  
CACHE_COLUMNS = [
    'gene1', 'gene2',
    "shortest_PPI_path_score_hippie",
    "shortest_PPI_path_score_stringdb",
    "shortest_PPI_path_score_biogrid",
    "shared_PPI_partners_score_hippie",
    "shared_PPI_partners_score_stringdb",
    "shared_PPI_partners_score_biogrid",
    "goa_similarity_lin",
    "goa_similarity_resnik",
    "goa_similarity_jc"
]



# def compute_scores_for_pair(tf1, tf2, datasets):
#     """Compute all scores for one pair"""
#     from tfitpy.indices.ppi import _ppi_single_pass
#     from tfitpy.indices.go import _gene_sim_bma_with_terms, lin_sim, resnik_sim, _jc_sim

#     scores = {'gene1': tf1, 'gene2': tf2}

#     for db_key in ["hippie", "stringdb", "biogrid"]:
#         graph = datasets[db_key]
#         background_size = len(graph.nodes())

#         path_score, partner_score = _ppi_single_pass(
#             [(tf1, tf2)], graph, background_size
#         )

#         scores[f'shortest_PPI_path_score_{db_key}'] = path_score
#         scores[f'shared_PPI_partners_score_{db_key}'] = partner_score
#     # GO scores
#     godag = datasets["go"]["godag"]
#     gene2go = datasets["go"]["gene2go"]
#     termcounts = datasets["go"]["termcounts"]
    
#     terms1 = list(gene2go.get(tf1, set()))
#     terms2 = list(gene2go.get(tf2, set()))
    
#     scores['goa_similarity_lin'] = _gene_sim_bma_with_terms(
#         terms1, terms2, godag, termcounts, lin_sim
#     )
#     scores['goa_similarity_resnik'] = _gene_sim_bma_with_terms(
#         terms1, terms2, godag, termcounts, resnik_sim
#     )
#     scores['goa_similarity_jc'] = _gene_sim_bma_with_terms(
#         terms1, terms2, godag, termcounts, _jc_sim
#     )
#     return scores


def compute_scores_batch(pairs_batch, data_path):
    """Compute scores for a batch of pairs - TermCounts built once per batch"""
    from tfitpy.indices.ppi import _ppi_single_pass
    from tfitpy.indices.go import _gene_sim_bma_with_terms, lin_sim, resnik_sim, _jc_sim
    from goatools.semantic import TermCounts
    
    datasets = {
        "hippie": PPI_DATASETS["hippie"]["load"](data_path),
        "stringdb": PPI_DATASETS["stringdb"]["load"](data_path),
        "biogrid": PPI_DATASETS["biogrid"]["load"](data_path),
        "go": GO_DATASET["go"]["load"](data_path)
    }
    # Build TermCounts once for this batch
    termcounts = TermCounts(datasets["go"]["godag"], datasets["go"]["gene2go"])
    
    results = []
    for tf1, tf2 in pairs_batch:
        scores = {'gene1': tf1, 'gene2': tf2}
        
        # PPI scores
        for db_key in ["hippie", "stringdb", "biogrid"]:
            graph = datasets[db_key]
            background_size = len(graph.nodes())
            path_score, partner_score = _ppi_single_pass(
                [(tf1, tf2)], graph, background_size
            )
            scores[f'shortest_PPI_path_score_{db_key}'] = path_score
            scores[f'shared_PPI_partners_score_{db_key}'] = partner_score
        
        # GO scores
        godag = datasets["go"]["godag"]
        gene2go = datasets["go"]["gene2go"]
        
        terms1 = list(gene2go.get(tf1, set()))
        terms2 = list(gene2go.get(tf2, set()))
        
        scores['goa_similarity_lin'] = _gene_sim_bma_with_terms(
            terms1, terms2, godag, termcounts, lin_sim
        )
        scores['goa_similarity_resnik'] = _gene_sim_bma_with_terms(
            terms1, terms2, godag, termcounts, resnik_sim
        )
        scores['goa_similarity_jc'] = _gene_sim_bma_with_terms(
            terms1, terms2, godag, termcounts, _jc_sim
        )
        
        results.append(scores)
    
    return results

def build(data_path, rerun=False, n_jobs=4,batch_size=5000):
    """
    """
    cache_file = Path(data_path)/"pairwise_score_cache.parquet"
    # Check if cache exists and has all required columns
    if cache_file.exists() and not rerun:
        existing_df = pd.read_parquet(cache_file)
        missing_cols = set(CACHE_COLUMNS) - set(existing_df.columns)

        if not missing_cols:
            print(f"Cache exists with all {len(CACHE_COLUMNS)} columns")
            return existing_df
        else:
            print(f"Cache exists but missing columns: {missing_cols}")
            print("Rebuilding cache...")

    tf = load_tflist(data_path)
    tflist = tf["gene_name"].tolist()[0:5]
    print(f"Number of TFs: {len(tflist)}")

    

    start = time.time()

    df = get_pairs(tflist)
    print(f"Generated {len(df):,} pairs")

    pairs_list = list(zip(df['gene1'], df['gene2']))
    
    # Split into batches
    batches = [pairs_list[i:i+batch_size] for i in range(0, len(pairs_list), batch_size)]
    
    print(f"Computing scores in {len(batches)} batches...")
    batch_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_scores_batch)(batch,data_path)
        for batch in batches
    )
    
    # Flatten results
    all_results = [item for batch in batch_results for item in batch]
    scores_df = pd.DataFrame(all_results)
    
    scores_df.to_parquet(cache_file, compression='snappy', index=False)

    # Parallel compute
    # results = Parallel(n_jobs=n_jobs, verbose=10)(
    #     delayed(compute_scores_for_pair)(row.gene1, row.gene2, datasets)
    #     for row in df.itertuples()
    # )

    # scores_df = pd.DataFrame(results)
    # scores_df.to_parquet(cache_file, compression='snappy', index=False)

    elapsed = time.time() - start
    print(f"Took {elapsed:.2f} seconds")


def get_pairs(tflist):
    """
    Generates a dataframe with all possible pairwise combinations of tf list. 
    2 cols: gene1, gene2 in alphabetical order
    """
    # Generate all pairs
    pairs = list(combinations(sorted(tflist), 2))

    # Create DataFrame
    df = pd.DataFrame(pairs, columns=['gene1', 'gene2'])

    return df


def load(data_path):
    """

    """
    cache_file = Path(data_path)/"pairwise_score_cache.parquet"
    df = pd.read_parquet(cache_file)
    return df


def download():
    """nothing here. just a placeholder"""
    return

PAIRWISE_CACHE = {
    "pairwise_score_cache": {
        "download": download,
        "process": build,
        "load": load,
    }
}
