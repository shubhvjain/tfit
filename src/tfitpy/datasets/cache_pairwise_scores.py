"""
Build a cache of pair wise scores
"""
from pathlib import Path
from tfitpy.datasets.regulators import get_regulator_list
import time
from itertools import combinations
import pandas as pd
from joblib import Parallel, delayed

from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.datasets.go import  GO_DATASET  

from tfitpy.utils import ORGANISM_METADATA, INDICES_DATA

organism_dataset = {
    "human":{
        "hippie": PPI_DATASETS["hippie"],
        "stringdb": PPI_DATASETS["stringdb"],
        "biogrid": PPI_DATASETS["biogrid"],
        "go": GO_DATASET["go"]
    },
    "arabidopsis":{
        "stringdb_arabidopsis": PPI_DATASETS["stringdb_arabidopsis"],
        "go_arabidopsis":GO_DATASET["go_arabidopsis"]
    }
}

def compute_scores_batch(pairs_batch, data_path,organism="human"):
    """Compute scores for a batch of pairs - TermCounts built once per batch"""
    from tfitpy.indices.ppi import _ppi_single_pass , shared_partners_pairwise, shortest_path_pairwise
    from tfitpy.indices.go import _gene_sim_bma_with_terms, lin_sim, resnik_sim, _jc_sim
    from goatools.semantic import TermCounts

    datasets = {}
    for d in organism_dataset[organism]:
        datasets[d] = organism_dataset[organism][d]["load"](data_path)
    
    go_key = ORGANISM_METADATA[organism]["go_key"]
    ppi_keys =  ORGANISM_METADATA[organism]["ppi_keys"]
    # Build TermCounts once for this batch
    termcounts = TermCounts(datasets[go_key]["godag"], datasets[go_key]["gene2go"])
    
    results = []
    for tf1, tf2 in pairs_batch:
        scores = {'gene1': tf1, 'gene2': tf2}
        
        # PPI scores
        for db_key in ppi_keys:
            graph = datasets[db_key]
            background_size = len(graph.nodes())

            score1,p,c = shared_partners_pairwise(tf1,tf2,graph,background_size)
            score2,l = shortest_path_pairwise(tf1,tf2,graph)
            scores[f'shortest_PPI_path_score_{db_key}'] = score2
            scores[f'shared_PPI_partners_score_{db_key}'] = score1
        
        # GO scores
        godag = datasets[ go_key ]["godag"]
        gene2go = datasets[ go_key ]["gene2go"]
        
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

def build(data_path, rerun=False, n_jobs=-1,batch_size=10000,organism="human"):
    """
    """

    cache_file = Path(data_path)/f"pairwise_score_cache_{organism}.parquet"

    if cache_file.exists() and not rerun:
        return 
    
    tflist =  get_regulator_list(data_path,organism)  
    print(f"Number of TFs: {len(tflist)}")
    
    start = time.time()

    df = get_pairs(tflist)
    print(f"Generated {len(df):,} pairs")

    pairs_list = list(zip(df['gene1'], df['gene2']))
    
    # Split into batches
    batches = [pairs_list[i:i+batch_size] for i in range(0, len(pairs_list), batch_size)]
    
    print(f"Computing scores in {len(batches)} batches...")
    batch_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_scores_batch)(batch,data_path,organism)
        for batch in batches
    )
    
    # Flatten results
    all_results = [item for batch in batch_results for item in batch]
    scores_df = pd.DataFrame(all_results)
    scores_df.to_parquet(cache_file, compression='snappy', index=False)
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


def load(data_path,organism="human"):
    """

    """
    cache_file = Path(data_path)/f"pairwise_score_cache_{organism}.parquet"
    df = pd.read_parquet(cache_file)
    df = df.set_index(['gene1', 'gene2'])
    return df



def load_human(data_path):
    """"""
    return load(data_path,"human")

def load_arabidopsis(data_path):
    """"""
    return load(data_path,"arabidopsis")

def process_human(data_path):
    """"""
    build(data_path,organism="human")

def process_arabidopsis(data_path):
    """"""
    build(data_path,organism="arabidopsis")

def download(data_path):
    """nothing here. just a placeholder"""
    return


PAIRWISE_CACHE = {
    "pairwise_score_cache_human": {
        "download": download,
        "process": process_human,
        "load": load_human,
    },
    "pairwise_score_cache_arabidopsis": {
        "download": download,
        "process": process_arabidopsis,
        "load": load_arabidopsis,
    }
}
