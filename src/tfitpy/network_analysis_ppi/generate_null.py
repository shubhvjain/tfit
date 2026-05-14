"""
generating n randomized PPI networks from the base model
"""
import igraph as ig
import numpy as np
import pickle
import gzip
from pathlib import Path
from joblib import Parallel, delayed
import random
from tfitpy.datasets.ppi import PPI_DATASETS

def _rewire_task(edge_list, n_nodes, n_swaps, seed):
    """
    Worker function. Note: Random seed is set inside to ensure 
    each parallel worker produces a unique network.
    """
    # Set the seed for the worker
    random.seed(seed)
    # Create graph from edge list
    g = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
    
    g.rewire(n=n_swaps, mode="simple")
    return np.array(g.get_edgelist(), dtype=np.uint32)

def generate_null_networks(data_path, args=None):
    if args is None: args = {}
    ppi_key = args.get('ppi_key', 'stringdb')
    n_jobs = args.get('njobs', -1) 
    n_models = args.get('nmodels', 500)
    rerun = args.get('rerun', False)

    output_dir = Path(data_path) / "network_analysis_ppi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{ppi_key}_null_{n_models}_models.pkl.gz"

    if output_file.exists() and not rerun:
        print(f"File exists: {output_file}")
        return output_file

    # Load and map
    G_nx = PPI_DATASETS[ppi_key]['load'](data_path)
    node_names = list(G_nx.nodes())
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    n_nodes = len(node_names)
    edge_list = [(name_to_idx[u], name_to_idx[v]) for u, v in G_nx.edges()]
    n_swaps = len(edge_list) * 10

    print(f"Generating {n_models} networks via joblib ({n_jobs} cores)...")

    # 1. The Parallel Call
    # 'prefer="processes"' is usually best for CPU-bound tasks like rewiring.
    # verbose=10 provides a nice progress bar in your HPC logs.
    null_edge_collections = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_rewire_task)(edge_list, n_nodes, n_swaps, i) 
        for i in range(n_models)
    )

    # 2. Save
    print(f"Saving to {output_file}...")
    data_to_save = {
        "ppi_key": ppi_key,
        "n_models": n_models,
        "n_swaps": n_swaps,
        "mapping": name_to_idx,
        "node_list": node_names,
        "null_models": null_edge_collections
    }

    with gzip.open(output_file, "wb") as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_file
