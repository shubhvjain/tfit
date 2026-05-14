from pathlib import Path
import os
from tfitpy.datasets.ppi import PPI_DATASETS
import igraph as ig
import gzip
import pickle
from pathlib import Path


def load_null_graphs(data_path, ppi_key, n):
    """"""
    file_path = Path(data_path)/"network_analysis_ppi" / \
        f"{ppi_key}_null_{n}_models.pkl.gz"
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_null_graphs(ppi_random):
    """
    Converts 500 edge lists into igraph objects once.
    """
    n_nodes = len(ppi_random['mapping'])
    graphs = []

    print(
        f"Pre-constructing {len(ppi_random['null_models'])} igraph objects...")
    for edges in ppi_random['null_models']:
        # Create igraph directly from the edge list
        g = ig.Graph(n=n_nodes, edges=edges, directed=False)
        graphs.append(g)

    return graphs


def get_aligned_igraph(data_path, ppi_key, ppi_random):
    """
    1. Loads the PPI (e.g., HIPPIE) using your existing NetworkX loader.
    2. Converts it to igraph.
    3. Re-indexes it to match the 'mapping' used in the null models.
    """
    # Use your existing loader
    G_nx = PPI_DATASETS[ppi_key]['load'](data_path)

    # Get the source-of-truth mapping from your random models file
    mapping = ppi_random['mapping']
    n_nodes = len(mapping)

    # Extract edges from NetworkX using the mapping
    # This ensures "TP53" gets the exact same integer ID as in the 500 nulls
    edges = []
    for u, v in G_nx.edges():
        if u in mapping and v in mapping:
            edges.append((mapping[u], mapping[v]))

    # Create the main igraph object using these integer IDs
    G_ig = ig.Graph(n=n_nodes, edges=edges, directed=False)

    # Store the names in a vertex attribute so you can still look them up
    G_ig.vs['name'] = ppi_random['node_list']

    return G_ig


def load_all_network_data(data_path, ppi_key="stringdb", n=500):
    """
    use this method to prepare for running computer method 
    """
    null_model_data = load_null_graphs(data_path, ppi_key, n)
    G = get_aligned_igraph(data_path, ppi_key, null_model_data)
    NULL_MODELS = prepare_null_graphs(null_model_data)
    MAPPING = null_model_data.get("mapping")
    return G, NULL_MODELS, MAPPING
