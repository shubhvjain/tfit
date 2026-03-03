import pandas as pd
from pathlib import Path
import decoupler as dc
import networkx as nx


def download_collectri(data_path):
    """No download required, used via package"""
    return 


def process_collectri(data_path):
    """No processing required """
    return 


def load_collectri(data_path):
    """Returns load_collectri as a network"""
    edges = dc.op.collectri(organism='human')
    # G = nx.from_pandas_edgelist(
    #     edges, 
    #     source='source', 
    #     target='target', 
    #     edge_attr=True,  # preserve all columns as edge attributes
    #     create_using=nx.DiGraph()
    # )
    # return G
    return edges


GRN_DATASETS = {
    'collectri': {
        'download': download_collectri,
        'process': process_collectri,
        'load': load_collectri
    }
}