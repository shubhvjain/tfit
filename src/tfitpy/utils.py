"""Utility functions """
import pandas as pd
from pathlib import Path
import os

def generate_tf_pairs(gene_cluster):
    """Generate all unique pairs from comma-separated gene cluster."""
    genes = [g.strip() for g in gene_cluster]
    return [(g1, g2) for i, g1 in enumerate(genes) for g2 in genes[i+1:]]

ORGANISM_METADATA = {
    "human":{ 
        "ppi_keys":["hippie","stringdb","biogrid"],
        "go_key":"go",
        "pair_cache":"pairwise_score_cache_human",
        "pair_cache_trap":"trap_cache_human",
        "jaspar":"jaspar",
        "source_background_list":"coreglist"
    },
    "arabidopsis":{
        "ppi_keys":["stringdb_arabidopsis"],
        "go_key":"go_arabidopsis",
        "pair_cache":"pairwise_score_cache_arabidopsis",
        "jaspar":"jaspar_plant",
        "gene_mapping": "plant_gene_mapping",
        "source_background_list":"tflist_plant"
    }
}
INDICES_DATA = {
    "ppi_cached_indices" : ["shortest_PPI_path_score","shared_PPI_partners_score"],
    "ppi_network_indices" : ["density","density_score","lcc","lcc_score","tc","tc_score","node_found_ratio"],
    "ppi_source_indices":["ppi_edges"],
    "go_cached_indices":["goa_similarity_lin","goa_similarity_resnik","goa_similarity_jc"],
    "go_source_indices":["go_ora"],
    
}