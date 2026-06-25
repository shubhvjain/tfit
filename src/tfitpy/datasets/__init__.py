from pathlib import Path
import os

from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.datasets.ppi_networks import PPI_NULL_NETWORKS
from tfitpy.datasets.grn import GRN_DATASETS
from tfitpy.datasets.gene_names import GENE_DATASETS
from tfitpy.datasets.go import GO_DATASET
from tfitpy.datasets.regulators import TF_DATASET
from tfitpy.datasets.binding import BINDING_DATASET
from tfitpy.datasets.cache_binding import BINDING_CACHE
from  tfitpy.datasets.cache_pairwise_scores import PAIRWISE_CACHE
from tfitpy.datasets.cache_dcorr import generate_source_dcorr_cache,generate_target_dcorr_cache
from tfitpy.utils import ORGANISM_METADATA

HUMAN_DATASETS = { 
    "stringdb": PPI_DATASETS["stringdb"],
    "hippie": PPI_DATASETS["hippie"],
    "biogrid": PPI_DATASETS["biogrid"],
    "stringdb_null": PPI_NULL_NETWORKS["stringdb_null"],
    "hippie_null": PPI_NULL_NETWORKS["hippie_null"],
    "biogrid_null": PPI_NULL_NETWORKS["biogrid_null"],
    "go": GO_DATASET["go"],
    "jaspar": BINDING_DATASET["jaspar"],
    "coreglist": TF_DATASET["coreglist"],
    "tflist_human": TF_DATASET["tflist_human"],
}

HUMAN_CACHE = {
    "pairwise_score_cache_human" : PAIRWISE_CACHE["pairwise_score_cache_human"],
    "trap_cache_human": BINDING_CACHE["trap_cache_human"]
}

arabidopsis_DATASETS  = {
    "plant_gene_mapping": GENE_DATASETS["plant_gene_mapping"],
    "stringdb_arabidopsis": PPI_DATASETS["stringdb_arabidopsis"],
    "stringdb_arabidopsis_null": PPI_NULL_NETWORKS["stringdb_arabidopsis_null"],
    "go_arabidopsis": GO_DATASET["go_arabidopsis"],
    "jaspar_plant":  BINDING_DATASET["jaspar_plant"],
    "tflist_plant": TF_DATASET["tflist_plant"]
}

arabidopsis_CACHE = {}

def load(data_path, organism="human", load_cache=True,generate_dcorr_cache=True,gene_expression_data=None,targets=[]):
    """
    return a dict of datasets required to generate indices. This can be called once and reused.
    """
    dataset_loaded = {}

    org_dataset = {}
    org_cache = {}
    if organism == "human":
        org_dataset = HUMAN_DATASETS
        org_cache = HUMAN_CACHE
    elif organism == "arabidopsis":
        org_dataset = arabidopsis_DATASETS
        org_cache = arabidopsis_CACHE
    else:
        raise ValueError('invalid organism')
  

    for d in org_dataset.keys():
        dataset_loaded[d] = org_dataset[d]["load"](data_path)
    if load_cache:
        for d in org_cache.keys():
            dataset_loaded[d] = org_cache[d]["load"](data_path)

    if generate_dcorr_cache:
        if gene_expression_data is None :
            raise ValueError("Expression data not provided")
        dataset_loaded["gene_expression"] = gene_expression_data

        source_list_key  = ORGANISM_METADATA[organism]["source_background_list"]
        source_list = dataset_loaded[source_list_key]
        # generate source_source cache
        dataset_loaded["dcorr_source_cache"] = generate_source_dcorr_cache(gene_expression_data,source_list)
        #  generate source target cache
        if targets is not None:
            dataset_loaded["dcorr_target_cache"] = generate_target_dcorr_cache(gene_expression_data,source_list,targets)

    return dataset_loaded



DATASETS = {
    **GENE_DATASETS,
    **GRN_DATASETS,
    **PPI_DATASETS,
    **PPI_NULL_NETWORKS,
    **GO_DATASET,
    **TF_DATASET,
    **PAIRWISE_CACHE,
    **BINDING_DATASET,
    **BINDING_CACHE
}

FIRST_ORDER = ["biomart", "gencode","go","tflist"]

def install(data_path):
    """
    Download and process all datasets
    
    This should be run once after package installation to prepare all data.
    
    Args:
        data_path: Path where datasets will be stored
        
    """
    data_path = Path(os.path.expandvars(data_path))
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up datasets in: {data_path}")
    print("=" * 60)


    d = DATASETS.keys()
    first = FIRST_ORDER
    rest_part  = [k for k in d if k not in first]  
    ordered_keys = first + rest_part
    # print(ordered_keys)

    
    # Download all datasets
    print("\n[1/2] Downloading datasets...")
    for ds_name in ordered_keys:
        ds_config = DATASETS[ds_name]
        print(f"\n  Downloading {ds_name}...")
        try:
            ds_config['download'](data_path)
            print(f"{ds_name} downloaded")
        except Exception as e:
            print(f"{ds_name} failed: {e}")
            raise
    
    # Process all datasets
    print("\n[2/2] Processing datasets...")
    for ds_name, ds_config in DATASETS.items():
        print(f"\n  Processing {ds_name}...")
        try:
            ds_config['process'](data_path)
            print(f"{ds_name} processed")
        except Exception as e:
            print(f"{ds_name} failed: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("All datasets setup complete!")
    print(f"Data stored in: {data_path.absolute()}")




__all__ = [
    "DATASETS"
    "install",
    "load"
]
