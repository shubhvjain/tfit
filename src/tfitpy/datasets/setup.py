from pathlib import Path
import os


from tfitpy.datasets.grn import GRN_DATASETS
from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.datasets.gene_names import GENE_DATASETS
from tfitpy.datasets.go import GO_DATASET
from tfitpy.datasets.tf import TF_DATASET
from tfitpy.datasets.pair_cache import PAIRWISE_CACHE


DATASETS = {
    **GENE_DATASETS,
    **GRN_DATASETS,
    **PPI_DATASETS,
    **GO_DATASET,
    **TF_DATASET,
    **PAIRWISE_CACHE
}

FIRST_ORDER = ["biomart", "gencode"]



def install(data_path):
    """
    Download and process all datasets
    
    This should be run once after package installation to prepare all data.
    
    Args:
        data_path: Path where datasets will be stored
        
    Example:
        >>> from tfit import setup_datasets
        >>> setup_datasets(data_path='./data')
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


