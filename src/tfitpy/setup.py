"""
This is the setup script. It can be run via the CLI.
It main purpose is to prepare the local system by downloading all the data required by the package and finishing other preprocessing steps. 
It requires the user to specify a location to a directory where all the data will be stored. Since the data is same, it can be installed as globally and the same folder path can be used in multiple projects. 

THe scripts downloads the following databases. Please see the reference page for more details about the source and their citations. 

- HIPPIE PPI (:cite:t:`hippie`)
- StringDB PPI (:cite:t:`string`)
- BioGRID Database (:cite:t:`biogrid`)
- BioMart (:cite:t:`biomart`)

"""

# from typing import List
# from pathlib import Path
# from typing import Any, Dict, Optional

# import tfitpy.data.hippie as hippie
# import tfitpy.data.biomart as biomart
# import tfitpy.data.stringdb as stringdb
# import tfitpy.data.biogrid as biogrid


# DATA_SOURCES = [
#     hippie,
#     biomart,
#     stringdb,
#     biogrid
# ]

# PPI_BUILD = [
#     hippie
# ]


# def ensure_all_data(config: Optional[Dict[str, Any]] = None) -> None:
#     """Download all required data sources using the provided config.

#     Args:
#         config: Global config dict (may be {} or None). Passed to each
#             data source's download() function.
#     """
#     missing = [src for src in DATA_SOURCES if not src.is_ready(config)]
    
#     if  missing:
#         print(f"Downloading {len(missing)} missing sources...")
#         for src in missing:
#         #src_name = src.__name__.split('.')[-1]  # "hippie", "biomart", etc.
#         #print(f"Downloading {src_name}...")
#             src.download(config)

#     ppi_missing = [src for src in PPI_BUILD if not src.is_ready(config,"ppi_hgnc_filename")]

#     if  ppi_missing:
#         print(f"Generating PPI for {len(ppi_missing)} missing sources...")
#         for src in ppi_missing:
#         #src_name = src.__name__.split('.')[-1]  # "hippie", "biomart", etc.
#         #print(f"Downloading {src_name}...")
#             p = src.get_ppi(config)

#     print("All data ready!")


#######

# core.py (or setup.py, whatever you prefer)

from pathlib import Path
from tfitpy.datasets import DATASETS, FIRST_ORDER
import os

def setup_datasets(data_path):
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