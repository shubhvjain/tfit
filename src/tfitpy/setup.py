from typing import List
from pathlib import Path
from typing import Any, Dict, Optional

import tfitpy.data.hippie as hippie
import tfitpy.data.biomart as biomart
import tfitpy.data.stringdb as stringdb
import tfitpy.data.biogrid as biogrid


DATA_SOURCES = [
    hippie,
    biomart,
    stringdb,
    biogrid
]

PPI_BUILD = [
    hippie
]


def ensure_all_data(config: Optional[Dict[str, Any]] = None) -> None:
    """Download all required data sources using the provided config.

    Args:
        config: Global config dict (may be {} or None). Passed to each
            data source's download() function.
    """
    missing = [src for src in DATA_SOURCES if not src.is_ready(config)]
    
    if  missing:
        print(f"Downloading {len(missing)} missing sources...")
        for src in missing:
        #src_name = src.__name__.split('.')[-1]  # "hippie", "biomart", etc.
        #print(f"Downloading {src_name}...")
            src.download(config)

    ppi_missing = [src for src in PPI_BUILD if not src.is_ready(config,"ppi_hgnc_filename")]

    if  ppi_missing:
        print(f"Generating PPI for {len(ppi_missing)} missing sources...")
        for src in ppi_missing:
        #src_name = src.__name__.split('.')[-1]  # "hippie", "biomart", etc.
        #print(f"Downloading {src_name}...")
            p = src.get_ppi(config)

    print("All data ready!")


