from typing import List
from tfit.data import (
    hippie,biomart,stringdb 
)
from .utils import DATA_DIR

DATA_SOURCES = [
    hippie,
    biomart,
    stringdb
]

def ensure_all_data():
    """Download all required data sources"""
    missing = [src for src in DATA_SOURCES if not src.is_ready()]
    if not missing:
        print("All data sources ready!")
        return
    
    print(f"Downloading {len(missing)} missing sources...")
    for src in missing:
        src.download()
    
    print(f"All data ready at {DATA_DIR}")
