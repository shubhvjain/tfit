from ..utils import download_file, DATA_DIR
import pandas as pd

META = {
    "url": "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt",
    "name": "hippie_ppi.txt",
    "about": "Latest HIPPIE PPI dataset.",
    "columns": ["uniprot_id_1", "entrez_id_1", "uniprot_id_2", "entrez_id_2", "score", "comments"]
}

def is_ready() -> bool:
    return (DATA_DIR / META["name"]).exists()

def download():
    download_file(META["url"], META["name"])

def get() -> pd.DataFrame:
    """Load HIPPIE PPI data (downloads if missing)"""
    if not is_ready():
        download()
    
    filepath = DATA_DIR / META["name"]
    print(f"Loading HIPPIE data from {filepath}...")
    
    df = pd.read_csv(filepath, sep='\t', header=None, names=META["columns"])
    df['entrez_id_1'] = pd.to_numeric(df['entrez_id_1'], errors='coerce').astype('Int64')
    df['entrez_id_2'] = pd.to_numeric(df['entrez_id_2'], errors='coerce').astype('Int64')
    
    return df
