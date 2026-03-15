"""
List of transcription factors
"""
# datasets/ppi

import pooch
import pandas as pd
from pathlib import Path


TFLIST = {
  "FOLDER":"tflist",
  "FILE":"allTFs_hg38.txt",
  "URL": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt"
}

def download_tflist(data_path,rerun=False):
    """Download dataset dataset"""
    raw_path = Path(data_path) / f"{TFLIST['FOLDER']}"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    file_path = pooch.retrieve(
        url= TFLIST['URL'],
        known_hash=None,
        path=raw_path,
        fname= TFLIST["FILE"]
    )
    
    print(f"Downloaded  DB to {file_path}")
    return file_path

def process_tflist(data_path,rerun=False):
    """Process the tflist dataset """
    return


def load_tflist(data_path):
    """Load hippie into memory"""
    file = Path(data_path) / f"{TFLIST['FOLDER']}" / f"{TFLIST['FILE']}" 
    print(file)
    if not file.exists():
        raise FileNotFoundError(f" {TFLIST['FILE']} not found. Run setup_datasets() first.")
    
    df =  pd.read_csv(file, names =  ["gene_name"], header = None, )

    return df

TF_DATASET = {
    'tflist': {
        'download': download_tflist,
        'process': process_tflist,
        'load': load_tflist
    },
}