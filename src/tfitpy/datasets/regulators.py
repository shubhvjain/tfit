"""
List of transcription factors
"""
# datasets/ppi

import pooch
import pandas as pd
from pathlib import Path
from tfitpy.datasets.go import get_gene_products

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
    # print(file)
    if not file.exists():
        raise FileNotFoundError(f" {TFLIST['FILE']} not found. Run setup_datasets() first.")
    
    df =  pd.read_csv(file, names =  ["gene_name"], header = None, )

    return df

COREG_LIST = {
  "about":"The list of genes related to a GO term",
  "FOLDER":"tflist",
  "FILE":"human_coreg.csv",
  "term":"GO:0003712",
}

def download_coreglist(data_path,rerun=False):
    return 

def process_coreglist(data_path,rerun=False):
    """Process the tflist dataset """
    file_path = Path(data_path) / f"{COREG_LIST['FOLDER']}" / f"{COREG_LIST['FILE']}" 
    df = get_gene_products(data_path,COREG_LIST["term"])
    df.to_csv(file_path,index=False)
    return

def load_coreglist(data_path):
    """Load hippie into memory"""
    file = Path(data_path) / f"{COREG_LIST['FOLDER']}" / f"{COREG_LIST['FILE']}" 
    # print(file)
    if not file.exists():
        raise FileNotFoundError(f" {COREG_LIST['FILE']} not found. Run setup_datasets() first.")

    df = pd.read_csv(file)
    return df


TFLIST_PLANT = {
  "FOLDER":"tflist",
  "FILE":"Ath_TF_list.txt.gz",
  "URL": "https://planttfdb.gao-lab.org/download/TF_list/Ath_TF_list.txt.gz"
}


def download_tflist_plant(data_path,rerun=False):
    """Download dataset dataset"""
    raw_path = Path(data_path) / f"{TFLIST_PLANT['FOLDER']}"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    file_path = pooch.retrieve(
        url= TFLIST_PLANT['URL'],
        known_hash=None,
        path=raw_path,
        fname= TFLIST_PLANT["FILE"]
    )
    
    print(f"Downloaded  DB to {file_path}")
    return file_path

def process_tflist_plant(data_path,rerun=False):
    """Process the tflist dataset """
    return


def load_tflist_plant(data_path):
    """Load hippie into memory"""
    file = Path(data_path) / f"{TFLIST_PLANT['FOLDER']}" / f"{TFLIST_PLANT['FILE']}" 
    # print(file)
    if not file.exists():
        raise FileNotFoundError(f" {TFLIST_PLANT['FILE']} not found. Run setup_datasets() first.")
    
    df = pd.read_csv(file, delimiter="\t")
    return df


def get_regulator_list(data_path,organism="human"):
    """"""
    if organism == "human":
        #print()
        tf = load_tflist(data_path)
        tf_list = tf["gene_name"].tolist()
        co = load_coreglist(data_path)
        co_list = co["symbol"].tolist()
        all_regulators = list(set(tf_list) | set(co_list))
        return all_regulators
    elif organism == "arabidopsis":
        #print()
        tf = load_tflist_plant(data_path)
        tf_list = tf["Gene_ID"].tolist() 
        return tf_list
    else:
        raise ValueError("unknown organism")

def get_regulator_list_plant(data_path):
    """"""
    tf = load_tflist_plant(data_path)
    tf_list = tf["Gene_ID"].tolist() 
    return tf_list


TF_DATASET = {
    'tflist': {
        'download': download_tflist,
        'process': process_tflist,
        'load': load_tflist
    },
    'coreglist': {
        'download': download_coreglist,
        'process': process_coreglist,
        'load': load_coreglist
    },
    'tflist_plant':{
        'download': download_tflist_plant,
        'process': process_tflist_plant,
        'load': get_regulator_list_plant
    },
    'tflist_human':{
        'download': download_tflist,
        'process': process_tflist,
        'load': get_regulator_list
    }
}