# datasets/geo_datasets.py

import pooch
import pandas as pd
from pathlib import Path
import networkx as nx

from tfitpy.datasets.gene_names import convert_gene_df,load_biomart

####### HIPPIE DB #######

HIPPIE = {
  "FOLDER":"hippie",
  "RAW_FILE":"hippie_current.txt",
  "FINAL_FILE":"hippie_ppi_hgnc.parquet",
  "columns": [
        "uniprot_id_1",
        "entrez_id_1",
        "uniprot_id_2",
        "entrez_id_2",
        "score",
        "comments",
    ],
  "URL": "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt"
}

def download_hippie(data_path,rerun=False):
    """Download HIPPIE PPI dataset dataset"""
    raw_path = Path(data_path) / f"{HIPPIE['FOLDER']}"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    file_path = pooch.retrieve(
        url= HIPPIE['URL'],
        known_hash=None,
        path=raw_path,
        fname= HIPPIE["RAW_FILE"]
    )
    
    print(f"Downloaded hippie PPI DB to {file_path}")
    return file_path

def process_hippie(data_path,rerun=False):
    """Process the HIPPIE dataset """
    raw_file = Path(data_path) / f"{HIPPIE['FOLDER']}" / f"{HIPPIE['RAW_FILE']}" 
    processed_file = Path(data_path) / f"{HIPPIE['FOLDER']}"/ f"{HIPPIE['FINAL_FILE']}" 
    if processed_file.exists() and not rerun:
        print("File already exists: {processed_file}")
        return 
    
    df = pd.read_csv(raw_file, sep="\t", header=None, names=HIPPIE["columns"])
    df["entrez_id_1"] = pd.to_numeric(df["entrez_id_1"], errors="coerce").astype("Int64")
    df["entrez_id_2"] = pd.to_numeric(df["entrez_id_2"], errors="coerce").astype("Int64")
    
    print("building network")
    data_biomart = {"biomart":load_biomart(data_path)}


    db_mapped = convert_gene_df(
        df = df,
        gene_columns_map={"entrez_id_1": "node1", "entrez_id_2": "node2"},
        gene_source="entrez_id",
        target_name="symbol",
        datasets=data_biomart,
    )
    #print(db_mapped)
    edges = db_mapped.dropna(subset=["node1", "node2"]).copy()
    #print(edges)
    edges = edges[edges["node1"] != edges["node2"]]
    #print(edges)
    edges["edge_source"] = "hippie_ppi"
    #print(edges)
    result = edges[["node1", "node2", "score", "comments", "edge_source"]].copy()
    #print(result)
    # Store as parquet
    result.to_parquet(processed_file, index=False)
    print("Done")


def load_hippie(data_path):
    """Load hippie into memory"""
    processed_file = Path(data_path) / f"{HIPPIE['FOLDER']}" / f"{HIPPIE['FINAL_FILE']}" 
    
    if not processed_file.exists():
        raise FileNotFoundError(f" {HIPPIE['FINAL_FILE']} not found. Run setup_datasets() first.")
    
    df =  pd.read_parquet(processed_file)

    G = nx.from_pandas_edgelist(
    df, 
    source='node1', 
    target='node2', 
    edge_attr=['score', 'comments', 'edge_source'])
    return G


PPI_DATASETS = {
    'hippie': {
        'download': download_hippie,
        'process': process_hippie,
        'load': load_hippie
    }
}