from pathlib import Path
from typing import Any, Dict, Optional,List
import pandas as pd

from tfitpy.utils import download_file, resolve_module_config

META: Dict[str, Any] = {
    "url": "http://www.ensembl.org/biomart/martservice?query=<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE Query><Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' datasetConfigVersion='0.6'><Dataset name='hsapiens_gene_ensembl' interface='default'><Attribute name='ensembl_gene_id'/><Attribute name='external_gene_name'/><Attribute name='entrezgene_id'/><Attribute name='uniprotswissprot'/><Attribute name='refseq_mrna'/><Attribute name='description'/></Dataset></Query>",
    "about": "BioMart gene mappings for human genes including Ensembl IDs, gene symbols (HGNC), Entrez Gene IDs, UniProt IDs, and RefSeq accessions.",
    "columns": ['ensembl_gene_id', 'symbol', 'entrez_id', 'uniprot_id', 'refseq_id', 'description'],
    "filename": "biomart_gene_mapping.txt", 
}


def _file_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Return the full path to the BioMart data file."""
    cfg = resolve_module_config(config, "biomart", META)
    return cfg["data_path"] / cfg["biomart"]["filename"]


def is_ready(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if the BioMart file already exists."""
    return _file_path(config).exists()


def download(config: Optional[Dict[str, Any]] = None) -> None:
    """Download the BioMart file if missing."""
    cfg = resolve_module_config(config, "biomart", META)
    download_file(
        META["url"],
        cfg["biomart"]["filename"],
        base_dir=cfg["data_path"],
    )

def get(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load BioMart gene mapping data (downloads if missing).
    
    This includes mappings between:
    - Ensembl Gene IDs
    - HGNC gene symbols
    - Entrez Gene IDs (NCBI)
    - UniProt/Swiss-Prot IDs
    - RefSeq mRNA accessions
    - Gene descriptions
    
    Args:
        config: Global config dict (may be {} or None).

    Returns:
        pandas DataFrame with gene ID mappings
    """
    if not is_ready(config):
        download(config)

    filepath = _file_path(config)
    print(f"Loading BioMart data from {filepath}...")
    
    df = pd.read_csv(filepath, sep='\t')
    df.columns = META["columns"]
    df['ensembl_gene_id'] = '9606.' + df['ensembl_gene_id'].astype(str)
    df['entrez_id'] = pd.to_numeric(df['entrez_id'], errors='coerce').astype('Int64')
    
    return df



def convert_genes(
    config: Optional[Dict[str, Any]] = None,
    input: List[str] = None,
    input_type: str = "symbol",
    output_type: str = "ensembl_gene_id",
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Convert gene identifiers from one type to another using BioMart data.
    
    Args:
        config: Global config dict for data locations
        input: List of gene identifiers to convert
        input_type: Source identifier type. Options:
            - 'symbol': HGNC gene symbol (e.g., 'TP53')
            - 'ensembl_gene_id': Ensembl gene ID (e.g., 'ENSG00000141510')
            - 'entrez_id': NCBI Entrez Gene ID (e.g., '7157')
            - 'uniprot_id': UniProt/Swiss-Prot ID
            - 'refseq_id': RefSeq mRNA accession
        output_type: Target identifier type (same options as input_type)
        data: Pre-loaded BioMart DataFrame. If None, will load via config
    
    Returns:
        dict: Mapping of input identifiers to output identifiers
              Returns None for identifiers that couldn't be mapped
    """
    # Load data if not provided
    if input is None :
        raise Exception("No input provided")

    if data is None:
        data = get(config)

    results = {}
    
    for gene_id in input:
        # Find matching rows
        mask = data[input_type].astype(str) == str(gene_id)
        matches = data[mask]
        
        if len(matches) > 0:
            # Take first match
            result = matches.iloc[0][output_type]
            # Handle NaN values
            results[gene_id] = result if pd.notna(result) else None
        else:
            results[gene_id] = None
    
    return results
