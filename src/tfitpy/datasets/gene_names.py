from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import pooch
import sqlite3


BIOMART = {
    "URL": "http://www.ensembl.org/biomart/martservice?query=<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE Query><Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' datasetConfigVersion='0.6'><Dataset name='hsapiens_gene_ensembl' interface='default'><Attribute name='ensembl_gene_id'/><Attribute name='external_gene_name'/><Attribute name='entrezgene_id'/><Attribute name='uniprotswissprot'/><Attribute name='refseq_mrna'/><Attribute name='description'/></Dataset></Query>",
    "FOLDER":"biomart",
    "RAW_FILE": "biomart_gene_mapping.txt",
    "FINAL_FILE":"biomart_gene_mappings.db",
    "COLUMNS": ['ensembl_gene_id', 'symbol', 'entrez_id', 'uniprot_id', 'refseq_id', 'description'],    
}

def download_biomart(data_path, rerun=False):
    """Download biomart dataset"""
    raw_path = Path(data_path) / BIOMART['FOLDER']
    raw_path.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_path / BIOMART['RAW_FILE']
    
    # Check if already downloaded
    if output_file.exists() and not rerun:
        print(f"Biomart already downloaded: {output_file}")
        return output_file
    
    # Download with pooch and specify the filename
    file_path = pooch.retrieve(
        url=BIOMART['URL'],
        known_hash=None,
        path=raw_path,
        fname=BIOMART['RAW_FILE']  # Pooch will save with this name
    )
    
    print(f"Downloaded biomart to {file_path}")
    return file_path

def process_biomart(data_path, rerun=False):
    """Process the biomart data and generate sqlite db file for easy mapping"""
    raw_file = Path(data_path) / BIOMART['FOLDER'] / BIOMART['RAW_FILE']
    processed_file = Path(data_path) / BIOMART['FOLDER'] / BIOMART['FINAL_FILE']
    
    if processed_file.exists() and not rerun:
        print(f"File already exists: {processed_file}")
        return
    
    # Read the raw file
    print(f"Reading {raw_file}...")
    df = pd.read_csv(raw_file, sep='\t')
    df.columns = BIOMART['COLUMNS']
    
    # Process the data
    df['ensembl_gene_id'] = '9606.' + df['ensembl_gene_id'].astype(str)
    df['entrez_id'] = pd.to_numeric(df['entrez_id'], errors='coerce').astype('Int64')
    
    # Create SQLite database
    print(f"Creating SQLite database at {processed_file}...")
    con = sqlite3.connect(processed_file)
    
    # Write dataframe to SQL table
    df.to_sql('gene_mappings', con, if_exists='replace', index=False)
    
    # Create indexes for fast lookup on common columns
    cursor = con.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ensembl ON gene_mappings(ensembl_gene_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON gene_mappings(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entrez ON gene_mappings(entrez_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_uniprot ON gene_mappings(uniprot_id)')
    con.commit()
    
    con.close()
    print(f"SQLite database created with {len(df)} rows and indexes")

def load_biomart(data_path):
    """Load hippie into memory"""
    processed_file = Path(data_path) / f"{BIOMART['FOLDER']}" / f"{BIOMART['FINAL_FILE']}" 
    
    if not processed_file.exists():
        raise FileNotFoundError(f" {HIPPIE['FINAL_FILE']} not found. Run setup_datasets() first.")
    
    con = sqlite3.connect(processed_file)
    return con
 
def convert_genes(
    input: List[str],
    input_type: str = "symbol",
    output_type: str = "ensembl_gene_id",
    datasets: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert gene identifiers from one type to another using BioMart data.
    
    Args:
        input: List of gene identifiers to convert
        input_type: Source identifier type
        output_type: Target identifier type
        datasets: Cache containing loaded datasets (must include 'biomart')
    
    Returns:
        dict: Mapping of input identifiers to output identifiers
    """
    if datasets is None:
        raise ValueError("datasets cache is required. Create cache with load_datasets() first.")
    
    if input is None:
        raise ValueError("No input provided")
    
    biomart_con = datasets['biomart']
    
    # Remove None/NaN and get unique values
    unique_values = [x for x in set(input) if x is not None and pd.notna(x)]
    
    if len(unique_values) == 0:
        print("No values to map")
        return {}
    
    # Create query
    placeholders = ','.join(['?'] * len(unique_values))
    query = f"""
        SELECT DISTINCT {input_type}, {output_type}
        FROM gene_mappings
        WHERE {input_type} IN ({placeholders})
        AND {output_type} IS NOT NULL
    """
    
    # Execute query and create mapping dict
    mapping_df = pd.read_sql_query(query, biomart_con, params=unique_values)
    mapping_dict = dict(zip(mapping_df[input_type], mapping_df[output_type]))
    
    # Report statistics
    total = len(input)
    unique_count = len(unique_values)
    mapped_count = len(mapping_dict)
    failed_count = unique_count - mapped_count
    null_in_results = sum(1 for gene in input if gene not in mapping_dict)
    
    print(f"Mapping from {input_type} to {output_type}:")
    print(f"  Total input values: {total}")
    print(f"  Unique values: {unique_count}")
    print(f"  Successfully mapped: {mapped_count}")
    print(f"  Failed to map: {failed_count}")
    print(f"  Null results: {null_in_results} ({null_in_results/total*100:.2f}%)")
    
    return mapping_dict

def convert_gene_df(
    df: pd.DataFrame,
    gene_columns_map: Dict[str, str],
    gene_source: str,
    target_name: str,
    datasets: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Bulk conversion of multiple gene ID columns using single SQLite query.
    
    Args:
        df: Input DataFrame with gene identifier columns
        gene_columns_map: Mapping of source columns to new column names
        gene_source: Type of gene ID in the source columns
        target_name: Type of gene ID to convert to
        datasets: Cache containing loaded datasets (must include 'biomart')
    
    Returns:
        DataFrame with new columns added containing converted gene IDs
    """
    if datasets is None:
        raise ValueError("datasets cache is required. Create cache with load_datasets() first.")
    
    biomart_con = datasets['biomart']
    
    # Validate columns exist
    for src_col in gene_columns_map.keys():
        if src_col not in df.columns:
            raise KeyError(f"Column '{src_col}' not in DataFrame")
    
    # Collect ALL unique gene IDs from all columns
    all_genes = set()
    for src_col in gene_columns_map.keys():
        all_genes.update(df[src_col].dropna().astype(str).unique())
    
    # Remove None/NaN
    unique_values = [x for x in all_genes if x is not None and pd.notna(x)]
    
    if len(unique_values) == 0:
        print("No values to map")
        return df.copy()
    
    # Single bulk query
    placeholders = ','.join(['?'] * len(unique_values))
    query = f"""
        SELECT DISTINCT {gene_source}, {target_name}
        FROM gene_mappings
        WHERE {gene_source} IN ({placeholders})
        AND {target_name} IS NOT NULL
    """
    
    # Execute query and create mapping dict
    mapping_df = pd.read_sql_query(query, biomart_con, params=unique_values)
    gene_map = dict(zip(mapping_df[gene_source], mapping_df[target_name]))
    
    # Report statistics
    mapped_count = len(gene_map)
    print(f"Bulk mapping from {gene_source} to {target_name}:")
    print(f"  Unique input values: {len(unique_values)}")
    print(f"  Successfully mapped: {mapped_count}")
    print(f"  Failed to map: {len(unique_values) - mapped_count}")
    
    # Apply mapping to all columns
    out = df.copy()
    for src_col, new_col in gene_columns_map.items():
        out[new_col] = df[src_col].map(gene_map)
    return out


GENE_DATASETS = {
    'biomart': {
        'download': download_biomart,
        'process': process_biomart,
        'load': load_biomart
    }
}
