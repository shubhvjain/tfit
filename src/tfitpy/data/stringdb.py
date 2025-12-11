from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd

from tfitpy.utils import download_file, resolve_module_config


META: Dict[str, Any] = {
    "about": "STRING v12 protein-protein interactions and protein info for human (9606)",
    "ppi": {
        "url": "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz",
        "filename": "string_ppi.txt.gz",
    },
    "protein": {
        "url": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "filename": "string_protein.txt.gz",
    },
}


def _ppi_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Return the full path to the STRING PPI file."""
    cfg = resolve_module_config(config, "stringdb", META)
    return cfg["data_path"] / cfg["stringdb"]["ppi"]["filename"]


def _protein_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Return the full path to the STRING protein info file."""
    cfg = resolve_module_config(config, "stringdb", META)
    return cfg["data_path"] / cfg["stringdb"]["protein"]["filename"]


def is_ready(config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if STRING data (both PPI and protein info) is ready."""
    return _ppi_path(config).exists() and _protein_path(config).exists()


def download(config: Optional[Dict[str, Any]] = None) -> None:
    """Download both STRING PPI and protein info if missing."""
    cfg = resolve_module_config(config, "stringdb", META)
    
    download_file(
        META["ppi"]["url"],
        cfg["stringdb"]["ppi"]["filename"],
        base_dir=cfg["data_path"],
    )
    download_file(
        META["protein"]["url"],
        cfg["stringdb"]["protein"]["filename"],
        base_dir=cfg["data_path"],
    )


def get_protein_info(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load STRING protein info (downloads if missing).

    Args:
        config: Global config dict (may be {} or None).

    Returns:
        pd.DataFrame: STRING protein info DataFrame.
    """
    if not _protein_path(config).exists():
        download(config)

    filepath = _protein_path(config)
    print(f"Loading STRING protein info from {filepath}...")
    return pd.read_csv(filepath, sep="\t", compression='gzip')


def get(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load STRING PPI data + protein info (downloads if missing).
    
    Args:
        config: Global config dict (may be {} or None).

    Returns:
        tuple: (ppi_df, protein_info_df)
    """
    if not is_ready(config):
        download(config)
    
    ppi_path = _ppi_path(config)
    protein_path = _protein_path(config)
    
    print(f"Loading STRING PPI from {ppi_path}...")
    df_ppi = pd.read_csv(ppi_path, sep=" ", compression='gzip')
    
    #print(f"Loading STRING protein info from {protein_path}...")
    # df_protein = pd.read_csv(protein_path, sep="\t", compression='gzip')
    
    return df_ppi 



def get_edges(
    config: Optional[Dict[str, Any]] = None,
    sources: List[str]= [],
    target: Optional[str] = None,
    db: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    include_type: bool = True,
    score_columns: Optional[List[str]] = ["combined_score"],
) -> pd.DataFrame:
    """
    Get edges from STRING database between sources and optionally to a target.
    
    Args:
        config: Global config dict for data locations
        sources: List of gene symbols (e.g., ['TP53', 'BRCA1'])
        target: Optional target gene symbol
        db: Pre-loaded STRING DataFrames tuple (ppi_df, info_df). 
            If None, will load via config
        include_type: Whether to add 'edge_type' column
        score_columns: List of score columns to include. 
            Default: ['combined_score']
    
    Returns:
        DataFrame with columns: node1, node2, [score columns], [edge_type], edge_source
    """
    if sources is None :
        raise Exception("No source provided")
    
    # Load db if not provided
    if db is None:
        db_ppi = get(config)
    

    db_info = get_protein_info(config)    
    # Map gene symbols to STRING protein IDs (ENSP format)
    symbol_to_protein = db_info.set_index('preferred_name')['#string_protein_id'].to_dict()
    protein_to_symbol = db_info.set_index('#string_protein_id')['preferred_name'].to_dict()
    
    # Convert sources to STRING protein IDs
    sources_proteins = []
    for gene in sources:
        protein_id = symbol_to_protein.get(gene)
        if protein_id and pd.notna(protein_id):
            sources_proteins.append(protein_id)
        else:
            print(f"Warning: Could not find STRING protein ID for gene '{gene}'")
    
    # Convert target if provided
    target_protein = None
    if target:
        target_protein = symbol_to_protein.get(target)
        if not target_protein or pd.isna(target_protein):
            print(f"Warning: Could not find STRING protein ID for target '{target}'")
    
    if len(sources_proteins) == 0:
        cols = ['node1', 'node2'] + score_columns
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    sources_set = set(sources_proteins)
    all_edges = []
    
    # Get within-sources edges
    within_mask = (
        db_ppi['protein1'].isin(sources_set) &
        db_ppi['protein2'].isin(sources_set)
    )
    within_edges = db_ppi[within_mask].copy()
    if len(within_edges) > 0:
        if include_type:
            within_edges['edge_type'] = 'within_cluster'
        all_edges.append(within_edges)
    
    # Get sources-to-target edges
    if target_protein:
        target_mask = (
            (db_ppi['protein1'].isin(sources_set) & (db_ppi['protein2'] == target_protein)) |
            ((db_ppi['protein1'] == target_protein) & db_ppi['protein2'].isin(sources_set))
        )
        target_edges = db_ppi[target_mask].copy()
        if len(target_edges) > 0:
            if include_type:
                target_edges['edge_type'] = 'to_target'
            all_edges.append(target_edges)
    
    if len(all_edges) == 0:
        cols = ['node1', 'node2'] + score_columns
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    edges = pd.concat(all_edges, ignore_index=True)
    
    # Convert STRING protein IDs back to gene symbols
    edges['node1'] = edges['protein1'].map(protein_to_symbol).fillna(edges['protein1'])
    edges['node2'] = edges['protein2'].map(protein_to_symbol).fillna(edges['protein2'])
    
    edges['edge_source'] = 'string_ppi'
    
    # Select result columns
    result_cols = ['node1', 'node2'] + score_columns
    if include_type:
        result_cols.append('edge_type')
    result_cols.append('edge_source')
    
    return edges[result_cols]
