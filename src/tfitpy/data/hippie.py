from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd

from tfitpy.utils import download_file, resolve_module_config


META: Dict[str, Any] = {
    "url": "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt",
    "about": "Latest HIPPIE PPI dataset.",
    "columns": [
        "uniprot_id_1",
        "entrez_id_1",
        "uniprot_id_2",
        "entrez_id_2",
        "score",
        "comments",
    ],
    "filename": "hippie_ppi.txt",  # module default config
}

def _file_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Return the full path to the HIPPIE data file."""
    cfg = resolve_module_config(config, "hippie", META)
    return cfg["data_path"] / cfg["hippie"]["filename"]


def is_ready(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if the HIPPIE file already exists."""
    return _file_path(config).exists()


def download(config: Optional[Dict[str, Any]] = None) -> None:
    """Download the HIPPIE file if missing."""
    cfg = resolve_module_config(config, "hippie", META)
    download_file(
        META["url"],
        cfg["hippie"]["filename"],
        base_dir=cfg["data_path"],
    )


def get(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load HIPPIE PPI data, downloading it first if necessary.

    Args:
        config: Global config dict (may be {} or None).

    Returns:
        pandas.DataFrame: Raw HIPPIE PPI interactions.
    """
    if not is_ready(config):
        download(config)

    filepath = _file_path(config)
    print(f"Loading HIPPIE data from {filepath}...")

    df = pd.read_csv(filepath, sep="\t", header=None, names=META["columns"])
    df["entrez_id_1"] = pd.to_numeric(df["entrez_id_1"], errors="coerce").astype("Int64")
    df["entrez_id_2"] = pd.to_numeric(df["entrez_id_2"], errors="coerce").astype("Int64")
    
    return df

def _get_edges_within_sources(db: pd.DataFrame, sources: List[int]) -> pd.DataFrame:
    """Get all edges between genes in sources list (uses Entrez IDs)."""
    sources_set = set(sources)
    
    mask_forward = (
        db['entrez_id_1'].isin(sources_set) &
        db['entrez_id_2'].isin(sources_set)
    )
    
    edges = db[mask_forward].copy()
    return edges

def _get_edges_sources_to_target(db: pd.DataFrame, sources: List[int], target: int) -> pd.DataFrame:
    """Get edges from sources to target (uses Entrez IDs)."""
    sources_set = set(sources)
    
    # Forward direction: source -> target
    mask_forward = (
        db['entrez_id_1'].isin(sources_set) &
        (db['entrez_id_2'] == target)
    )
    
    # Reverse direction: target -> source
    mask_reverse = (
        (db['entrez_id_1'] == target) &
        db['entrez_id_2'].isin(sources_set)
    )
    
    edges = db[mask_forward | mask_reverse].copy()
    return edges

def get_edges(
    config: Optional[Dict[str, Any]] = None,
    sources: List[str] = None,
    target: Optional[str] = None,
    db: Optional[pd.DataFrame] = None,
    gene_mapping: Optional[pd.DataFrame] = None,
    include_type: bool = True,
) -> pd.DataFrame:
    """Get edges from HIPPIE database between sources and optionally to a target.
    
    Args:
        config: Global config dict for data locations
        sources: List of gene symbols (e.g., ['TP53', 'BRCA1'])
        target: Optional target gene symbol
        db: Pre-loaded HIPPIE DataFrame (if None, will load via config)
        gene_mapping: Pre-loaded BioMart gene mapping DataFrame 
            (if None, will load via config)
        include_type: Whether to add 'edge_type' column
    
    Returns:
        DataFrame with columns: node1, node2, score, comments, [edge_type], edge_source
    """
    # Load db if not provided
    if db is None:
        db = get(config)

    if sources is None :
        raise Exception("No source provided")

    # Note: gene_mapping loading would need similar config support
    # For now, assuming it's loaded separately or passed in
    if gene_mapping is None:
        raise ValueError("gene_mapping must be provided or loaded separately")
    
    # Create symbol to Entrez ID mapping
    symbol_to_entrez = (
        gene_mapping.dropna(subset=['entrez_id'])
        .set_index('symbol')['entrez_id']
        .to_dict()
    )
    
    # Convert sources to Entrez IDs
    source_entrez_ids = []
    for gene in sources:
        entrez_id = symbol_to_entrez.get(gene)
        if entrez_id and pd.notna(entrez_id):
            source_entrez_ids.append(int(entrez_id))
        else:
            print(f"Warning: Could not find Entrez ID for gene '{gene}'")
    
    # Convert target to Entrez ID
    target_entrez_id = None
    if target:
        target_entrez_id = symbol_to_entrez.get(target)
        if target_entrez_id and pd.notna(target_entrez_id):
            target_entrez_id = int(target_entrez_id)
        else:
            print(f"Warning: Could not find Entrez ID for target gene '{target}'")
    
    if len(source_entrez_ids) == 0:
        print("No valid Entrez IDs found for source genes")
        cols = ['node1', 'node2', 'score', 'comments']
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    all_edges = []
    
    # Get within-sources edges
    within_edges = _get_edges_within_sources(db, source_entrez_ids)
    if len(within_edges) > 0:
        if include_type:
            within_edges = within_edges.copy()
            within_edges['edge_type'] = 'within_cluster'
        all_edges.append(within_edges)
    
    # Get sources-to-target edges
    if target_entrez_id:
        target_edges = _get_edges_sources_to_target(db, source_entrez_ids, target_entrez_id)
        if len(target_edges) > 0:
            if include_type:
                target_edges = target_edges.copy()
                target_edges['edge_type'] = 'to_target'
            all_edges.append(target_edges)
    
    if len(all_edges) == 0:
        cols = ['node1', 'node2', 'score', 'comments']
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    edges = pd.concat(all_edges, ignore_index=True)
    
    # Rename columns
    edges = edges.rename(columns={
        'entrez_id_1': 'node1_entrez',
        'entrez_id_2': 'node2_entrez'
    })
    
    # Convert Entrez IDs back to gene symbols
    entrez_to_symbol = (
        gene_mapping.dropna(subset=['entrez_id'])
        .set_index('entrez_id')['symbol']
        .to_dict()
    )
    
    edges['node1'] = edges['node1_entrez'].map(entrez_to_symbol).fillna(edges['node1_entrez'].astype(str))
    edges['node2'] = edges['node2_entrez'].map(entrez_to_symbol).fillna(edges['node2_entrez'].astype(str))
    
    # Add edge source
    edges["edge_source"] = "hippie_ppi"
    
    # Select final columns
    result_cols = ['node1', 'node2', 'score', 'comments', 'edge_source']
    if include_type:
        result_cols.insert(-1, 'edge_type')  # insert before edge_source
    
    return edges[result_cols]
