"""Utility functions """
import pandas as pd
from pathlib import Path
import os
import sqlite3

def generate_tf_pairs(gene_cluster):
    """Generate all unique pairs from comma-separated gene cluster."""
    genes = [g.strip() for g in gene_cluster]
    return [(g1, g2) for i, g1 in enumerate(genes) for g2 in genes[i+1:]]

ORGANISM_METADATA = {
    "human":{ 
        "ppi_keys":["hippie","stringdb","biogrid"],
        "go_key":"go",
        "pair_cache":"pairwise_score_cache_human",
        "pair_cache_trap":"trap_cache_human",
        "jaspar":"jaspar",
        "source_background_list":"tflist_human"
    },
    "arabidopsis":{
        "ppi_keys":["stringdb_arabidopsis"],
        "go_key":"go_arabidopsis",
        "pair_cache":"pairwise_score_cache_arabidopsis",
        "jaspar":"jaspar_plant",
        "gene_mapping": "plant_gene_mapping",
        "source_background_list":"tflist_plant"
    }
}
INDICES_DATA = {
    "ppi_cached_indices" : ["shortest_PPI_path_score","shared_PPI_partners_score"],
    "ppi_network_indices" : ["density","density_score","lcc","lcc_score","tc","tc_score","node_found_ratio"],
    "ppi_source_indices":["ppi_edges"],
    "go_cached_indices":["goa_similarity_lin","goa_similarity_resnik","goa_similarity_jc"],
    "go_source_indices":["go_ora"],
    
}


def get_mappings(CONFIG, gene_list, source, target, batch_size=900):
    """
    Map gene identifiers from source format to target format.
    Assuming tfitpy setup was run and the source db file is available.
    Args:
        gene_list: List of identifiers to map
        source: Source attribute name in database (e.g., 'gene_name', 'gene_id')
        target: Target attribute name in database (e.g., 'gene_id', 'gene_name')
        batch_size: Number of items to process per query (default 900, under SQLite's 999 limit)
    Returns:
        Dictionary mapping {source_value: target_value, ...}
    """
    db_path = Path( os.path.expandvars(  CONFIG["DATA_PATH"])) /"gencode"/"gene_name_mapping.db"
    #print(db_path)
    #print(db_path.exists())
    con = sqlite3.connect(db_path)

    try:
        # Remove None/NaN and get unique values
        unique_values = [x for x in set(
            gene_list) if x is not None and pd.notna(x)]
        if len(unique_values) == 0:
            print("No values to map")
            return {}

        # Process in batches to avoid SQLite variable limit
        mapping_dict = {}
        for i in range(0, len(unique_values), batch_size):
            batch = unique_values[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f"""
                SELECT DISTINCT {source}, {target}
                FROM mappings
                WHERE {source} IN ({placeholders})
                AND {target} IS NOT NULL
            """

            # Execute query and update mapping dict
            mapping_df = pd.read_sql_query(query, con, params=batch)
            mapping_dict.update(
                dict(zip(mapping_df[source], mapping_df[target])))

        # Report statistics
        total = len(gene_list)
        unique_count = len(unique_values)
        mapped_count = len(mapping_dict)
        failed_count = unique_count - mapped_count
        null_in_results = sum(
            1 for gene in gene_list if gene not in mapping_dict)

        print(f"Mapping from {source} to {target}:")
        print(f"  Total input values: {total}")
        print(f"  Unique values: {unique_count}")
        print(f"  Successfully mapped: {mapped_count}")
        print(f"  Failed to map: {failed_count}")
        print(
            f"  Null results: {null_in_results} ({null_in_results/total*100:.2f}%)")

        return mapping_dict

    finally:
        con.close()


def read_dataset(details, config):
    """"""
    dtype = details.get("type", None)
    if dtype is None:
        raise ValueError("invalid dataset.type")

    #print(config.get("data_path"))
    if dtype == "gct":
        return read_gct(details.get("path"), config, convert_gene_names=details.get("convert_gene_names", True),transpose= details.get("transpose", True))
    elif dtype == "tcga":
        return read_tcga(
            file_path=details.get("path"), 
            CONFIG=config, 
            convert_gene_names=details.get("convert_gene_names", True)
        )
    elif dtype == "csv":
        fpath = Path(os.path.expandvars(details.get("path")))
        df  = pd.read_csv(fpath,index_col=0)
        transpose_data = details.get("transpose",True)
        if transpose_data:
            df = df.T
        return df


def read_gct(file_path, CONFIG=None, convert_gene_names=False,transpose=True):
    """
    Read Gene Cluster Text (GCT) format file into a pandas DataFrame.

    GCT is a tab-delimited format which include
    - Line 1: Version information
    - Line 2: Dimensions (genes x samples)  
    - Line 3+: Header with Name, Description, and sample columns
    - Data rows: Gene information and expression values 

    Assuming there is  "Description" column that has the name of genes have the gene names for each gene row.

    Args:
        file_path (str or Path) : Path to the GCT file.

    Returns:
        pd.DataFrame :  DataFrame with genes as rows and samples as columns. The index is gene_name (from Description column), columns are sample identifiers. Each cell had gene expression levels

    Notes:
    ------
    - Removes the 'Name' column (gene IDs) and uses 'Description' as gene names

    """

    file_path = Path(os.path.expandvars(file_path))
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read GCT file, skipping version and dimension lines
        df = pd.read_csv(file_path, skiprows=2, sep="\t")
        gene = df["Name"].values.tolist()
        if convert_gene_names:
            if CONFIG is None:
                raise ValueError("data_path not provided")
            mps = get_mappings(CONFIG, gene_list=gene, source='gene_id',target='gene_name')
            df["Name"] = df["Name"].map(mps)
        if "Name" not in df.columns or "Description" not in df.columns:
            raise ValueError(
                "GCT file must contain 'Name' and 'Description' columns")

        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(
            columns={"Description": "gene_name"})
        df = df.set_index("gene_name")
        if transpose :
            df = df.transpose().rename_axis("sample_name")
        return df
    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")


def read_tcga(file_path, CONFIG=None, convert_gene_names=False):
    """
    Read TCGA expression CSV file into a pandas DataFrame.
    """
    file_path = Path(os.path.expandvars(file_path))
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load CSV with the gene IDs as the index
        df = pd.read_csv(file_path, index_col=0)

        if convert_gene_names:
            if CONFIG is None:
                raise ValueError("data_path not provided")
            
            gene = df.index.tolist()
            mps = get_mappings(CONFIG, gene_list=gene, source='gene_id', target='gene_name')
            
            # Map the index using a list comprehension to safely handle fallbacks without pandas Index bugs
            df.index = [mps.get(g, g) for g in gene]

        df.index.name = "gene_name"
        df = df.transpose().rename_axis("sample_name")
        return df
    except Exception as e:
        raise ValueError(f"Error reading TCGA file {file_path}: {str(e)}")
    

def get_valid_gene_expression(gene_expression, gene_list=None):
    """
    Cleans gene_expression for use with dcorr cache functions:
    - dedupes columns (keeps first occurrence)
    - restricts to gene_list if provided (only genes present in columns)
    - drops zero-variance genes (undefined dCor)
    Returns (cleaned_df, dropped_genes)
    """
    ge = gene_expression.loc[:, ~gene_expression.columns.duplicated()]

    if gene_list is not None:
        valid = [g for g in gene_list if g in ge.columns]
        ge = ge[valid]

    variances = ge.var()
    zero_var_genes = variances[variances == 0].index.tolist()

    if zero_var_genes:
        ge = ge.drop(columns=zero_var_genes)

    return ge