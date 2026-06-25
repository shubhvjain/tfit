# datasets/ppi

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

####### STRING DB #######
STRINGDB = {
    "FOLDER": "stringdb",
    "RAW_FILE": "9606.protein.links.detailed.v12.0.txt.gz",
    "INFO_FILE": "9606.protein.info.v12.0.txt.gz",
    "FINAL_FILE": "stringdb_ppi_hgnc.parquet",
    "columns": [
        "protein1",
        "protein2",
        "neighborhood",
        "neighborhood_transferred",
        "fusion",
        "cooccurence",
        "homology",
        "coexpression",
        "coexpression_transferred",
        "experiments",
        "experiments_transferred",
        "database",
        "database_transferred",
        "textmining",
        "textmining_transferred",
        "combined_score",
    ],
    "URL": "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz",
    "INFO_URL": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    "SCORE_THRESHOLD": 400,  # STRING scores are 0-1000; 400 = medium confidence
}

def download_stringdb(data_path, rerun=False):
    """Download STRING PPI dataset for human (taxon 9606)"""
    raw_path = Path(data_path) / f'{STRINGDB["FOLDER"]}'
    raw_path.mkdir(parents=True, exist_ok=True)

    links_path = raw_path / STRINGDB["RAW_FILE"]
    info_path = raw_path / STRINGDB["INFO_FILE"]

    # Download main interaction file
    file1 =  raw_path / f"{STRINGDB['RAW_FILE']}"
    if not file1.exists():
        links_path = pooch.retrieve(
            url=STRINGDB["URL"],
            known_hash=None,
            path=raw_path,
            fname=STRINGDB["RAW_FILE"],
        )

    # Download protein info file (contains preferred_name / gene symbol)
    file2 =  raw_path / f"{STRINGDB['INFO_FILE']}"
    if not file2.exists():
        info_path = pooch.retrieve(
            url=STRINGDB["INFO_URL"],
            known_hash=None,
            path=raw_path,
            fname=STRINGDB["INFO_FILE"],
        )

    print(f"Downloaded STRING links to {links_path}")
    print(f"Downloaded STRING info  to {info_path}")
    return links_path, info_path

def process_stringdb(data_path, rerun=False):
    """
    Process the STRING dataset.

    STRING protein IDs look like '9606.ENSP00000000233'.
    The info file maps them to gene symbols (preferred_name), which is
    equivalent to HGNC symbols for most human proteins
    """
    raw_path = Path(data_path) / STRINGDB["FOLDER"]
    raw_file = raw_path / STRINGDB["RAW_FILE"]
    info_file = raw_path / STRINGDB["INFO_FILE"]
    processed_file = raw_path / STRINGDB["FINAL_FILE"]

    if processed_file.exists() and not rerun:
        print(f"File already exists: {processed_file}")
        return

    threshold = STRINGDB["SCORE_THRESHOLD"]

    # --- load protein to gene-symbol map ---
    info = pd.read_csv(info_file, sep="\t", usecols=["#string_protein_id", "preferred_name"])
    info.columns = ["string_id", "symbol"]
    id_to_symbol = info.set_index("string_id")["symbol"].to_dict()

    # --- load interactions ---
    print("Loading STRING interactions …")

    # Peek at available columns before loading — the "detailed" and basic
    # link files have different schemas, so we select only what's present.
    header = pd.read_csv(raw_file, sep=" ", nrows=0)
    available = set(header.columns)

    OPTIONAL_SCORE_COLS = [
        "experiments", "experiments_transferred",
        "database", "database_transferred",
        "coexpression", "coexpression_transferred",
        "textmining", "textmining_transferred",
    ]
    extra_cols = [c for c in OPTIONAL_SCORE_COLS if c in available]
    usecols = ["protein1", "protein2", "combined_score"] + extra_cols
    print(f"Columns found: {usecols}")

    df = pd.read_csv(raw_file, sep=" ", usecols=usecols)

    # Apply confidence threshold
    df = df[df["combined_score"] >= threshold].copy()
    print(f"Retained {len(df):,} edges with combined_score ≥ {threshold}")

    # Map to gene symbols
    df["node1"] = df["protein1"].map(id_to_symbol)
    df["node2"] = df["protein2"].map(id_to_symbol)

    # Drop unmapped / self-loops
    df = df.dropna(subset=["node1", "node2"])
    df = df[df["node1"] != df["node2"]]

    # Normalise scores to [0, 1] to match HIPPIE convention
    score_cols = ["combined_score"] + extra_cols
    df[score_cols] = df[score_cols] / 1000.0

    df["edge_source"] = "stringdb_ppi"

    result = df[["node1", "node2"] + score_cols + ["edge_source"]].copy()

    result.to_parquet(processed_file, index=False)
    print(f"Saved {len(result):,} edges  {processed_file}")

def load_stringdb(data_path):
    """Load STRING DB into a NetworkX graph"""
    processed_file = Path(data_path) / STRINGDB["FOLDER"] / STRINGDB["FINAL_FILE"]

    if not processed_file.exists():
        raise FileNotFoundError(
            f"{STRINGDB['FINAL_FILE']} not found. Run setup_datasets() first."
        )

    df = pd.read_parquet(processed_file)

    edge_attr = [c for c in df.columns if c not in ("node1", "node2")]

    G = nx.from_pandas_edgelist(
        df,
        source="node1",
        target="node2",
        edge_attr=edge_attr,
    )

    #print(f"STRING graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

####### BIOGRID DB #######

BIOGRID = {
    "FOLDER": "biogrid",
    "ZIP_FILE": "BIOGRID-ORGANISM-5.0.252.mitab.zip",
    "RAW_FILE": "BIOGRID-ORGANISM-Homo_sapiens-5.0.252.mitab.txt",
    "FINAL_FILE": "biogrid_ppi_hgnc.parquet",
    "URL": "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.252/BIOGRID-ORGANISM-5.0.252.mitab.zip",
    # PSI-MITAB 2.5 columns (tab-separated, no header in file)
    "columns": [
        "ID_A", "ID_B",
        "Alt_ID_A", "Alt_ID_B",
        "Aliases_A", "Aliases_B",
        "Detection_Methods",
        "First_Authors",
        "Publication_IDs",
        "Taxonomy_IDs_A", "Taxonomy_IDs_B",
        "Interaction_Types",
        "Source_Databases",
        "Interaction_IDs",
        "Confidence_Scores",
    ],
}

# Regex to pull the first entrez ID out of Alt_ID columns.
# BioGRID encodes them as  "entrez gene/locuslink:7157|..."
_ENTREZ_RE = r"entrez gene/locuslink:(\d+)"


def _parse_entrez(series: "pd.Series") -> "pd.Series":
    """Extract the first entrez gene ID from a BioGRID Alt_ID column."""
    return (
        series.str.extract(_ENTREZ_RE, expand=False)
              .pipe(pd.to_numeric, errors="coerce")
              .astype("Int64")
    )

def download_biogrid(data_path, rerun=False):
    """Download and unpack the BioGRID human MITAB file."""
    raw_path = Path(data_path) / BIOGRID["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    target_file = raw_path / BIOGRID["RAW_FILE"]
    if target_file.exists() and not rerun:
        print(f"BioGRID raw file already exists: {target_file}")
        return target_file

    # pooch can unpack zip archives and return the list of extracted files
    extracted = pooch.retrieve(
        url=BIOGRID["URL"],
        known_hash=None,
        path=raw_path,
        fname=BIOGRID["ZIP_FILE"],
        processor=pooch.Unzip(extract_dir=str(raw_path)),
    )

    print(f"Downloaded and extracted BioGRID to {raw_path}")
    # Return the specific human file path
    return target_file

def process_biogrid(data_path, rerun=False):
    """
    Process the BioGRID MITAB dataset.

    Entrez IDs are parsed from the Alt_ID_A / Alt_ID_B columns and mapped
    to HGNC gene symbols via Biomart, consistent with the HIPPIE pipeline.
    The Detection_Methods column is retained as an edge attribute so callers
    can filter to experimental evidence if desired.
    """
    raw_path = Path(data_path) / BIOGRID["FOLDER"]
    raw_file = raw_path / BIOGRID["RAW_FILE"]
    processed_file = raw_path / BIOGRID["FINAL_FILE"]

    if processed_file.exists() and not rerun:
        print(f"File already exists: {processed_file}")
        return

    if not raw_file.exists():
        raise FileNotFoundError(
            f"BioGRID raw file not found: {raw_file}. Run download_biogrid() first."
        )

    print("Loading BioGRID MITAB file …")
    df = pd.read_csv(
        raw_file,
        sep="\t",
        comment="#",
        header=None,
        names=BIOGRID["columns"],
        low_memory=False,
    )
    print(f"  {len(df):,} raw interactions loaded")

    # --- parse entrez IDs ---
    # In this BioGRID MITAB export the entrez IDs are in ID_A/ID_B
    # (format: "entrez gene/locuslink:1234"), not in the Alt_ID columns.
    df["entrez_id_1"] = _parse_entrez(df["ID_A"])
    df["entrez_id_2"] = _parse_entrez(df["ID_B"])

    # --- map to HGNC symbols via Biomart ---
    print("  Mapping entrez IDs to gene symbols via Biomart …")
    data_biomart = {"biomart": load_biomart(data_path)}

    db_mapped = convert_gene_df(
        df=df,
        gene_columns_map={"entrez_id_1": "node1", "entrez_id_2": "node2"},
        gene_source="entrez_id",
        target_name="symbol",
        datasets=data_biomart,
    )
    # print(db_mapped)
    edges = db_mapped.dropna(subset=["node1", "node2"]).copy()
    edges = edges[edges["node1"] != edges["node2"]]
    edges["edge_source"] = "biogrid_ppi"
    # print(df)
    result = edges[["node1", "node2", "Detection_Methods", "Interaction_Types", "edge_source"]].copy()
    result = result.rename(columns={
        "Detection_Methods": "detection_methods",
        "Interaction_Types": "interaction_types",
    })
    # print(result)
    result.to_parquet(processed_file, index=False)
    print(f"Saved {len(result):,} edges to {processed_file}")

def load_biogrid(data_path):
    """Load BioGRID into a NetworkX graph."""
    processed_file = Path(data_path) / BIOGRID["FOLDER"] / BIOGRID["FINAL_FILE"]

    if not processed_file.exists():
        raise FileNotFoundError(
            f"{BIOGRID['FINAL_FILE']} not found. Run setup_datasets() first."
        )

    df = pd.read_parquet(processed_file)

    edge_attr = [c for c in df.columns if c not in ("node1", "node2")]

    G = nx.from_pandas_edgelist(
        df,
        source="node1",
        target="node2",
        edge_attr=edge_attr,
    )

    #print(f"BioGRID graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

####### STRING DB (ARABIDOPSIS) #######
STRINGDB_ARABIDOPSIS = {
    "FOLDER": "stringdb_arabidopsis",
    "RAW_FILE": "3702.protein.links.detailed.v12.0.txt.gz",   # Aligned with human "detailed" file
    "INFO_FILE": "3702.protein.info.v12.0.txt.gz",            # Pointing to correct metadata mapping
    "MAPPING_FILE": "ARATH_3702_idmapping.dat.gz",
    "MAPPING_URL": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/ARATH_3702_idmapping.dat.gz",  # NEW 
    "FINAL_FILE": "stringdb_ppi_hgnc.parquet",
    "columns": [
        "protein1",
        "protein2",
        "neighborhood",
        "neighborhood_transferred",
        "fusion",
        "cooccurence",
        "homology",
        "coexpression",
        "coexpression_transferred",
        "experiments",
        "experiments_transferred",
        "database",
        "database_transferred",
        "textmining",
        "textmining_transferred",
        "combined_score",
    ],
    "URL": "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/3702.protein.links.detailed.v12.0.txt.gz",
    "INFO_URL": "https://stringdb-downloads.org/download/protein.info.v12.0/3702.protein.info.v12.0.txt.gz",
    "SCORE_THRESHOLD": 400,  
}

def download_stringdb_arabidopsis(data_path, rerun=False):
    raw_path = Path(data_path) / STRINGDB_ARABIDOPSIS["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    links_path = raw_path / STRINGDB_ARABIDOPSIS["RAW_FILE"]
    info_path  = raw_path / STRINGDB_ARABIDOPSIS["INFO_FILE"]
    mapping_path = raw_path / STRINGDB_ARABIDOPSIS["MAPPING_FILE"]  # NEW

    if not links_path.exists() or rerun:
        pooch.retrieve(url=STRINGDB_ARABIDOPSIS["URL"], known_hash=None,
                       path=raw_path, fname=STRINGDB_ARABIDOPSIS["RAW_FILE"])

    if not info_path.exists() or rerun:
        pooch.retrieve(url=STRINGDB_ARABIDOPSIS["INFO_URL"], known_hash=None,
                       path=raw_path, fname=STRINGDB_ARABIDOPSIS["INFO_FILE"])

    # NEW
    if not mapping_path.exists() or rerun:
        pooch.retrieve(url=STRINGDB_ARABIDOPSIS["MAPPING_URL"], known_hash=None,
                       path=raw_path, fname=STRINGDB_ARABIDOPSIS["MAPPING_FILE"])

    print(f"Downloaded STRING links to {links_path}")
    print(f"Downloaded STRING info  to {info_path}")
    print(f"Downloaded UniProt mapping to {mapping_path}")
    return links_path, info_path, mapping_path


def process_stringdb_arabidopsis(data_path, rerun=False):
    raw_path      = Path(data_path) / STRINGDB_ARABIDOPSIS["FOLDER"]
    raw_file      = raw_path / STRINGDB_ARABIDOPSIS["RAW_FILE"]
    mapping_file  = raw_path / STRINGDB_ARABIDOPSIS["MAPPING_FILE"]
    processed_file = raw_path / STRINGDB_ARABIDOPSIS["FINAL_FILE"]

    if processed_file.exists() and not rerun:
        print(f"File already exists: {processed_file}")
        return

    threshold = STRINGDB_ARABIDOPSIS["SCORE_THRESHOLD"]

    # --- build UniProt -> AGI mapping ---
    print("Loading UniProt -> AGI mapping …")
    mapping_df = pd.read_csv(mapping_file, sep="\t", header=None, compression="gzip",
                             names=["uniprot_id", "id_type", "id_value"])
    tair_df = mapping_df[mapping_df["id_type"] == "TAIR"]
    uniprot2agi = dict(zip(tair_df["uniprot_id"], tair_df["id_value"].str.upper()))
    print(f"Loaded {len(uniprot2agi):,} UniProt -> AGI entries")

    # --- load interactions ---
    print("Loading STRING interactions …")
    header = pd.read_csv(raw_file, sep=" ", compression="gzip", nrows=0)
    available = set(header.columns)

    OPTIONAL_SCORE_COLS = [
        "experiments", "experiments_transferred",
        "database", "database_transferred",
        "coexpression", "coexpression_transferred",
        "textmining", "textmining_transferred",
    ]
    extra_cols = [c for c in OPTIONAL_SCORE_COLS if c in available]
    usecols = ["protein1", "protein2", "combined_score"] + extra_cols

    df = pd.read_csv(raw_file, sep=" ", compression="gzip", usecols=usecols)
    df = df[df["combined_score"] >= threshold].copy()
    print(f"Retained {len(df):,} edges with combined_score >= {threshold}")

    # --- map UniProt IDs -> AGI ---
    df["node1"] = df["protein1"].str.split(".").str[1].map(uniprot2agi)
    df["node2"] = df["protein2"].str.split(".").str[1].map(uniprot2agi)

    unmapped = df["node1"].isna().sum() + df["node2"].isna().sum()
    print(f"Unmapped proteins (dropped): {unmapped:,}")

    df = df.dropna(subset=["node1", "node2"])
    df = df[df["node1"] != df["node2"]]
    print(f"After mapping and cleaning: {len(df):,} edges")

    # --- normalize scores ---
    score_cols = ["combined_score"] + extra_cols
    df[score_cols] = df[score_cols] / 1000.0

    df["edge_source"] = "stringdb_ppi"
    result = df[["node1", "node2"] + score_cols + ["edge_source"]].copy()

    all_nodes = pd.unique(result[["node1", "node2"]].values.ravel())
    print(f"Unique genes in graph: {len(all_nodes):,}")
    print(f"First 10 nodes: {list(all_nodes[:10])}")

    result.to_parquet(processed_file, index=False)
    print(f"Saved {len(result):,} edges to {processed_file}")


def load_stringdb_arabidopsis(data_path):
    """Load STRING DB into a NetworkX graph"""
    processed_file = Path(data_path) / STRINGDB_ARABIDOPSIS["FOLDER"] / STRINGDB_ARABIDOPSIS["FINAL_FILE"]

    if not processed_file.exists():
        raise FileNotFoundError(
            f"{STRINGDB_ARABIDOPSIS['FINAL_FILE']} not found. Run setup_datasets() first."
        )

    df = pd.read_parquet(processed_file)
    edge_attr = [c for c in df.columns if c not in ("node1", "node2")]

    G = nx.from_pandas_edgelist(df, source="node1", target="node2", edge_attr=edge_attr)
    return G


PPI_DATASETS = {
    'hippie': {
        'download': download_hippie,
        'process': process_hippie,
        'load': load_hippie
    },
    'stringdb': {
        'download': download_stringdb,
        'process': process_stringdb,
        'load': load_stringdb
    },
    "biogrid": {
        "download": download_biogrid,
        "process": process_biogrid,
        "load": load_biogrid,
    },
    'stringdb_arabidopsis': {
        'download': download_stringdb_arabidopsis,
        'process': process_stringdb_arabidopsis,
        'load': load_stringdb_arabidopsis
    }
}