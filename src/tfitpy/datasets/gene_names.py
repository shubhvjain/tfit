import sqlite3
import pooch
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from pyfaidx import Fasta

GENCODE_ATTRIBUTES = [
    'gene_id', 'transcript_id', 'gene_type',
    'gene_name', 'transcript_name', 'protein_id', 'exon_id',
    'chromosome', 'feature', 'start', 'end', 'strand', 'tss',
]

GENCODE = {
    "FOLDER": "gencode",
    "GTF_FILE": "gencode.v39.primary_assembly.annotation.gtf",
    "GTF_FILE_GZ": "gencode.v39.primary_assembly.annotation.gtf.gz",
    "FINAL_FILE": "gene_name_mapping.db",
    "URL": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.primary_assembly.annotation.gtf.gz",
    "FASTA_FILE": "GRCh38.primary_assembly.genome.fa",
    "FASTA_FILE_GZ": "GRCh38.primary_assembly.genome.fa.gz",
    "URL_FASTA": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/GRCh38.primary_assembly.genome.fa.gz",
}


def download_gencode(data_path, rerun=False):
    """Download and decompress GENCODE GTF annotation and genome FASTA."""
    raw_path = Path(data_path) / GENCODE["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    gtf_file = raw_path / GENCODE["GTF_FILE"]
    if not gtf_file.exists() or rerun:
        pooch.retrieve(
            url=GENCODE["URL"],
            known_hash=None,
            path=raw_path,
            fname=GENCODE["GTF_FILE_GZ"],
            processor=pooch.Decompress(name=GENCODE["GTF_FILE"])
        )
        print(f"Downloaded GTF to {gtf_file}")
    else:
        print(f"GTF already exists: {gtf_file}")

    fasta_file = raw_path / GENCODE["FASTA_FILE"]
    if not fasta_file.exists() or rerun:
        pooch.retrieve(
            url=GENCODE["URL_FASTA"],
            known_hash=None,
            path=raw_path,
            fname=GENCODE["FASTA_FILE_GZ"],
            processor=pooch.Decompress(name=GENCODE["FASTA_FILE"])
        )
        print(f"Downloaded FASTA to {fasta_file}")
    else:
        print(f"FASTA already exists: {fasta_file}")


def generate_kv(inp):
    parts = inp.strip().split(' ')
    if len(parts) == 2:
        key = parts[0].strip()
        value = parts[1].replace('"', "").strip()
        return key, value
    return None


def process_line(line):
    fields = line.replace('\n', '').split('\t')
    attribute_str = fields[8].replace("'", ' ')
    items = attribute_str.split(";")
    result = {}
    for item in items:
        pair = generate_kv(item)
        if pair:
            k, v = pair
            result[k] = v

    strand = fields[6]
    start  = int(fields[3])
    end    = int(fields[4])

    result['chromosome'] = fields[0]
    result['feature']    = fields[2]
    result['start']      = start
    result['end']        = end
    result['strand']     = strand
    result['tss']        = start if strand == '+' else end

    return result


def process_gencode(data_path, rerun=False, chunksize=200000):
    gtf_path = Path(data_path) / GENCODE["FOLDER"] / GENCODE["GTF_FILE"]
    db_path = Path(data_path) / GENCODE["FOLDER"] / GENCODE["FINAL_FILE"]

    if db_path.exists() and not rerun:
        print(f"File already exists: {db_path}")
        return

    print("Creating database...")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA journal_mode = MEMORY")

    first_chunk = True

    try:
        with open(gtf_path, "r") as f:
            chunk = []
            for line in f:
                if line.startswith('#'):
                    continue

                processed = process_line(line)
                row = {col: processed.get(col, None) for col in GENCODE_ATTRIBUTES}
                chunk.append(row)

                if len(chunk) >= chunksize:
                    pd.DataFrame(chunk, columns=GENCODE_ATTRIBUTES).to_sql(
                        "mappings", con,
                        if_exists='replace' if first_chunk else 'append',
                        index=False
                    )
                    first_chunk = False
                    print(f"Processed {len(chunk)} lines")
                    chunk = []

            if chunk:
                pd.DataFrame(chunk, columns=GENCODE_ATTRIBUTES).to_sql(
                    "mappings", con,
                    if_exists='replace' if first_chunk else 'append',
                    index=False
                )
                print(f"Processed final {len(chunk)} lines")

        print("Database created successfully")

    finally:
        con.execute("CREATE INDEX IF NOT EXISTS idx_gene_id ON mappings(gene_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_gene_name ON mappings(gene_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_transcript_id ON mappings(transcript_id)")
        con.close()


def load_gencode(data_path):
    """Return an open SQLite connection to the GENCODE database."""
    db_path = Path(data_path) / GENCODE["FOLDER"] / GENCODE["FINAL_FILE"]
    if not db_path.exists():
        raise FileNotFoundError(f"{GENCODE['FINAL_FILE']} not found. Run process_gencode() first.")
    return sqlite3.connect(db_path)

def load_genome(data_path):
    """
    """
    fasta_path = Path(data_path) / GENCODE["FOLDER"] / GENCODE["FASTA_FILE"]
    genome = Fasta(str(fasta_path))
    return genome


##### Biomart database 

BIOMART = {
    "URL": "http://www.ensembl.org/biomart/martservice?query=<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE Query><Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' datasetConfigVersion='0.6'><Dataset name='hsapiens_gene_ensembl' interface='default'><Attribute name='ensembl_gene_id'/><Attribute name='external_gene_name'/><Attribute name='entrezgene_id'/><Attribute name='uniprotswissprot'/><Attribute name='refseq_mrna'/><Attribute name='description'/></Dataset></Query>",
    "FOLDER": "biomart",
    "RAW_FILE": "biomart_gene_mapping.txt",
    "FINAL_FILE": "biomart_gene_mappings.db",
    "COLUMNS": ['ensembl_gene_id', 'symbol', 'entrez_id', 'uniprot_id', 'refseq_id', 'description'],
}


def download_biomart(data_path, rerun=False):
    """Download biomart dataset"""
    raw_path = Path(data_path) / BIOMART['FOLDER']
    raw_path.mkdir(parents=True, exist_ok=True)

    output_file = raw_path / BIOMART['RAW_FILE']

    if output_file.exists() and not rerun:
        print(f"Biomart already downloaded: {output_file}")
        return output_file

    file_path = pooch.retrieve(
        url=BIOMART['URL'],
        known_hash=None,
        path=raw_path,
        fname=BIOMART['RAW_FILE']
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

    print(f"Reading {raw_file}...")
    df = pd.read_csv(raw_file, sep='\t')
    df.columns = BIOMART['COLUMNS']

    df['entrez_id'] = pd.to_numeric(df['entrez_id'], errors='coerce').astype('Int64')

    print(f"Creating SQLite database at {processed_file}...")
    con = sqlite3.connect(processed_file)

    df.to_sql('gene_mappings', con, if_exists='replace', index=False)

    con.execute('CREATE INDEX IF NOT EXISTS idx_ensembl ON gene_mappings(ensembl_gene_id)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON gene_mappings(symbol)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_entrez ON gene_mappings(entrez_id)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_uniprot ON gene_mappings(uniprot_id)')
    con.commit()
    con.close()

    print(f"SQLite database created with {len(df)} rows and indexes")


def load_biomart(data_path):
    """Return an open SQLite connection to the BioMart database."""
    processed_file = Path(data_path) / BIOMART['FOLDER'] / BIOMART['FINAL_FILE']

    if not processed_file.exists():
        raise FileNotFoundError(f"{BIOMART['FINAL_FILE']} not found. Run process_biomart() first.")

    return sqlite3.connect(processed_file)



GENE_DATASETS = {
    'gencode': {
        'download': download_gencode,
        'process': process_gencode,
        'load': load_gencode
    },
    'biomart': {
        'download': download_biomart,
        'process': process_biomart,
        'load': load_biomart,
    }
}

def convert_genes(
    input: List[Any],
    input_type: str,
    output_type: str,
    datasets: Optional[dict] = None,
    db_key: str = 'biomart',
) -> Dict[str, Any]:
    """
    Convert gene identifiers from one type to another.

    Args:
        input      : List of identifiers to map
        input_type : Source column name in the database
        output_type: Target column name in the database
        datasets   : Cache containing loaded dataset connections
        db_key     : Which db to use - 'gencode' or 'biomart'

    Returns:
        Dict mapping {source_id: target_id}
    """
    if datasets is None:
        raise ValueError("datasets cache is required.")
    if input is None:
        raise ValueError("No input provided")

    con = datasets[db_key]
    unique_values = [x for x in set(input) if x is not None and pd.notna(x)]

    if len(unique_values) == 0:
        print("No values to map")
        return {}

    placeholders = ','.join(['?'] * len(unique_values))
    query = f"""
        SELECT DISTINCT {input_type}, {output_type}
        FROM mappings
        WHERE {input_type} IN ({placeholders})
        AND {output_type} IS NOT NULL
    """

    mapping_df = pd.read_sql_query(query, con, params=unique_values)
    mapping_dict = dict(zip(mapping_df[input_type], mapping_df[output_type]))

    total = len(input)
    unique_count = len(unique_values)
    mapped_count = len(mapping_dict)
    null_in_results = sum(1 for gene in input if gene not in mapping_dict)

    print(f"Mapping from {input_type} to {output_type}:")
    print(f"  Total input values: {total}")
    print(f"  Unique values: {unique_count}")
    print(f"  Successfully mapped: {mapped_count}")
    print(f"  Failed to map: {unique_count - mapped_count}")
    print(f"  Null results: {null_in_results} ({null_in_results/max(total,1)*100:.2f}%)")

    return mapping_dict


def convert_gene_df(
    df: pd.DataFrame,
    gene_columns_map: Dict[str, str],
    gene_source: str,
    target_name: str,
    datasets: Optional[dict] = None,
    db_key: str = 'biomart',
) -> pd.DataFrame:
    """
    Bulk conversion of multiple gene ID columns using a single SQLite query.

    Args:
        df              : Input DataFrame with gene identifier columns
        gene_columns_map: Mapping of source columns to new column names
        gene_source     : Type of gene ID in the source columns
        target_name     : Type of gene ID to convert to
        datasets        : Cache containing loaded dataset connections
        db_key          : Which db to use - 'gencode' or 'biomart'

    Returns:
        DataFrame with new columns added containing converted gene IDs
    """
    if datasets is None:
        raise ValueError("datasets cache is required.")

    con = datasets[db_key]

    for src_col in gene_columns_map.keys():
        if src_col not in df.columns:
            raise KeyError(f"Column '{src_col}' not in DataFrame")

    all_genes = set()
    for src_col in gene_columns_map.keys():
        all_genes.update(df[src_col].dropna().astype(str).unique())

    unique_values = [x for x in all_genes if x is not None and pd.notna(x)]

    if len(unique_values) == 0:
        print("No values to map")
        return df.copy()

    placeholders = ','.join(['?'] * len(unique_values))
    query = f"""
        SELECT DISTINCT {gene_source}, {target_name}
        FROM mappings
        WHERE {gene_source} IN ({placeholders})
        AND {target_name} IS NOT NULL
    """

    mapping_df = pd.read_sql_query(query, con, params=unique_values)
    gene_map = dict(zip(mapping_df[gene_source], mapping_df[target_name]))

    print(f"Bulk mapping from {gene_source} to {target_name}:")
    print(f"  Unique input values: {len(unique_values)}")
    print(f"  Successfully mapped: {len(gene_map)}")
    print(f"  Failed to map: {len(unique_values) - len(gene_map)}")

    out = df.copy()
    for src_col, new_col in gene_columns_map.items():
        out[new_col] = df[src_col].map(gene_map)
    return out