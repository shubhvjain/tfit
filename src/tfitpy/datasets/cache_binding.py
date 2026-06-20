"""
To build are cache of TF binding affinity scores for 20k protein coding genes.
"""

from tfitpy.datasets.gene_names import  load_gencode, load_genome, GENCODE
import sqlite3
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta
from tfitpy.datasets.binding import get_jasper_path
from pyjaspar import jaspardb
from tfitpy.datasets.regulators import load_tflist
import numpy as np
from numba import njit, prange



def generate_promoter_reference_human(data_path, upstream_bp: int = 2000, rerun=False):
    base_path = Path(data_path)
    db_path = base_path / GENCODE["FOLDER"] / GENCODE["FINAL_FILE"]
    fasta_path = base_path / GENCODE["FOLDER"] / GENCODE["FASTA_FILE"]

    output_dir = base_path / "tfbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"human_promoters_{upstream_bp}bp.parquet"
    if output_file.exists() and not rerun:
        print("File already exists")
        return

    if not db_path.exists():
        raise FileNotFoundError(f"Database missing at {db_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Genome FASTA missing at {fasta_path}")

    # Load protein-coding transcript rows only
    query = """
    SELECT
        gene_id,
        gene_name,
        transcript_id,
        chromosome,
        strand,
        start,
        end,
        tss,
        tag
    FROM mappings
    WHERE feature = 'transcript'
      AND gene_type = 'protein_coding'
      AND gene_id IS NOT NULL
      AND transcript_id IS NOT NULL
      AND chromosome IS NOT NULL
      AND strand IN ('+', '-')
      AND tss IS NOT NULL
    """

    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, con)
    con.close()

    def get_priority(tag_value):
        tags = set()
        if pd.notna(tag_value) and tag_value:
            tags = {x.strip() for x in str(tag_value).split(",") if x.strip()}

        # Priority order:
        # 1. MANE_Select
        # 2. Ensembl_canonical
        # 3. appris_principal_1, 2, ...
        # 4. longest transcript span fallback
        if "MANE_Select" in tags:
            return (0, 0)

        if "Ensembl_canonical" in tags:
            return (1, 0)

        appris_tags = [t for t in tags if t.startswith("appris_principal_")]
        if appris_tags:
            nums = []
            for t in appris_tags:
                try:
                    nums.append(int(t.split("_")[-1]))
                except ValueError:
                    pass
            return (2, min(nums) if nums else 999)

        return (3, 999)

    # Rank transcripts within each gene
    df["priority"] = df["tag"].apply(get_priority)
    df["tx_len"] = df["end"] - df["start"]

    # Keep one transcript per gene
    df = df.sort_values(
        by=["gene_id", "priority", "tx_len", "transcript_id"],
        ascending=[True, True, False, True]
    ).drop_duplicates(subset=["gene_id"], keep="first").copy()

    genome = Fasta(str(fasta_path))
    records = []

    for row in df.itertuples(index=False):
        if row.chromosome not in genome:
            continue

        # Build promoter interval from TSS
        if row.strand == "+":
            start0 = row.tss - upstream_bp
            end0 = row.tss

            if start0 < 0:
                continue

            seq = str(genome[row.chromosome][start0:end0]).upper()

        else:
            start0 = row.tss - 1
            end0 = start0 + upstream_bp

            seq_obj = genome[row.chromosome][start0:end0]
            if len(seq_obj) != upstream_bp:
                continue

            # Reverse-complement for minus strand promoters
            seq = str(-seq_obj).upper()

        if len(seq) != upstream_bp:
            continue

        records.append({
            "gene_id": row.gene_id,
            "gene_name": row.gene_name,
            "transcript_id": row.transcript_id,
            "chromosome": row.chromosome,
            "strand": row.strand,
            "tss": row.tss,
            "promoter_start": start0 + 1,
            "promoter_end": end0,
            "promoter_sequence": seq,
        })

    df_promoters = pd.DataFrame(records)
    df_promoters.to_parquet(output_file, index=False)

    print(f"Saved {len(df_promoters)} promoters to {output_file}")


def get_promoter_reference_human(data_path, upstream_bp: int = 2000):
    """
    """
    pt = Path(data_path) / "tfbs"/ f"human_promoters_{upstream_bp}bp.parquet"
    df = pd.read_parquet(pt)
    return df 


def list_human_tfs_pyjaspar(data_path):
    # Initialize the JASPAR database object
    # (pyjaspar will automatically pull metadata for the release)
    jaspar_db = get_jasper_path(data_path,organism="human")
    jdb = jaspardb(sqlite_db_path=str(jaspar_db))
    
    # Fetch human motifs using the NCBI Taxonomy ID for Homo sapiens (9606)
    # human_motifs = jdb.fetch_motifs(
    #     collection='CORE',
    #     tax_group='vertebrates',
    #     species=['9606']
    # )
    # We restrict it to the 'CORE' collection and 'vertebrates'
    all_vertebrate_motifs = jdb.fetch_motifs(
        collection='CORE',
        tax_group='vertebrates'
    )

    # Extract unique TF names from ALL vertebrates
    unique_vertebrate_tf_names = sorted(list(set([motif.name for motif in all_vertebrate_motifs])))
    return all_vertebrate_motifs

    #print(f"Total vertebrate TF profiles (motifs) found: {len(all_vertebrate_motifs)}")
    #print(f"Total unique vertebrate TF names available: {len(unique_vertebrate_tf_names)}")
    #print("\nFirst 10 vertebrate TFs:", unique_vertebrate_tf_names[:10])

    # tf_list = load_tflist(data_path)["gene_name"].tolist()
    # coreg_path = Path(data_path) / "coregulators_list"/ "list.csv"
    # coreg_list =  pd.read_csv(coreg_path)["symbol"].tolist()
    # reg = set(tf_list) | set(coreg_list)
    # print("Checking ",len(reg)," regulators")
    # target_set = set(tf.upper() for tf in tf_list)
    # matched_tfs = [name for name in unique_vertebrate_tf_names if name.upper() in target_set]
    # print(f"Successfully matched {len(matched_tfs)} / {len(tf_list)} TFs")
    # matched_tfs = [name for name in unique_vertebrate_tf_names if name.upper() in reg]
    # print(f"Successfully matched {len(matched_tfs)} / {len(reg)} reg")
    
    


def get_regulator_jaspar_status(data_path) -> pd.DataFrame:
    """
    Intersects the user's TF and CoRegulator lists with the local JASPAR database,
    accounting for heterodimers (e.g., 'ARNT::HIF1A'), and returns a summary DataFrame.
    """
    # 1. Load the local JASPAR database via pyjaspar
    jaspar_db = get_jasper_path(data_path)
    jdb = jaspardb(sqlite_db_path=str(jaspar_db))
    
    # Fetch all vertebrate core motifs to widen the matching net
    all_vertebrate_motifs = jdb.fetch_motifs(
        collection='CORE',
        tax_group='vertebrates'
    )
    
    # Create a mapping of individual components to their full JASPAR motif names
    # E.g., 'HIF1A' -> ['HIF1A', 'ARNT::HIF1A']
    jaspar_component_map = {}
    for motif in all_vertebrate_motifs:
        name_upper = motif.name.upper()
        # Split complexes to handle dimer subunits
        components = [c.strip() for c in name_upper.split('::')]
        
        for comp in components:
            if comp not in jaspar_component_map:
                jaspar_component_map[comp] = []
            jaspar_component_map[comp].append(motif.name)

    # 2. Load your input gene lists
    tf_list = load_tflist(data_path)["gene_name"].dropna().astype(str).tolist()
    
    coreg_path = Path(data_path) / "coregulators_list" / "list.csv"
    coreg_list = pd.read_csv(coreg_path)["symbol"].dropna().astype(str).tolist()
    
    # Standardize to uppercase sets for fast classification logic
    tf_set = set(tf.upper().strip() for tf in tf_list)
    coreg_set = set(coreg.upper().strip() for coreg in coreg_list)
    all_regulators = tf_set | coreg_set
    
    results = []
    
    # 3. Classify and map each unique regulator gene
    for gene in sorted(list(all_regulators)):
        # Determine Regulator Type
        if gene in tf_set and gene in coreg_set:
            reg_type = "TF & CoReg"
        elif gene in tf_set:
            reg_type = "TF"
        else:
            reg_type = "CoReg"
            
        # Find matches in our JASPAR component map
        matching_motifs = jaspar_component_map.get(gene, [])
        found_status = "Found" if len(matching_motifs) > 0 else "Not Found"
        
        # Format list of actual profiles for clarity (e.g., "HIF1A, ARNT::HIF1A")
        jaspar_profiles = ", ".join(matching_motifs) if matching_motifs else None
        
        results.append({
            "Gene_Symbol": gene,
            "Regulator_Type": reg_type,
            "JASPAR_Status": found_status,
            "Matched_JASPAR_Motifs": jaspar_profiles
        })
        
    # 4. Generate summary DataFrame
    df_status = pd.DataFrame(results)
    
    # Print high-level diagnostics
    print(f"--- Regulator Coverage Summary ---")
    print(f"Total Regulators Evaluated: {len(df_status)}")
    print(df_status.groupby(['Regulator_Type', 'JASPAR_Status']).size().to_string())
    print("-" * 34)
    df_status.to_csv(Path(data_path)/"tfbs"/"reg_jaspar_status_check.csv")

    return df_status



###### optimized implementation of TRAP score

@njit(cache=True)
def _encode_sequence_numba(seq_bytes: np.ndarray) -> np.ndarray:
    L = len(seq_bytes)
    encoded = np.empty(L, dtype=np.int8)
    for i in range(L):
        b = seq_bytes[i]
        if b == 65 or b == 97: encoded[i] = 0
        elif b == 67 or b == 99: encoded[i] = 1
        elif b == 71 or b == 103: encoded[i] = 2
        elif b == 84 or b == 116: encoded[i] = 3
        else: encoded[i] = -1
    return encoded

@njit(cache=True)
def _calculate_single_affinity_numba(
    seq_encoded: np.ndarray, energy_matrix_f: np.ndarray, energy_matrix_r: np.ndarray, r0: float, W: int
) -> float:
    L = len(seq_encoded)
    if L < W: return 0.0
    total_expected_bound = 0.0
    num_windows = L - W + 1
    for l in range(num_windows):
        has_masked = False
        for i in range(W):
            if seq_encoded[l + i] < 0:
                has_masked = True
                break
        if has_masked: continue
        energy_f = 0.0
        for i in range(W):
            energy_f += energy_matrix_f[seq_encoded[l + i], i]
        energy_r = 0.0
        for i in range(W):
            energy_r += energy_matrix_r[3 - seq_encoded[l + W - 1 - i], i]
        p_f = (r0 * np.exp(-energy_f)) / (1.0 + r0 * np.exp(-energy_f))
        p_r = (r0 * np.exp(-energy_r)) / (1.0 + r0 * np.exp(-energy_r))
        total_expected_bound += (p_f + p_r)
    return total_expected_bound

@njit(parallel=True, cache=True)
def _execute_trap_matrix_numba(
    flattened_sequences: np.ndarray, boundaries: np.ndarray,
    energy_f_matrices: np.ndarray, energy_r_matrices: np.ndarray,
    r0_values: np.ndarray, w_values: np.ndarray, num_genes: int, num_tfs: int
) -> np.ndarray:
    output_matrix = np.empty((num_genes, num_tfs), dtype=np.float64)
    for i in prange(num_genes):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        seq_encoded = flattened_sequences[start_idx:end_idx]
        for j in range(num_tfs):
            output_matrix[i, j] = _calculate_single_affinity_numba(
                seq_encoded, energy_f_matrices[j], energy_r_matrices[j], r0_values[j], w_values[j]
            )
    return output_matrix


def precompute_trap_affinity_cache(
    promoter_dict: dict, 
    jaspar_matrices_dict: dict, 
    lambda_param: float = 0.7, 
    bg_gc: float = 0.5
) -> pd.DataFrame:
    """Orchestrates high-throughput parallel computation of TRAP affinities.
    
    Arguments:
        promoter_dict: Dict mapping {gene_id: promoter_string}
        jaspar_matrices_dict: Dict mapping {tf_id: jaspar_count_dict}
    """
    gene_ids = list(promoter_dict.keys())
    tf_ids = list(jaspar_matrices_dict.keys())
    
    num_genes = len(gene_ids)
    num_tfs = len(tf_ids)
    
    nuc_order = ['A', 'C', 'G', 'T']
    bg_frequencies = np.array([
        (1.0 - bg_gc) / 2.0, bg_gc / 2.0, bg_gc / 2.0, (1.0 - bg_gc) / 2.0
    ], dtype=np.float64)
    bg_max = np.max(bg_frequencies)
    energy_bg = np.log(bg_frequencies / bg_max) / lambda_param

    # Determine maximum motif width to allocate regular array sizes for Numba
    max_w = max(len(jaspar_matrices_dict[tf]['A']) for tf in tf_ids)
    
    # Allocate energy matrix blocks
    energy_f_block = np.zeros((num_tfs, 4, max_w), dtype=np.float64)
    energy_r_block = np.zeros((num_tfs, 4, max_w), dtype=np.float64)
    r0_values = np.empty(num_tfs, dtype=np.float64)
    w_values = np.empty(num_tfs, dtype=np.int32)
    
    # 1. Precompute fixed TF structures
    for j, tf_id in enumerate(tf_ids):
        matrix_counts = np.array([jaspar_matrices_dict[tf_id][nuc] for nuc in nuc_order], dtype=np.float64)
        matrix_counts += 1.0
        W = matrix_counts.shape[1]
        
        m_max = np.max(matrix_counts, axis=0)
        energy_matrix_base = np.log(m_max / matrix_counts) / lambda_param
        
        energy_matrix_f = energy_matrix_base + energy_bg[:, np.newaxis]
        energy_matrix_r = energy_matrix_base[:, ::-1] + energy_bg[:, np.newaxis]
        
        energy_f_block[j, :, :W] = energy_matrix_f
        energy_r_block[j, :, :W] = energy_matrix_r
        r0_values[j] = np.exp(0.585 * W - 5.66)
        w_values[j] = W

    # 2. Flatten and sequence encode inputs efficiently to avoid separate arrays overhead
    encoded_list = []
    boundaries = [0]
    for gene_id in gene_ids:
        seq_bytes = np.frombuffer(promoter_dict[gene_id].encode('ascii'), dtype=np.uint8)
        encoded_seq = _encode_sequence_numba(seq_bytes)
        encoded_list.append(encoded_seq)
        boundaries.append(boundaries[-1] + len(encoded_seq))
        
    flattened_sequences = np.concatenate(encoded_list)
    boundaries = np.array(boundaries, dtype=np.int32)
    
    # 3. Fire parallel engine
    print(f"Processing matrix calculations ({num_genes} genes x {num_tfs} TFs)...")
    matrix_results = _execute_trap_matrix_numba(
        flattened_sequences, boundaries, 
        energy_f_block, energy_r_block, 
        r0_values, w_values, 
        num_genes, num_tfs
    )
    
    # 4. Generate structured cache frame
    return pd.DataFrame(matrix_results, index=gene_ids, columns=tf_ids)


def generate_and_cache_trap_scores_human(
    data_path: str,
    upstream_bp: int = 2000,
    lambda_param: float = 0.7,
    bg_gc: float = 0.5
) -> None:
    base_dir = Path(data_path)
    tfbs_dir = base_dir / "tfbs"
    tfbs_dir.mkdir(parents=True, exist_ok=True)

    output_file = tfbs_dir / "trap_scores_human.parquet"
    if output_file.exists():
        print("TRAP cache already exists:", output_file)
        return

    # 1. Load promoters
    df_promoters = get_promoter_reference_human(data_path, upstream_bp=upstream_bp)
    promoter_dict = {
        row.gene_id: row.promoter_sequence
        for row in df_promoters.itertuples(index=False)
    }

    # 2. Load motifs from local JASPAR
    print("Fetching vertebrate motifs from local JASPAR repository...")
    all_motifs = list_human_tfs_pyjaspar(data_path)

    jaspar_matrices_dict = {}
    tf_labels = []
    nuc_order = ["A", "C", "G", "T"]

    for motif in all_motifs:
        counts = motif.pwm_counts if hasattr(motif, "pwm_counts") else motif.counts
        jaspar_matrices_dict[motif.name] = {
            nuc: np.array(counts[nuc], dtype=np.float64) for nuc in nuc_order
        }
        tf_labels.append(motif.name)

    print(f"Scoring {len(promoter_dict)} genes x {len(tf_labels)} TFs")

    # 3. Run TRAP core
    df_cache = precompute_trap_affinity_cache(
        promoter_dict=promoter_dict,
        jaspar_matrices_dict=jaspar_matrices_dict,
        lambda_param=lambda_param,
        bg_gc=bg_gc,
    )

    # 4. Map gene_ids back to gene_names for readability (optional)
    gene_id_to_name = {
        row.gene_id: row.gene_name
        for row in df_promoters.itertuples(index=False)
    }
    df_cache.index = [gene_id_to_name.get(gid, gid) for gid in df_cache.index]

    # Deduplicate gene names and TF names if needed
    if df_cache.index.duplicated().any():
        df_cache = df_cache.groupby(df_cache.index).mean()
    if df_cache.columns.duplicated().any():
        df_cache = df_cache.groupby(df_cache.columns, axis=1).mean()

    # 5. Save
    df_cache.to_parquet(output_file)
    print("TRAP cache written to:", output_file)

def download(data_path):
    """"""
    return 


def process_human(data_path):
    """"""
    generate_promoter_reference_human(data_path, upstream_bp=2000)
    generate_and_cache_trap_scores_human(data_path, upstream_bp=2000)



def load_human(data_path):
    """"""
    pt = Path(data_path) / "tfbs"/ f"trap_scores_human.parquet"
    df = pd.read_parquet(pt)
    return df 


BINDING_CACHE = {
    "trap_cache_human":{
        "download":download,
        "process":process_human,
        "load":load_human
    }
}