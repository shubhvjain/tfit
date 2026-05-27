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
from tfitpy.datasets.tf import load_tflist

def generate_promoter_reference(data_path: str, upstream_bp: int = 2000,rerun=False) -> None:
    """Generates an optimized, clean database of 2000bp promoters for downstream TRAP tasks.
    
    Filters for canonical protein-coding transcripts utilizing the parsed 'tag' metadata, 
    and writes the final matrix directly to data_path/tfbs as a high-performance Parquet file.
    """
    base_path = Path(data_path)
    db_path = base_path / GENCODE["FOLDER"] / GENCODE["FINAL_FILE"]
    fasta_path = base_path / GENCODE["FOLDER"] / GENCODE["FASTA_FILE"]
    
    # Dynamically build and check output folder: data_path / tfbs
    output_dir = base_path / "tfbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"canonical_promoters_{upstream_bp}bp.parquet"
    if output_file.exists() and not rerun:
        print("File already exists")
        return

    if not db_path.exists():
        raise FileNotFoundError(f"Database missing at {db_path}. Please run process_gencode first.")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Genome FASTA missing at {fasta_path}.")

    print("Querying canonical protein-coding transcripts from SQLite mappings...")
    con = sqlite3.connect(db_path)
    
    # We query feature='transcript' because tags like 'appris_principal_1' 
    # are assigned specifically to transcript rows rather than broad gene rows.
    query = """
        SELECT gene_id, gene_name, chromosome, strand, tss, tag 
        FROM mappings 
        WHERE feature = 'transcript' 
          AND gene_type = 'protein_coding'
          AND (tag LIKE '%appris_principal_1%' OR tag LIKE '%ensembl_canonical%')
          AND chromosome IS NOT NULL
          AND tss IS NOT NULL
    """
    df_transcripts = pd.read_sql_query(query, con)
    con.close()
    
    # To be absolutely safe against genes with complex annotations possessing BOTH tags,
    # we drop duplicates so we keep exactly ONE canonical transcript promoter per unique gene_id.
    df_transcripts = df_transcripts.drop_duplicates(subset=["gene_id"])
    
    print(f"Loaded coordinates for {len(df_transcripts)} unique canonical genes. Initializing pyfaidx...")
    genome = Fasta(str(fasta_path))
    records = []
    
    # Extract sequences sequentially using low-overhead memory slices
    for _, row in df_transcripts.iterrows():
        chrom = row['chromosome']
        tss = int(row['tss'])
        strand = row['strand']
        
        if chrom not in genome:
            continue
            
        # Determine exact upstream sequence coordinates based on strand
        if strand == '+':
            start = max(0, tss - upstream_bp)
            end = tss
        else:
            start = tss
            end = tss + upstream_bp
            
        # Grab slice string and standardize to uppercase characters (removes soft-masking)
        seq_str = str(genome[chrom][start:end]).upper()
        
        # Verify sequence length matches requested window length exactly (skips boundary cutoffs)
        if len(seq_str) == upstream_bp:
            records.append({
                "gene_id": row['gene_id'],
                "gene_name": row['gene_name'],
                "chromosome": chrom,
                "strand": strand,
                "tss": tss,
                "promoter_sequence": seq_str
            })
            
    # Build reference pandas frame
    df_promoters = pd.DataFrame(records)
    
    # Save output using high-velocity binary Parquet inside your requested folder
    
    df_promoters.to_parquet(output_file, index=False)
    
    print(f"Successfully compiled {len(df_promoters)} promoters.")
    print(f"Saved binary reference file to: {output_file}")


def get_promoter_reference(data_path, upstream_bp: int = 2000):
    """
    """
    pt = Path(data_path) / "tfbs"/ f"canonical_promoters_{upstream_bp}bp.parquet"
    df = pd.read_parquet(pt)
    return df 



def list_human_tfs_pyjaspar(data_path):
    # Initialize the JASPAR database object
    # (pyjaspar will automatically pull metadata for the release)
    jaspar_db = get_jasper_path(data_path)
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


import numpy as np
import pandas as pd
from numba import njit, prange

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange

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



def generate_and_cache_trap_scores(
    data_path: str, 
    upstream_bp: int = 2000, 
    lambda_param: float = 0.7, 
    bg_gc: float = 0.5
) -> None:
    """Loads extracted promoter references and local JASPAR matrices, computes physical binding 
    affinity matrices using Numba parallelism, and stores scores as a binary Parquet dataframe.
    """
    base_dir = Path(data_path)
    tfbs_dir = base_dir / "tfbs"
    output_file = tfbs_dir / "trap_scores.parquet"
    
    # 1. Load the generated canonical promoters reference
    df_promoters = get_promoter_reference(data_path, upstream_bp=upstream_bp)
    
    # Extract structural components and use gene_name for readable output indexing
    gene_labels = df_promoters["gene_name"].astype(str).tolist()
    sequences = df_promoters["promoter_sequence"].astype(str).tolist()
    num_genes = len(gene_labels)

    # 2. Extract Motifs from local Database via your existing list_human_tfs_pyjaspar wrapper
    print("Fetching vertebrate motifs from local JASPAR repository...")
    all_motifs = list_human_tfs_pyjaspar(data_path)
    
    tf_labels = []
    jaspar_objects = []
    for motif in all_motifs:
        tf_labels.append(motif.name)
        jaspar_objects.append(motif)
    num_tfs = len(tf_labels)
    
    print(f"Dataset configurations: {num_genes} Target Genes | {num_tfs} Active Transcription Factors.")

    # 3. Construct base matrices and track sizes
    nuc_order = ['A', 'C', 'G', 'T']
    bg_frequencies = np.array([
        (1.0 - bg_gc) / 2.0, bg_gc / 2.0, bg_gc / 2.0, (1.0 - bg_gc) / 2.0
    ], dtype=np.float64)
    bg_max = np.max(bg_frequencies)
    energy_bg = np.log(bg_frequencies / bg_max) / lambda_param

    max_w = max(motif.pwm_counts['A'].shape[0] if hasattr(motif, 'pwm_counts') else len(motif.counts['A']) for motif in jaspar_objects)
    
    energy_f_block = np.zeros((num_tfs, 4, max_w), dtype=np.float64)
    energy_r_block = np.zeros((num_tfs, 4, max_w), dtype=np.float64)
    r0_values = np.empty(num_tfs, dtype=np.float64)
    w_values = np.empty(num_tfs, dtype=np.int32)
    
    print("Pre-computing structural background energy blocks for profiles...")
    for j, motif in enumerate(jaspar_objects):
        # Access background count records safely handling structural variants
        counts = motif.pwm_counts if hasattr(motif, 'pwm_counts') else motif.counts
        matrix_counts = np.array([counts[nuc] for nuc in nuc_order], dtype=np.float64)
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

    # 4. Flatten and encode sequences into a continuous pointer block
    print("Converting promoter strings to numerical index arrays...")
    encoded_list = []
    boundaries = [0]
    for seq in sequences:
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        encoded_seq = _encode_sequence_numba(seq_bytes)
        encoded_list.append(encoded_seq)
        boundaries.append(boundaries[-1] + len(encoded_seq))
        
    flattened_sequences = np.concatenate(encoded_list)
    boundaries = np.array(boundaries, dtype=np.int32)
    
    # 5. Execute calculations using full thread pool
    print(f"Launching parallel TRAP calculation grid on all available CPU cores...")
    matrix_results = _execute_trap_matrix_numba(
        flattened_sequences, boundaries, 
        energy_f_block, energy_r_block, 
        r0_values, w_values, 
        num_genes, num_tfs
    )
    
    # 6. Formulate Structured Matrix Dataframe and Save
    print("Structuring final dataset and committing to local disk cache...")
    df_cache = pd.DataFrame(matrix_results, index=gene_labels, columns=tf_labels)
    
    # Consolidate multiple target predictions if any genes share the exact same gene_name
    if df_cache.index.duplicated().any():
        print("Note: Found duplicate gene names. Aggregating values via mean value mapping...")
        df_cache = df_cache.groupby(df_cache.index).mean()
        
    df_cache.to_parquet(output_file)
    print(f"Pipeline complete! Cache successfully written to: {output_file}")