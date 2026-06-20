import numpy as np
import pandas as pd
from tfitpy.datasets.gene_names import get_promoter_sequence_online, get_gene_promoter_sequence_human
from tfitpy.datasets.cache_binding import BINDING_CACHE
from tfitpy.utils import ORGANISM_METADATA, INDICES_DATA

import random


def calculate_trap_affinity(
    promoter_seq,
    jaspar_matrix,
    lambda_param=0.7,
    bg_gc=0.5
):
    """Calculates the biophysical binding affinity of a transcription factor (TF) 
    to a double-stranded DNA promoter sequence using the TRAP model ((Roider et al., 2007)).
    Calculates the biophysical binding affinity of a transcription factor (TF) 
    to a double-stranded DNA promoter sequence using the TRAP model.
    """
    nuc_order = ['A', 'C', 'G', 'T']
    try:
        matrix_counts = np.array([jaspar_matrix[nuc]
                                 for nuc in nuc_order], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"Input jaspar_matrix must contain all keys A, C, G, and T. Missing: {e}")

    # 1. Apply pseudo-count to prevent log-of-zero errors
    matrix_counts += 1.0
    W = matrix_counts.shape[1]

    # Identify maximum frequency count per position to establish baseline optimal energy
    m_max = np.max(matrix_counts, axis=0)

    # Pre-compute core mismatch energy matrix (matrix contribution only)
    energy_matrix_base = np.log(m_max / matrix_counts) / lambda_param

    # 2. Compute background nucleotide frequencies based on target GC distribution
    bg_frequencies = np.array([
        (1.0 - bg_gc) / 2.0,  # Background frequency of A
        bg_gc / 2.0,          # Background frequency of C
        bg_gc / 2.0,          # Background frequency of G
        (1.0 - bg_gc) / 2.0   # Background frequency of T
    ], dtype=np.float64)

    bg_max = np.max(bg_frequencies)

    # Background energy adjustment vector: ln(Bg_w_a / Bg_w_max) / lambda
    energy_bg = np.log(bg_frequencies / bg_max) / lambda_param

    # Total energy matrix for forward strand: epsilon_matrix + epsilon_background
    energy_matrix_f = energy_matrix_base + energy_bg[:, np.newaxis]

    # Total energy matrix for reverse strand context
    # Reversing columns [:, ::-1] aligns the spatial motif matrix with the reversed window slice.
    # The row mapping remains un-flipped because rev_window_nucs handles complementation values.
    energy_matrix_r = energy_matrix_base[:, ::-1] + energy_bg[:, np.newaxis]

    # 3. Standardize and encode the target DNA sequence
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_encoded = np.array([nuc_to_idx.get(nuc, -1)
                           for nuc in promoter_seq.upper()], dtype=np.int8)

    L = len(seq_encoded)
    if L < W:
        return 0.0

    # Calibrated regression coefficient for scaling constant R0 based on motif width
    r0 = np.exp(0.585 * W - 5.66)

    total_expected_bound = 0.0
    num_windows = L - W + 1

    # 4. Slide motif matrix across all valid windows of the sequence
    for l in range(num_windows):
        window_nucs = seq_encoded[l: l + W]

        # Omit windows containing unresolved or masked characters (e.g., 'N')
        if np.any(window_nucs < 0):
            continue

        # ------------------------------------------------------------------
        # Forward Strand Mapping
        # ------------------------------------------------------------------
        # Extract forward strand energies from the pre-computed grid.
        # Uses coordinate pairing: (nucleotide_index, motif_position).
        energy_f = np.sum(energy_matrix_f[window_nucs, np.arange(W)])

        # ------------------------------------------------------------------
        # Reverse Strand Mapping
        # ------------------------------------------------------------------
        # NumPy slice [[::-1]] reverses the window order spatially.
        # Subtracting from 3 complements the base integer encodings:
        # 3 - 0(A) = 3(T), 3 - 1(C) = 2(G), 3 - 2(G) = 1(C), 3 - 3(T) = 0(A).
        rev_window_nucs = 3 - window_nucs[::-1]

        # Extract reverse strand energies using the spatially reversed matrix
        # grid, pairing complemented base rows with motif positions 0 to W-1.
        energy_r = np.sum(energy_matrix_r[rev_window_nucs, np.arange(W)])

        # Calculate local equilibrium binding affinity for individual strands
        p_f = (r0 * np.exp(-energy_f)) / (1.0 + r0 * np.exp(-energy_f))
        p_r = (r0 * np.exp(-energy_r)) / (1.0 + r0 * np.exp(-energy_r))

        # Add values together to integrate multi-strand sequence occupancy
        total_expected_bound += (p_f + p_r)

    return total_expected_bound


ORGANISM_TAX_GROUP = {
    "human": "vertebrates",
    "arabidopsis": "plants",
}


def get_all_motifs(jdb, organism):
    """
    Fetch all CORE non-redundant motifs for the given organism from JASPAR.
    Returns a list of pyjaspar motif objects.
    """
    tax_group = ORGANISM_TAX_GROUP[organism]
    return jdb.fetch_motifs(
        collection="CORE",
        tax_group=[tax_group]
    )


def get_motifs_found(jdb, organism, gene_list):
    """
    Fetch all CORE non-redundant motifs for the given organism from JASPAR.
    Returns a list of pyjaspar motif objects for the given set of genes
    """
    tax_group = ORGANISM_TAX_GROUP[organism]
    return jdb.fetch_motifs(
        collection="CORE",
        tax_group=[tax_group],
        tf_name=gene_list,
    )


def compute_tfbs_score_cache(target_gene, gene_list, cache_df):
    """
    Computes the pure TF found percentage and average TFBS affinity 
    for a single target gene and a discrete list of input genes.
    """
    target_clean = str(target_gene).strip().upper()
    input_set = set(gene_list)
    if len(input_set) == 0:
        return 0, 0, 0, None

    cache_columns = set(cache_df.columns)
    matched_tfs = input_set.intersection(cache_columns)
    tf_found_per = round(len(matched_tfs) / len(gene_list), 2) * 100

    if target_clean not in cache_df.index or len(matched_tfs) == 0:
        return 0, 0, 0, None

    observed_values = cache_df.loc[target_clean, list(matched_tfs)]
    # print(observed_values)
    m = round(float(observed_values.mean()), 5)
    s = round(float(observed_values.sum()), 5)
    # mean_affinity = np.nanmean(observed_values)

    df_new = observed_values.reset_index()
    df_new.columns = ["source", "value"]

    return tf_found_per, m, s, df_new


def TFBA_SCORE(source, target, dataset_cache, organism="human", use_pairwise_cache=True, data_path=None):
    """"""
    # print(dataset_cache)
    # print(ORGANISM_METADATA[organism])
    jkey = ORGANISM_METADATA[organism]["jaspar"]
    jaspar = dataset_cache[jkey]

    evidence = None
    summary = {
        "TFBS_affinity": 0,
        "TFBS_affinity_score": 0,
        "TF_found_per": 0,
        "TFBS_affinity_sum": 0,
        "TFBS_affinity_sum_score": 0,
    }

    n_permutations = 200
    n_items = len(source)

    if organism == "human":
        # pairwise cache is available
        if use_pairwise_cache:
            # and trap_cache is not None:
            # print()
            cache_key = ORGANISM_METADATA[organism]["pair_cache_trap"]
            trap_cache = dataset_cache[cache_key]
            p, m, s, e = compute_tfbs_score_cache(target, source, trap_cache)
            summary["TF_found_per"] = p
            summary["TFBS_affinity"] = m
            summary["TFBS_affinity_sum"] = s

            evidence = e

            if summary["TF_found_per"] > 0:
                bg_key = ORGANISM_METADATA[organism]["source_background_list"]
                background_genes = dataset_cache[bg_key]["symbol"].tolist()
                extreme_count = 0
                extreme_count_sum = 0
                for _ in range(n_permutations):
                    simulated_genes = random.sample(background_genes, n_items)

                    p1, m1, s1, e1 = compute_tfbs_score_cache(
                        target, simulated_genes, trap_cache)

                    if m1 >= m:
                        extreme_count += 1

                    if s1 >= s:
                        extreme_count_sum += 1

                p_value = (extreme_count + 1) / (n_permutations + 1)
                score = round(float(max(0.0, -np.log(p_value))), 5)
                summary["TFBS_affinity_score"] = score

                p_value_sum = (extreme_count_sum + 1) / (n_permutations + 1)
                score_sum = round(float(max(0.0, -np.log(p_value_sum))), 5)
                summary["TFBS_affinity_sum_score"] = score_sum

        else:
            target_promoter = get_gene_promoter_sequence_human(
                data_path, target)
            motif_list = get_motifs_found(jaspar, organism, source)
            # print(motif_list)
            summary["TF_found_per"] = round(
                len(motif_list) / len(source) * 100, 2)
            scores = []
            for motif in motif_list:
                # print(motif.name)
                gname = motif.name

                sc = calculate_trap_affinity(
                    target_promoter, dict(motif.counts))
                # print(sc)
                row = {
                    "source": gname,
                    "value": round(sc, 7)
                }
                scores.append(row)

            evidence = pd.DataFrame(scores)
            # print(evidence)
            observed_trap = round(float(evidence["value"].mean()), 5)
            summary["TFBS_affinity"] = observed_trap

            observed_trap_sum = round(float(evidence["value"].sum()), 5)
            summary["TFBS_affinity_sum"] = observed_trap_sum

            bg_key = ORGANISM_METADATA[organism]["source_background_list"]
            background_genes = dataset_cache[bg_key]["symbol"].tolist()
            # print(background_genes)

            if summary["TF_found_per"] > 0:
                extreme_count = 0
                extreme_count_sum = 0
                for _ in range(n_permutations):
                    simulated_genes = random.sample(background_genes, n_items)

                    motif_list1 = get_motifs_found(
                        jaspar, organism, simulated_genes)

                    gene_scores = []

                    for motif in motif_list1:
                        # print(motif.name)
                        # gname = motif.name
                        sc = calculate_trap_affinity(
                            target_promoter, dict(motif.counts))
                        # print(simulated_gene_name)
                        # print(sc)
                        gene_scores.append(round(sc, 7))

                    mean_value = 0
                    if len(gene_scores) > 0:
                        mean_value = sum(gene_scores)/len(gene_scores)

                    if mean_value >= observed_trap:
                        extreme_count += 1

                    sum_value = 0
                    if len(gene_scores) > 0:
                        sum_value = sum(gene_scores)

                    if sum_value >= observed_trap_sum:
                        extreme_count_sum += 1

                p_value = (extreme_count + 1) / (n_permutations + 1)
                score = round(float(max(0.0, -np.log(p_value))), 5)
                summary["TFBS_affinity_score"] = score

                p_value_sum = (extreme_count_sum + 1) / (n_permutations + 1)
                score_sum = round(float(max(0.0, -np.log(p_value_sum))), 5)
                summary["TFBS_affinity_sum_score"] = score_sum

    elif organism == "arabidopsis":
        gkey = ORGANISM_METADATA[organism]["gene_mapping"]
        gene_map = dataset_cache[gkey]
        gene_map_rev = {v: k for k, v in gene_map.items()}
        # print(gene_map)
        # plant gene mapping
        target_name = gene_map[target]
        # get the promoter sequence of the target

        target_promoter = get_promoter_sequence_online(target_name, organism)
        # map loci ids to gene names
        source_name = [gene_map[s] for s in source if s in gene_map.keys()]
        # print(source)
        # print(source_name)

        # look for motifs for the sources
        motif_list = get_motifs_found(jaspar, organism, source_name)

        summary["TF_found_per"] = round(
            len(motif_list) / len(source_name) * 100, 2)
        # if not motif_list:
        #    print(f"No JASPAR motifs found for: {source}")
        # print(len(motif_list))

        scores = []
        for motif in motif_list:
            # print(motif.name)
            gname = motif.name
            gname_rev = gene_map_rev[gname]

            sc = calculate_trap_affinity(target_promoter, dict(motif.counts))
            # print(sc)
            row = {
                "source_name": gname,
                "source": gname_rev,
                "value": round(sc, 7)
            }
            scores.append(row)

        evidence = pd.DataFrame(scores)
        # print(evidence)
        observed_trap = round(float(evidence["value"].mean()), 5)
        summary["TFBS_affinity"] = observed_trap

        observed_trap_sum = round(float(evidence["value"].sum()), 5)
        summary["TFBS_affinity_sum"] = observed_trap_sum

        # print(trap_value)
        # permutation testing
        bg_key = ORGANISM_METADATA[organism]["source_background_list"]
        background_genes = dataset_cache[bg_key]["Gene_ID"].tolist()
        # print(background_genes)
        if summary["TF_found_per"] > 0:
            extreme_count = 0
            extreme_count_sum = 0
            for _ in range(n_permutations):
                simulated_genes = random.sample(background_genes, n_items)
                simulated_gene_name = [gene_map[s]
                                       for s in simulated_genes if s in gene_map.keys()]
                motif_list1 = get_motifs_found(
                    jaspar, organism, simulated_gene_name)

                gene_scores = []

                for motif in motif_list1:
                    # print(motif.name)
                    gname = motif.name
                    gname_rev = gene_map_rev[gname]

                    sc = calculate_trap_affinity(
                        target_promoter, dict(motif.counts))
                    # print(simulated_gene_name)
                    # print(sc)
                    gene_scores.append(round(sc, 7))
                mean_value = 0
                sum_value = 0
                if len(gene_scores) > 0:
                    mean_value = sum(gene_scores)/len(gene_scores)
                    sum_value = sum(gene_scores)

                if mean_value >= observed_trap:
                    extreme_count += 1

                if sum_value >= extreme_count_sum:
                    extreme_count_sum += 1

            p_value = (extreme_count + 1) / (n_permutations + 1)
            score = round(float(max(0.0, -np.log(p_value))), 5)
            summary["TFBS_affinity_score"] = score

            p_value_sum = (extreme_count_sum + 1) / (n_permutations + 1)
            score_sum = round(float(max(0.0, -np.log(p_value_sum))), 5)
            summary["TFBS_affinity_sum_score"] = score_sum

    else:
        raise ValueError("Invalid human")

    return summary, evidence
