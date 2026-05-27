import numpy as np

def calculate_trap_affinity_test(
    promoter_seq: str, 
    jaspar_matrix: dict, 
    lambda_param: float = 0.7
) -> float:
    """Calculates the biophysical binding affinity of a TF to a promoter sequence.

    This function implements the TRAP model (Roider et al., 2007) to estimate the 
    expected number of bound transcription factor molecules on a double-stranded 
    DNA sequence.

    Args:
        promoter_seq: A string representing the DNA sequence (A, C, G, T).
        jaspar_matrix: A dictionary containing lists or numpy arrays of raw position
          frequency counts for keys 'A', 'C', 'G', 'T'. Each array has length W.
        lambda_param: The scaling parameter for mismatch energy (default: 0.7).

    Returns:
        The total expected number of bound TF molecules (float).
    """
    # 1. Standardize matrix layout and apply pseudo-counts
    nuc_order = ['A', 'C', 'G', 'T']
    try:
        matrix_counts = np.array([jaspar_matrix[nuc] for nuc in nuc_order], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"Input jaspar_matrix must contain all keys A, C, G, and T. Missing: {e}")

    # Add a pseudo-count of pi = 1 to prevent zero probabilities (Section 2.2)
    matrix_counts += 1.0
    
    # Identify the maximum observed frequency for each position i (m_i,max)
    m_max = np.max(matrix_counts, axis=0)
    
    # Pre-compute the physical mismatch energy matrix: ln(m_max / m_alpha) / lambda
    # Resulting shape: [4, W]
    energy_matrix = np.log(m_max / matrix_counts) / lambda_param
    
    # 2. Convert and validate sequence input
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_encoded = np.array([nuc_to_idx.get(nuc, -1) for nuc in promoter_seq.upper()], dtype=np.int8)
    
    L = len(seq_encoded)
    W = matrix_counts.shape[1]
    
    # If the promoter sequence is shorter than the motif width, binding cannot occur
    if L < W:
        return 0.0
        
    # Calculate R0 based on motif width W using calibrated regression
    r0 = np.exp(0.585 * W - 5.66)
    
    total_expected_bound = 0.0
    num_windows = L - W + 1
    
    # 3. Slide the motif matrix along the sequence windows
    for l in range(num_windows):
        window_nucs = seq_encoded[l : l + W]
        
        # Skip windows that contain undefined characters (e.g., 'N' or gaps masked as -1)
        if np.any(window_nucs < 0):
            continue
            
        # Extract forward strand mismatch energy penalties (Equation 4)
        energy_f = np.sum(energy_matrix[window_nucs, np.arange(W)])
        
        # Compute reverse complement window indices: (3 - index) and reversed in direction
        rev_window_nucs = 3 - window_nucs[::-1]
        
        # Extract reverse strand mismatch energy penalties
        energy_r = np.sum(energy_matrix[rev_window_nucs, np.arange(W)])
        
        # Compute local equilibrium binding probabilities using Equation 3
        # e^(-beta * E) is calculated here where beta is integrated via lambda scaling
        exp_neg_e_f = np.exp(-energy_f)
        exp_neg_e_r = np.exp(-energy_r)
        
        p_f = (r0 * exp_neg_e_f) / (1.0 + r0 * exp_neg_e_f)
        p_r = (r0 * exp_neg_e_r) / (1.0 + r0 * exp_neg_e_r)
        
        # Combine probabilities from both strands assuming structural independence (Equation 6)
        p_total = p_f + p_r - (p_f * p_r)
        
        total_expected_bound += p_total
        
    return total_expected_bound



if __name__ == "__main__":
    # Example transcription factor profile (Width W = 6)
    sample_jaspar_matrix = {
        "A": [15,  2,  1, 24,  0, 11],
        "C": [ 3, 20,  0,  1,  2,  4],
        "G": [ 2,  1, 22,  0, 25,  3],
        "T": [ 5,  2,  2,  0,  1,  7]
    }
    
    # Target promoter sequence containing variations of the optimal motif
    sample_promoter = "ACGTGATCGCAACGCATAGCTAGC"
    
    affinity_score = calculate_trap_affinity(sample_promoter, sample_jaspar_matrix)
    print(f"Calculated Binding Affinity Score (<N>): {affinity_score:.6f}")


import numpy as np

def calculate_trap_affinity1(
    promoter_seq: str, 
    jaspar_matrix: dict, 
    lambda_param: float = 0.7,
    bg_gc: float = 0.5
) -> float:
    """Calculates the biophysical binding affinity of a transcription factor (TF) 
    to a double-stranded DNA promoter sequence using the TRAP model ((Roider et al., 2007)).
    """
    nuc_order = ['A', 'C', 'G', 'T']
    try:
        matrix_counts = np.array([jaspar_matrix[nuc] for nuc in nuc_order], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"Input jaspar_matrix must contain all keys A, C, G, and T. Missing: {e}")

    # 1. Apply pseudo-count to prevent log-of-zero errors
    matrix_counts += 1.0
    W = matrix_counts.shape[1]
    
    # Identify maximum frequency count per position to establish baseline optimal energy
    m_max = np.max(matrix_counts, axis=0)
    
    # Pre-compute core mismatch energy matrix
    energy_matrix = np.log(m_max / matrix_counts) / lambda_param
    
    # 2. Compute background nucleotide frequencies based on target GC distribution
    bg_frequencies = np.array([
        (1.0 - bg_gc) / 2.0,  # Background frequency of A
        bg_gc / 2.0,          # Background frequency of C
        bg_gc / 2.0,          # Background frequency of G
        (1.0 - bg_gc) / 2.0   # Background frequency of T
    ], dtype=np.float64)
    
    # Derive background chemical potential offsets relative to maximum baseline
    bg_max = np.max(bg_frequencies)
    energy_bg = np.log(bg_frequencies / bg_max) / lambda_param
    
    # Calibrate matrix by subtracting background adjustments across all motif positions
    energy_matrix -= energy_bg[:, np.newaxis]

    # 3. Standardize and encode the target DNA sequence
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_encoded = np.array([nuc_to_idx.get(nuc, -1) for nuc in promoter_seq.upper()], dtype=np.int8)
    
    L = len(seq_encoded)
    if L < W:
        return 0.0
        
    # Calibrated regression coefficient for scaling constant R0 based on motif width
    r0 = np.exp(0.585 * W - 5.66)
    
    total_expected_bound = 0.0
    num_windows = L - W + 1
    
    # 4. Slide motif matrix across all valid windows of the sequence
    for l in range(num_windows):
        window_nucs = seq_encoded[l : l + W]
        
        # Omit windows containing unresolved or masked characters (e.g., 'N')
        if np.any(window_nucs < 0):
            continue
            
        # Extract forward strand mismatch energy sum via coordinate index mapping
        energy_f = np.sum(energy_matrix[window_nucs, np.arange(W)])
        
        # Generate reverse complement window indexes (Complement: 3 - index, Reverse: [::-1])
        rev_window_nucs = 3 - window_nucs[::-1]
        
        # Extract reverse strand mismatch energy sum
        energy_r = np.sum(energy_matrix[rev_window_nucs, np.arange(W)])
        
        # Calculate local equilibrium binding affinity for individual strands
        p_f = (r0 * np.exp(-energy_f)) / (1.0 + r0 * np.exp(-energy_f))
        p_r = (r0 * np.exp(-energy_r)) / (1.0 + r0 * np.exp(-energy_r))
        
        # Add values together to integrate multi-strand sequence occupancy
        total_expected_bound += (p_f + p_r)
        
    return total_expected_bound



def calculate_trap_affinity(
    promoter_seq: str, 
    jaspar_matrix: dict, 
    lambda_param: float = 0.7,
    bg_gc: float = 0.5
) -> float:
    """Calculates the biophysical binding affinity of a transcription factor (TF) 
    to a double-stranded DNA promoter sequence using the TRAP model ((Roider et al., 2007)).
    Calculates the biophysical binding affinity of a transcription factor (TF) 
    to a double-stranded DNA promoter sequence using the TRAP model.
    """
    nuc_order = ['A', 'C', 'G', 'T']
    try:
        matrix_counts = np.array([jaspar_matrix[nuc] for nuc in nuc_order], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"Input jaspar_matrix must contain all keys A, C, G, and T. Missing: {e}")

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
    seq_encoded = np.array([nuc_to_idx.get(nuc, -1) for nuc in promoter_seq.upper()], dtype=np.int8)
    
    L = len(seq_encoded)
    if L < W:
        return 0.0
        
    # Calibrated regression coefficient for scaling constant R0 based on motif width
    r0 = np.exp(0.585 * W - 5.66)
    
    total_expected_bound = 0.0
    num_windows = L - W + 1
    
    # 4. Slide motif matrix across all valid windows of the sequence
    for l in range(num_windows):
        window_nucs = seq_encoded[l : l + W]
        
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