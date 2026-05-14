import numpy as np
import igraph as ig

def compute_module_network_metrics(M,t, ppi, null_graphs, mapping, debug=False):
    """
    M: List of gene symbols
    ppi: Original aligned igraph object
    null_graphs: List of  pre-built igraph objects
    mapping: Symbol -> Int dictionary
    debug: If True, prints mapping and subgraph samples
    """
    # 1. Map symbols to indices
    m_indices = [mapping[g] for g in M if g in mapping]
    n_m = len(m_indices)
    target_idx = mapping.get(t)
    
    if debug:
        print(f"--- DEBUG MODE ---")
        print(f"Module size (input): {len(M)} | Mapped indices: {n_m}")
        # Print first 3 mappings to verify
        sample_genes = [g for g in M if g in mapping][:3]
        for g in sample_genes:
            print(f"  Gene '{g}' -> Index {mapping[g]}")
    
    if n_m < 2:
        if debug: print("Insufficient nodes (n < 2). Returning zeros.")
        return 0.0, 0.0, 0.0, 0.0

    possible_edges = (n_m * (n_m - 1)) / 2

    # --- 2. Observed Metrics ---
    sub_obs = ppi.induced_subgraph(m_indices)
    e_obs = sub_obs.ecount()
    
    obs_density = e_obs / possible_edges if possible_edges > 0 else 0.0
    obs_lcc_count = sub_obs.connected_components().giant().vcount() if e_obs > 0 else 1
    obs_lcc_ratio = obs_lcc_count / n_m

    obs_tc_count = 0
    if target_idx is not None:
        # Get neighbors of target and find intersection with module indices
        target_neighbors = set(ppi.neighbors(target_idx))
        obs_tc_count = len(target_neighbors.intersection(m_indices))
    
    obs_tc_ratio = obs_tc_count / n_m



    if debug:
        print(f"Observed PPI: Edges={e_obs}, LCC Count={obs_lcc_count}, Density={obs_density:.4f}")

    # --- 3. Null Distribution ---
    density_hits = 0
    lcc_hits = 0
    tc_hits = 0
    R = len(null_graphs)
    
    for i, g_null in enumerate(null_graphs):
        sub_null = g_null.induced_subgraph(m_indices)
        e_null = sub_null.ecount()
        
        # Null Density
        null_density = e_null / possible_edges if possible_edges > 0 else 0.0
        if null_density >= obs_density:
            density_hits += 1
            
        # Null LCC Ratio
        null_lcc_count = sub_null.connected_components().giant().vcount() if e_null > 0 else 1
        null_lcc_ratio = null_lcc_count / n_m
        
        if null_lcc_ratio >= obs_lcc_ratio:
            lcc_hits += 1

        null_target_neighbors = set(g_null.neighbors(target_idx))
        null_tc_count = len(null_target_neighbors.intersection(m_indices))
        if (null_tc_count / n_m) >= obs_tc_ratio:
            tc_hits += 1
            
        # Peek at the first null model to verify variation
        if debug:
            print(f"Null Model [{i}]: Edges={e_null}, LCC Count={null_lcc_count}, Density={null_density:.4f}")

    # --- 4. Final Scores (-ln(p)) ---
    p_density = (density_hits + 1) / (R + 1)
    density_score = max(0.0, -np.log(p_density))
    
    p_lcc = (lcc_hits + 1) / (R + 1)
    lcc_score = max(0.0, -np.log(p_lcc))

    p_tc = (tc_hits + 1) / (R + 1)
    target_connectivity_score = -np.log(p_tc)

    if debug:
        print(f"Final Scores: Density_Score={density_score:.4f}, LCC_Score={lcc_score:.4f}")
        print(f"------------------")

    return obs_density, density_score, obs_lcc_ratio, lcc_score, obs_tc_ratio, target_connectivity_score