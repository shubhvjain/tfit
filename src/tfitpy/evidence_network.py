from typing import Union
from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.indices.binding import scan_promoter

import pandas as pd

def get_ppi_interactions(
    data_path,
    gene_set: list,
    source: Union[str, list] = "stringdb",
) -> pd.DataFrame:
    """
    Return all PPI interactions found between any pair of genes in *gene_set*.

    Parameters
    ----------
    data_path : str | Path
        Root data directory (same as used by load_*/process_* functions).
    gene_set : list[str]
        HGNC gene symbols to query.
    source : str | list[str]
        One or more PPI sources: 'hippie', 'stringdb', 'biogrid', or 'all'.
        Multiple sources can be passed as a comma-separated string
        (e.g. "hippie,stringdb") or a Python list.

    Returns
    -------
    pd.DataFrame
        Columns: node1, node2, edge_source, + source-specific score columns.
        Contains only intra-set edges (both endpoints in gene_set).
    """
    # --- normalise source argument ---
    if isinstance(source, str):
        if source == "all":
            sources = list(PPI_DATASETS.keys())
        else:
            sources = [s.strip() for s in source.split(",")]
    else:
        sources = list(source)

    unknown = set(sources) - set(PPI_DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown PPI source(s): {unknown}. "
                         f"Valid options: {list(PPI_DATASETS.keys())}")

    gene_set = set(gene_set)
    frames = []

    for src in sources:
        loader = PPI_DATASETS[src]["load"]
        G = loader(data_path)

        # Keep only nodes present in gene_set, then extract induced subgraph
        nodes_in_graph = gene_set & set(G.nodes())
        subgraph = G.subgraph(nodes_in_graph)

        if subgraph.number_of_edges() == 0:
            print(f"[{src}] No intra-set edges found.")
            continue

        # Convert subgraph edges to DataFrame
        rows = [
            {"node1": u, "node2": v, **attrs}
            for u, v, attrs in subgraph.edges(data=True)
        ]
        df = pd.DataFrame(rows)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["node1", "node2", "edge_source"])

    result = pd.concat(frames, ignore_index=True)
    # Ensure node1/node2 are the first columns
    front = ["node1", "node2", "edge_source"]
    rest = [c for c in result.columns if c not in front]
    res = result[front + rest].copy()
    res["evidence_type"]="ppi"
    return res



def get_tf_target_edges(
    gene_symbol: str,
    tf_symbols: list,
    data_path,
    pseudocount=0.1,
    fpr=0.001,
    datasets=None,
) -> pd.DataFrame:
    """
    Return a network edge DataFrame of TF -> target gene relationships
    derived from promoter motif scanning.

    Each row is one unique TF that has at least one binding site in the
    promoter of gene_symbol. Multiple hits from the same TF are collapsed
    into a single edge; summary statistics are retained as edge attributes.

    Parameters
    ----------
    gene_symbol : str
        Target gene whose promoter is scanned.
    tf_symbols : list[str]
        Candidate TF gene symbols to scan for.
    data_path : str | Path
        Root data directory.
    upstream, downstream : int
        Promoter window around TSS.
    pseudocount : float
        PFM pseudocount before log-odds conversion.
    fpr : float
        False positive rate per position for score thresholding.
    datasets : dict
        Cache from get_cache().

    Returns
    -------
    pd.DataFrame
        Columns: node1 (TF), node2 (target), motif_id, n_sites,
                 best_score, mean_score, best_position, edge_source
    """
    hits = scan_promoter(
        gene_symbol=gene_symbol,
        tf_symbols=tf_symbols,
        data_path=data_path,
        pseudocount=pseudocount,
        fpr=fpr,
        datasets=datasets,
    )

    if hits.empty:
        return pd.DataFrame(columns=[
            "node1", "node2", "motif_id", "n_sites",
            "best_score", "mean_score", "best_position", "edge_source"
        ])

    edges = (
        hits
        .groupby(["tf_name", "motif_id"], sort=False)
        .agg(
            n_sites=("score", "count"),
            best_score=("score", "max"),
            mean_score=("score", "mean"),
            best_position=("position", lambda x: x.iloc[hits.loc[x.index, "score"].argmax()]),
        )
        .reset_index()
        .rename(columns={"tf_name": "node1"})
    )

    edges["node2"] = gene_symbol
    edges["mean_score"] = edges["mean_score"].round(4)
    edges["edge_source"] = "jaspar_motif"

    front = ["node1", "node2", "motif_id", "n_sites",
             "best_score", "mean_score", "best_position", "edge_source"]
    return edges[front].sort_values("best_score", ascending=False).reset_index(drop=True)



def build_evidence_network(
    target_gene: str,
    gene_set: list,
    data_path,
    ppi_source: Union[str, list] = "stringdb",
    pseudocount: float = 0.1,
    fpr: float = 0.001,
    datasets: dict = None,
) -> pd.DataFrame:
    """
    Build a combined evidence DataFrame suitable for network construction.

    Combines two evidence types:
      1. PPI interactions — undirected, between any pair of genes in gene_set
      2. TF->target motif edges — directed, from any TF in gene_set to target_gene

    Parameters
    ----------
    target_gene : str
        The gene of interest. Its promoter is scanned for TF binding sites.
    gene_set : list[str]
        All genes in the analysis (candidate regulators + target).
        Used both as the PPI gene set and as the TF candidate list for
        promoter scanning.
    data_path : str | Path
        Root data directory.
    ppi_source : str | list[str]
        PPI database(s) to query. See get_ppi_interactions().
    pseudocount : float
        PFM pseudocount for PSSM scoring.
    fpr : float
        False positive rate per position for motif score thresholding.
    datasets : dict, optional
        Cache dict from get_cache(). If None, it is built automatically.
        Pass an existing cache to avoid reloading genome/JASPAR across calls.

    Returns
    -------
    pd.DataFrame
        Unified edge table with columns:
          node1, node2, edge_source, evidence_type
          + source-specific columns (score, best_score, n_sites, etc.)
        All edges have both endpoints within gene_set ∪ {target_gene}.
    """
    if datasets is None:
        raise ValueError("Cache not provided")
        datasets = get_cache(data_path)

    frames = []

    # --- PPI evidence ---
    ppi_edges = get_ppi_interactions(
        data_path=data_path,
        gene_set=gene_set,
        source=ppi_source,
    )
    if not ppi_edges.empty:
        frames.append(ppi_edges)

    # --- TF -> target motif evidence ---
    # Only scan TFs that are actually in gene_set (excluding target itself)
    tf_candidates = [g for g in gene_set if g != target_gene]

    motif_edges = get_tf_target_edges(
        gene_symbol=target_gene,
        tf_symbols=tf_candidates,
        data_path=data_path,
        pseudocount=pseudocount,
        fpr=fpr,
        datasets=datasets,
    )
    if not motif_edges.empty:
        motif_edges["evidence_type"] = "tf_motif"
        frames.append(motif_edges)

    if not frames:
        return pd.DataFrame(columns=["node1", "node2", "edge_source", "evidence_type"])

    result = pd.concat(frames, ignore_index=True)

    # Canonical column order: identity columns first, evidence metadata, then scores
    front = ["node1", "node2", "edge_source", "evidence_type"]
    rest = [c for c in result.columns if c not in front]
    return result[front + rest].reset_index(drop=True)


def analyze_network(
    edge_df: pd.DataFrame,
    target_gene: str = None,
) -> dict:
    """
    Compute network properties from a build_evidence_network() DataFrame.

    Returns a dict suitable for JSON serialisation with json.dump().

    Parameters
    ----------
    edge_df : pd.DataFrame
        Output of build_evidence_network().
    target_gene : str, optional
        If provided, included in the summary metadata.

    Returns
    -------
    dict with keys:
        summary       — node/edge counts, density, connected components
        centrality    — degree/betweenness/closeness for the full graph
        centrality_ppi    — same metrics on the PPI-only subgraph
        centrality_motif  — same metrics on the motif-only subgraph
        edges         — list of edge records (node1, node2, edge_source, evidence_type)
    """
    import networkx as nx
    import json

    def _centrality(G: nx.Graph) -> dict:
        if G.number_of_nodes() == 0:
            return {}
        deg  = nx.degree_centrality(G)
        btw  = nx.betweenness_centrality(G, normalized=True)
        # closeness requires connected graph — compute on largest component
        if isinstance(G, nx.DiGraph):
            Gu = G.to_undirected()
        else:
            Gu = G
        largest = Gu.subgraph(max(nx.connected_components(Gu), key=len))
        clo = nx.closeness_centrality(largest)

        # merge into per-node dict
        all_nodes = set(deg) | set(btw) | set(clo)
        return {
            node: {
                "degree":      round(deg.get(node, 0.0), 6),
                "betweenness": round(btw.get(node, 0.0), 6),
                "closeness":   round(clo.get(node, 0.0), 6),
            }
            for node in sorted(all_nodes)
        }

    def _summary(G: nx.Graph, label: str) -> dict:
        n = G.number_of_nodes()
        e = G.number_of_edges()
        Gu = G.to_undirected() if isinstance(G, nx.DiGraph) else G
        comps = list(nx.connected_components(Gu))
        return {
            "graph":       label,
            "nodes":       n,
            "edges":       e,
            "density":     round(nx.density(G), 6) if n > 1 else 0.0,
            "components":  len(comps),
            "largest_component_size": max(len(c) for c in comps) if comps else 0,
        }

    # --- build graphs ---
    full_G  = nx.MultiDiGraph()
    ppi_G   = nx.Graph()
    motif_G = nx.DiGraph()

    for _, row in edge_df.iterrows():
        u   = row["node1"]
        v   = row["node2"]
        src = row.get("edge_source", "")
        ev  = row.get("evidence_type", "")

        full_G.add_edge(u, v, edge_source=src)

        if ev == "ppi" or "ppi" in src:
            ppi_G.add_edge(u, v)
        if ev == "tf_motif" or "motif" in src:
            motif_G.add_edge(u, v)

    # --- edge records for JSON ---
    edge_records = edge_df[
        [c for c in ["node1", "node2", "edge_source", "evidence_type"] if c in edge_df.columns]
    ].to_dict(orient="records")

    result = {
        "target_gene": target_gene,
        "summary": {
            "full":  _summary(full_G,  "full"),
            "ppi":   _summary(ppi_G,   "ppi"),
            "motif": _summary(motif_G, "motif"),
        },
        "centrality": {
            "full":  _centrality(full_G.to_undirected()),
            "ppi":   _centrality(ppi_G),
            "motif": _centrality(motif_G),
        },
        "edges": edge_records,
    }

    return result