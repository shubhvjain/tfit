"""Utility functions """


def generate_tf_pairs(gene_cluster):
    """Generate all unique pairs from comma-separated gene cluster."""
    genes = [g.strip() for g in gene_cluster]
    return [(g1, g2) for i, g1 in enumerate(genes) for g2 in genes[i+1:]]