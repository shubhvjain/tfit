from itertools import combinations, product
import pandas as pd

def generate_pairs(module, save_order=False, duplicate_pairs=False):
    """
    module: list[str]
    returns: DataFrame with columns ['gene1', 'gene2']
    """
    # 1) build raw pairs
    if save_order:
        # ordered pairs, including (A,B) and (B,A), excluding self-pairs
        pairs = [(a, b) for a, b in product(module, module) if a != b]
    else:
        # unordered unique pairs, (A,B) but not (B,A)
        pairs = list(combinations(module, 2))  # no self-pairs[web:22]

    df = pd.DataFrame(pairs, columns=["gene1", "gene2"])

    # 2) deduplicate if requested
    if not duplicate_pairs:
        df = df.drop_duplicates(ignore_index=True)  # removes repeated rows[web:33]

    return df
