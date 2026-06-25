# datasets/go.py
import pooch
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
# from goatools.semantic import TermCounts

GO_META = {
    "FOLDER": "go",
    "ontology": {
        "url": "http://purl.obolibrary.org/obo/go/go-basic.obo",
        "filename": "go-basic.obo",
    },
    "annotations": {
        "url": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "filename_gz": "goa_human.gaf.gz",
        "filename": "goa_human.gaf",
    },
    "cache_filename": "gene2go_symbols.json",
}


def download_go(data_path, rerun=False):
    raw_path = Path(data_path) / GO_META["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    # --- ontology ---
    obo_path = raw_path / GO_META["ontology"]["filename"]
    if not obo_path.exists() or rerun:
        pooch.retrieve(
            url=GO_META["ontology"]["url"],
            known_hash=None,
            path=raw_path,
            fname=GO_META["ontology"]["filename"],
        )
        print(f"Downloaded GO ontology to {obo_path}")
    else:
        print(f"GO ontology already exists: {obo_path}")

    # --- annotations (gzipped GAF) ---
    gaf_path = raw_path / GO_META["annotations"]["filename"]
    if not gaf_path.exists() or rerun:
        pooch.retrieve(
            url=GO_META["annotations"]["url"],
            known_hash=None,
            path=raw_path,
            fname=GO_META["annotations"]["filename_gz"],
            processor=pooch.Decompress(
                method="gzip",
                name=GO_META["annotations"]["filename"],  # output filename after decompression
            ),
        )
        print(f"Downloaded and decompressed GOA annotations to {gaf_path}")
    else:
        print(f"GOA annotations already exists: {gaf_path}")

    return obo_path, gaf_path


def process_go(data_path, rerun=False):
    """Parse GAF annotations and build a symbol-keyed gene2go JSON cache.

    This is the slow step (~45s). Run once; load_go reads the cached JSON.
    """
    raw_path  = Path(data_path) / GO_META["FOLDER"]
    obo_path  = raw_path / GO_META["ontology"]["filename"]
    gaf_path  = raw_path / GO_META["annotations"]["filename"]
    cache_path = raw_path / GO_META["cache_filename"]

    if cache_path.exists() and not rerun:
        print(f"gene2go symbol cache already exists: {cache_path}")
        return

    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}. Run download_go() first.")
    if not gaf_path.exists():
        raise FileNotFoundError(f"GAF file not found: {gaf_path}. Run download_go() first.")

    print("Loading GO ontology for processing …")
    godag = GODag(str(obo_path), optional_attrs={"relationship"})

    print("Parsing GOA annotations by gene symbol")
    reader = GafReader(str(gaf_path), godag=godag)

    gene2go = defaultdict(set)
    for rec in reader.associations:
        if rec.GO_ID in godag:
            gene2go[rec.DB_Symbol].add(rec.GO_ID)

    # Serialise sets → lists for JSON
    with open(cache_path, "w") as f:
        json.dump({k: list(v) for k, v in gene2go.items()}, f)

    print(f"  {len(gene2go):,} genes cached to {cache_path}")


def load_go(data_path):
    """Load GO ontology and symbol-keyed gene2go from the processed cache.

    Returns
    -------
    dict with keys:
        godag    : GODag
        gene2go  : dict[str, set[str]]  —  HGNC symbol → set of GO term IDs
    """
    raw_path   = Path(data_path) / GO_META["FOLDER"]
    obo_path   = raw_path / GO_META["ontology"]["filename"]
    cache_path = raw_path / GO_META["cache_filename"]

    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}. Run download_go() first.")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Symbol cache not found: {cache_path}. Run process_go() first."
        )

    #print("Loading GO ontology …")
    godag = GODag(str(obo_path), optional_attrs={"relationship"})
    #print(f"  {len(godag):,} GO terms loaded")

    #print("Loading gene2go symbol cache …")
    with open(cache_path) as f:
        gene2go = {k: set(v) for k, v in json.load(f).items()}
    #print(f"  {len(gene2go):,} genes with GO annotations loaded")

    return {"godag": godag, "gene2go": gene2go}


def get_gene_products(data_path, go_term, include_children=True):
    """
    Return a DataFrame of gene products annotated to a GO term.

    Columns:
      - symbol: HGNC gene symbol
      - n_annotations: total number of GO IDs annotated to this symbol (from cache)
      - direct: True if the gene has the exact go_term in its annotations
      - via_descendant: True if the gene has any descendant (child) of go_term in its annotations
      - matching_go_ids: list of GO IDs from the gene's annotation that match (exact term and/or descendants)

    Parameters
    ----------
    data_path : str or Path
        Path where GO data (folder "go") is stored (same as used by load_go()).
    go_term : str
        GO term id, e.g. "GO:0008150".
    include_children : bool, optional
        If True, will include descendants of go_term (recommended).

    Returns
    -------
    pandas.DataFrame
        One row per gene symbol, sorted by matching count (descending).
    """
    data = load_go(data_path)
    godag = data["godag"]
    gene2go = data["gene2go"]

    if go_term not in godag:
        raise ValueError(f"Unknown GO term: {go_term}")

    # Build set of GO IDs to match: the term itself plus children if requested
    match_ids = {go_term}
    if include_children:
        # get_all_children() returns a set of child GO IDs in goatools DAG nodes
        match_ids |= set(godag[go_term].get_all_children())

    rows = []
    for symbol, gos in gene2go.items():
        gos_set = set(gos)
        matching = sorted(gos_set & match_ids)
        if not matching:
            continue
        rows.append({
            "symbol": symbol,
            "n_annotations": len(gos_set),
            "direct": (go_term in gos_set),
            "via_descendant": (len(gos_set & (match_ids - {go_term})) > 0),
            "matching_go_ids": ";".join(matching),
        })

    if not rows:
        # return empty DataFrame with expected columns
        return pd.DataFrame(columns=["symbol","n_annotations","direct","via_descendant","matching_go_ids"])

    df = pd.DataFrame(rows)
    # sort by how many matching GO IDs then by symbol
    df["n_matching"] = df["matching_go_ids"].apply(len)
    df = df.sort_values(["n_matching","n_annotations"], ascending=[False, False]).reset_index(drop=True)
    df = df[["symbol","n_annotations","direct","via_descendant","matching_go_ids"]]
    return df



####### GO (ARABIDOPSIS) #######
GO_META_ARABIDOPSIS = {
    "FOLDER": "go_arabidopsis",
    "ontology": {
        "url": "http://purl.obolibrary.org/obo/go/go-basic.obo",
        "filename": "go-basic.obo",
    },
    "annotations": {
        "url": "https://current.geneontology.org/annotations/tair.gaf.gz",
        "filename_gz": "tair.gaf.gz",
        "filename": "tair.gaf",
    },
    "cache_filename": "gene2go_symbols.json",
}


def download_go_arabidopsis(data_path, rerun=False):
    """Download GO ontology and TAIR annotations for Arabidopsis."""
    raw_path = Path(data_path) / GO_META_ARABIDOPSIS["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    # --- ontology ---
    obo_path = raw_path / GO_META_ARABIDOPSIS["ontology"]["filename"]
    if not obo_path.exists() or rerun:
        pooch.retrieve(
            url=GO_META_ARABIDOPSIS["ontology"]["url"],
            known_hash=None,
            path=raw_path,
            fname=GO_META_ARABIDOPSIS["ontology"]["filename"],
        )
        print(f"Downloaded GO ontology to {obo_path}")
    else:
        print(f"GO ontology already exists: {obo_path}")

    # --- annotations (gzipped GAF) ---
    gaf_path = raw_path / GO_META_ARABIDOPSIS["annotations"]["filename"]
    if not gaf_path.exists() or rerun:
        pooch.retrieve(
            url=GO_META_ARABIDOPSIS["annotations"]["url"],
            known_hash=None,
            path=raw_path,
            fname=GO_META_ARABIDOPSIS["annotations"]["filename_gz"],
            processor=pooch.Decompress(
                method="gzip",
                name=GO_META_ARABIDOPSIS["annotations"]["filename"],
            ),
        )
        print(f"Downloaded and decompressed TAIR annotations to {gaf_path}")
    else:
        print(f"TAIR annotations already exist: {gaf_path}")

    return obo_path, gaf_path


def process_go_arabidopsis(data_path, rerun=False):
    """Parse TAIR GAF annotations and build a locus-keyed gene2go JSON cache.

    Normalizes identifiers directly to canonical AGI locus names (e.g., AT1G14040).
    """
    raw_path  = Path(data_path) / GO_META_ARABIDOPSIS["FOLDER"]
    obo_path  = raw_path / GO_META_ARABIDOPSIS["ontology"]["filename"]
    gaf_path  = raw_path / GO_META_ARABIDOPSIS["annotations"]["filename"]
    cache_path = raw_path / GO_META_ARABIDOPSIS["cache_filename"]

    if cache_path.exists() and not rerun:
        print(f"gene2go symbol cache already exists: {cache_path}")
        return

    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}. Run download_go_arabidopsis() first.")
    if not gaf_path.exists():
        raise FileNotFoundError(f"GAF file not found: {gaf_path}. Run download_go_arabidopsis() first.")

    print("Loading GO ontology for processing …")
    godag = GODag(str(obo_path), optional_attrs={"relationship"})

    print("Parsing TAIR annotations by gene locus ID")
    reader = GafReader(str(gaf_path), godag=godag)

    # Use the official method provided by goatools to fetch {gene_id: {go_ids}}
    raw_id2gos = reader.get_id2gos()

    gene2go = {}
    for primary_id, go_ids in raw_id2gos.items():
        symbol = primary_id.strip().upper()
        
        # Handle splice variant formats safely
        if "." in symbol:
            symbol = symbol.split(".")[0]
            
        # Filter terms to ensure they exist within your current godag
        valid_gos = [go for go in go_ids if go in godag]
        if valid_gos:
            if symbol in gene2go:
                gene2go[symbol].update(valid_gos)
            else:
                gene2go[symbol] = set(valid_gos)

    # Serialize sets → lists for JSON output
    with open(cache_path, "w") as f:
        json.dump({k: list(v) for k, v in gene2go.items()}, f)

    print(f"  {len(gene2go):,} genes cached to {cache_path}")


def load_go_arabidopsis(data_path):
    """Load GO ontology and locus-keyed gene2go from the processed cache."""
    raw_path   = Path(data_path) / GO_META_ARABIDOPSIS["FOLDER"]
    obo_path   = raw_path / GO_META_ARABIDOPSIS["ontology"]["filename"]
    cache_path = raw_path / GO_META_ARABIDOPSIS["cache_filename"]

    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}. Run download_go_arabidopsis() first.")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Symbol cache not found: {cache_path}. Run process_go_arabidopsis() first."
        )

    godag = GODag(str(obo_path), optional_attrs={"relationship"})

    with open(cache_path) as f:
        gene2go = {k: set(v) for k, v in json.load(f).items()}

    return {"godag": godag, "gene2go": gene2go}




GO_DATASET = {
    "go": {
        "download": download_go,
        "process": process_go,
        "load": load_go,
    },
    "go_arabidopsis": {
        "download": download_go_arabidopsis,
        "process": process_go_arabidopsis,
        "load": load_go_arabidopsis,
    }
}