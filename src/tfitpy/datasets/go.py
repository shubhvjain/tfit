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

    print("Loading GO ontology …")
    godag = GODag(str(obo_path), optional_attrs={"relationship"})
    print(f"  {len(godag):,} GO terms loaded")

    print("Loading gene2go symbol cache …")
    with open(cache_path) as f:
        gene2go = {k: set(v) for k, v in json.load(f).items()}
    print(f"  {len(gene2go):,} genes with GO annotations loaded")

    return {"godag": godag, "gene2go": gene2go}


GO_DATASET = {
    "go": {
        "download": download_go,
        "process": process_go,
        "load": load_go,
    }
}