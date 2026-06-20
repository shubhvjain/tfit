# datasets/binding.py
import pooch
import pandas as pd
from pathlib import Path
import json
import sqlite3
from pyjaspar import jaspardb


JASPAR = {
    "FOLDER": "jaspar",
    "DB_FILE": "JASPAR2026.sqlite3",
    "PFM_FILE": "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "URL_DB": "https://mencius.uio.no/JASPAR/JASPAR_metadata/2026/JASPAR2026.sqlite3",
    "URL_PFM": "https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
}

def download_jaspar(data_path, rerun=False):
    """Download JASPAR2026 SQLite database and vertebrates CORE PFM file."""
    raw_path = Path(data_path) / JASPAR["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    db_file = raw_path / JASPAR["DB_FILE"]
    if not db_file.exists() or rerun:
        pooch.retrieve(
            url=JASPAR["URL_DB"],
            known_hash=None,
            path=raw_path,
            fname=JASPAR["DB_FILE"]
        )
        print(f"Downloaded JASPAR SQLite DB to {db_file}")
    else:
        print(f"JASPAR DB already exists: {db_file}")

    pfm_file = raw_path / JASPAR["PFM_FILE"]
    if not pfm_file.exists() or rerun:
        pooch.retrieve(
            url=JASPAR["URL_PFM"],
            known_hash=None,
            path=raw_path,
            fname=JASPAR["PFM_FILE"]
        )
        print(f"Downloaded JASPAR PFM file to {pfm_file}")
    else:
        print(f"JASPAR PFM already exists: {pfm_file}")

def process_jasper(data_path, rerun=False):
    """"""

def load_jaspar(data_path):
    """Return an open SQLite connection to the JASPAR database."""
    db_file = Path(data_path) / JASPAR["FOLDER"] / JASPAR["DB_FILE"]
    if not db_file.exists():
        raise FileNotFoundError(f"{JASPAR['DB_FILE']} not found. Run download_jaspar() first.")
    jdb = jaspardb(sqlite_db_path=str(db_file))
    return jdb


# datasets/binding.py  (additions)

JASPAR_PLANT = {
    "FOLDER": "jaspar_plant",
    "DB_FILE": "JASPAR2026.sqlite3",
    "PFM_FILE": "JASPAR2026_CORE_plants_non-redundant_pfms_jaspar.txt",
    "URL_DB": "https://mencius.uio.no/JASPAR/JASPAR_metadata/2026/JASPAR2026.sqlite3",
    "URL_PFM": "https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_plants_non-redundant_pfms_jaspar.txt",
}


def download_jaspar_plant(data_path, rerun=False):
    """Download JASPAR2026 SQLite database and plants CORE PFM file."""
    raw_path = Path(data_path) / JASPAR_PLANT["FOLDER"]
    raw_path.mkdir(parents=True, exist_ok=True)

    db_file = raw_path / JASPAR_PLANT["DB_FILE"]
    if not db_file.exists() or rerun:
        pooch.retrieve(
            url=JASPAR_PLANT["URL_DB"],
            known_hash=None,
            path=raw_path,
            fname=JASPAR_PLANT["DB_FILE"]
        )
        print(f"Downloaded JASPAR plant SQLite DB to {db_file}")
    else:
        print(f"JASPAR plant DB already exists: {db_file}")

    pfm_file = raw_path / JASPAR_PLANT["PFM_FILE"]
    if not pfm_file.exists() or rerun:
        pooch.retrieve(
            url=JASPAR_PLANT["URL_PFM"],
            known_hash=None,
            path=raw_path,
            fname=JASPAR_PLANT["PFM_FILE"]
        )
        print(f"Downloaded JASPAR plant PFM file to {pfm_file}")
    else:
        print(f"JASPAR plant PFM already exists: {pfm_file}")


def process_jaspar_plant(data_path, rerun=False):
    """"""


def load_jaspar_plant(data_path):
    """Return an open SQLite connection to the JASPAR plant database."""
    db_file = Path(data_path) / JASPAR_PLANT["FOLDER"] / JASPAR_PLANT["DB_FILE"]
    if not db_file.exists():
        raise FileNotFoundError(f"{JASPAR_PLANT['DB_FILE']} not found. Run download_jaspar_plant() first.")
    jdb = jaspardb(sqlite_db_path=str(db_file))
    return jdb


def get_jasper_path(data_path,organism="human"):
    """
    """
    if organism=="human":
        jaspar_db  = Path(data_path) / JASPAR["FOLDER"] / JASPAR["DB_FILE"]
    elif organism == "arabidopsis":
        jaspar_db = Path(data_path) / JASPAR_PLANT["FOLDER"] / JASPAR_PLANT["DB_FILE"]
    else:
        raise ValueError("Invalid organism")
    
    return jaspar_db


BINDING_DATASET = {
    "jaspar": {
        "download": download_jaspar,
        "process": process_jasper,
        "load": load_jaspar,
    },
     "jaspar_plant": {
        "download": download_jaspar_plant,
        "process": process_jaspar_plant,
        "load": load_jaspar_plant,
    },
}