from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from ..utils import download_zip, resolve_module_config

META: Dict[str, Any] = {
    "url": "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.252/BIOGRID-ORGANISM-5.0.252.mitab.zip",
    "about": "PSI MITAB level 2.5 compatible. Human (Homo sapiens) subset.",
    "folder_name": "biogrid",  # where we extract the zip
    "file": "BIOGRID-ORGANISM-Homo_sapiens-5.0.252.mitab.txt",  # file inside the zip
    "columns": [ 'ID_A', 'ID_B', 'Alt_ID_A', 'Alt_ID_B', 'Aliases_A', 'Aliases_B', 'Detection_Methods', 'First_Authors', 'Publication_IDs', 'Taxonomy_IDs_A', 'Taxonomy_IDs_B', 'Interaction_Types','Source_Databases', 'Interaction_IDs', 'Confidence_Scores'],
    
}


def _file_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Return full path to the BioGRID human MITAB file."""
    cfg = resolve_module_config(config, "biogrid", META)
    base_dir: Path = cfg["data_path"]
    return base_dir / META["folder_name"] / META["file"]


def is_ready(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if the BioGRID human MITAB file already exists."""
    return _file_path(config).exists()


def download(config: Optional[Dict[str, Any]] = None) -> None:
    """Download and extract the BioGRID human MITAB dataset if missing."""
    cfg = resolve_module_config(config, "biogrid", META)
    base_dir: Path = cfg["data_path"]

    extract_folder = META["folder_name"]

    # This will create base_dir / extract_folder and extract the zip there
    download_zip(
        url=META["url"],
        extract_folder=extract_folder,
        base_dir=base_dir,
    )


def get(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load BioGRID human MITAB data (downloads/extracts if missing)."""
    if not is_ready(config):
        download(config)

    filepath = _file_path(config)
    print(f"Loading BioGRID human data from {filepath}...")

    # MITAB 2.5 is tab-separated; you can adjust columns later as needed
    df = pd.read_csv(filepath, sep='\t', low_memory=False, comment='#', header = None, names = META["columns"])
    return df
