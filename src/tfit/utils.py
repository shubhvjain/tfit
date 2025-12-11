"""Utility functions for data paths, downloads, and module config merging."""
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tqdm import tqdm
import zipfile

from .config import expand_path, DEFAULT_CONFIG, DEFAULT_DATA_DIR 

def resolve_module_config(
    user_config: Optional[Dict[str, Any]],
    module_key: str,
    module_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Return config dict with same shape as input, plus data_path as Path.

    - Start from DEFAULT_CONFIG.
    - Overlay user_config (if any).
    - Ensure cfg["data_path"] is a Path (expanding env vars / ~, or DEFAULT_DATA_DIR).
    - Ensure cfg[module_key] exists and is a dict.
    - Merge module_defaults into cfg[module_key] only for missing keys.
    """
    # 1) Start from global defaults and overlay user config into a NEW dict
    cfg: Dict[str, Any] = DEFAULT_CONFIG.copy()
    if user_config:
        cfg.update(user_config)

    # 2) Resolve data_path -> Path
    data_path_val = cfg.get("data_path")
    if isinstance(data_path_val, str) and data_path_val.strip():
        data_path = expand_path(data_path_val)
        data_path.mkdir(parents=True, exist_ok=True)
    else:
        data_path = DEFAULT_DATA_DIR
    cfg["data_path"] = data_path  # now always a Path

    # 3) Ensure module section exists and merge defaults
    section = cfg.get(module_key)
    if not isinstance(section, dict):
        section = {}
    merged_section = module_defaults.copy()
    merged_section.update(section)  # user overrides defaults
    cfg[module_key] = merged_section

    return cfg


def verify_hash(path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify file integrity using a cryptographic hash.

    Args:
        path: Path to the file whose integrity should be verified.
        expected_hash: Expected hexadecimal digest string.
        algorithm: Hash algorithm name understood by :mod:`hashlib`,
            such as ``"sha256"`` or ``"md5"``.

    Returns:
        bool: ``True`` if the computed hash matches ``expected_hash``,
        otherwise ``False``.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def download_file(
    url: str,
    filename: str,
    expected_hash: Optional[str] = None,
    chunk_size: int = 8192,
    base_dir: Optional[Path] = None,
) -> Path:
    """Download a file with resume, progress bar, and optional hash check.

    This function is "config-agnostic": callers are expected to resolve
    their own base directory (for example via :func:`resolve_module_config`)
    and pass it as ``base_dir``.

    Args:
        url: HTTP/HTTPS URL to download from.
        filename: Name of the local file relative to ``base_dir``.
        expected_hash: Optional expected hash (hex digest). If provided,
            the file will be verified after download using
            :func:`verify_hash`. If the file already exists and matches
            this hash, it will not be re-downloaded.
        chunk_size: Number of bytes to read per chunk when streaming the
            response.
        base_dir: Directory in which to place the downloaded file. If
            ``None``, :data:`DEFAULT_DATA_DIR` is used.

    Returns:
        Path: The path to the downloaded (or already-existing) file.

    Raises:
        ValueError: If ``expected_hash`` is provided and the downloaded file
            does not match the expected hash.
        requests.HTTPError: If the HTTP request fails.
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR

    path = base_dir / filename

    # Check if complete file exists and verify hash.
    if path.exists() and (expected_hash is None or verify_hash(path, expected_hash)):
        print(f"{filename} already verified: {path}")
        return path

    # Attempt to resume download if file partially exists.
    resume_byte_pos = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos else {}

    print(f"Downloading {filename} ({'resuming' if resume_byte_pos else 'starting'})...")
    path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=headers, stream=True) as r, \
            open(path, "ab" if resume_byte_pos else "wb") as f, \
            tqdm(
                desc=filename,
                total=None,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:

        r.raise_for_status()
        _ = int(r.headers.get("content-length", 0)) + resume_byte_pos

        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    # Verify hash if provided.
    if expected_hash and not verify_hash(path, expected_hash):
        raise ValueError(f"Hash mismatch for {filename}")

    print(f"{filename} ready: {path}")
    return path


def download_zip(
    url: str,
    extract_folder: str,
    expected_hash: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Download a ZIP archive and extract it to a folder.

    The archive is downloaded into ``base_dir`` and extracted into
    ``base_dir / extract_folder``. If the extraction folder already
    exists and contains at least one item, the download is skipped.

    Args:
        url: HTTP/HTTPS URL of the ZIP file.
        extract_folder: Name of the folder under ``base_dir`` where the
            contents will be extracted.
        expected_hash: Optional expected hash (hex digest) for the ZIP
            file. If provided, the downloaded ZIP is verified using
            :func:`verify_hash`.
        base_dir: Base directory to use for both the temporary ZIP file
            and the extraction folder. If ``None``, :data:`DEFAULT_DATA_DIR`
            is used.

    Returns:
        Path: The path to the extraction directory. Guaranteed to exist.

    Raises:
        ValueError: If ``expected_hash`` is provided and the downloaded file
            does not match the expected hash.
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP.
        requests.HTTPError: If the HTTP request fails.
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR

    extract_path = base_dir / extract_folder

    # If folder already exists and is non-empty, assume it's ready.
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"{extract_folder} already exists with contents")
        return extract_path

    # Download ZIP to a temporary location first.
    zip_path = base_dir / f"temp_{extract_folder}.zip"

    # Remove any previous partial download.
    if zip_path.exists():
        zip_path.unlink()

    print(f"Downloading ZIP to {extract_folder}...")
    download_file(
        url,
        zip_path.name,
        expected_hash=expected_hash,
        base_dir=base_dir,
    )

    # Extract archive.
    print(f"Extracting to {extract_path}...")
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up temporary ZIP.
    zip_path.unlink()
    print(f"{extract_folder} extracted: {extract_path}")
    return extract_path
