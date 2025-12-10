import platformdirs
from pathlib import Path
import requests
from typing import Optional
from tqdm import tqdm
import hashlib
import zipfile
import io

TOOL_NAME="tfit"
TOOL_AUTHOR="SVJ"
DATA_DIR = Path(platformdirs.user_data_dir(TOOL_NAME,TOOL_AUTHOR))
DATA_DIR.mkdir(exist_ok=True)

def download_file(url: str, filename: str, expected_hash: Optional[str] = None, 
                  chunk_size: int = 8192) -> Path:
    """Robust download for  data with progress, resume, verification"""
    path = DATA_DIR / filename
    
    # Check if complete file exists and verify hash
    if path.exists() and (expected_hash is None or verify_hash(path, expected_hash)):
        print(f"{filename} already verified: {path}")
        return path
    
    # Resume download
    resume_byte_pos = path.stat().st_size if path.exists() else 0
    headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
    
    print(f"Downloading {filename} ({'resuming' if resume_byte_pos else 'starting'})...")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with requests.get(url, headers=headers, stream=True) as r, \
         open(path, 'ab' if resume_byte_pos else 'wb') as f, \
         tqdm(desc=filename, total=None, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0)) + resume_byte_pos
        
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    # Verify hash if provided
    if expected_hash and not verify_hash(path, expected_hash):
        raise ValueError(f"Hash mismatch for {filename}")
    
    print(f"{filename} ready: {path}")
    return path


def download_zip(url: str, extract_folder: str, expected_hash: Optional[str] = None) -> Path:
    """Download ZIP, extract to specific folder if not already exists"""
    extract_path = DATA_DIR / extract_folder
    
    # Check if extracted folder already exists and has contents
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"{extract_folder} already exists with contents")
        return extract_path
    
    # Download ZIP to temp location first
    zip_path = DATA_DIR / f"temp_{extract_folder}.zip"
    
    # Remove partial download if exists
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"Downloading ZIP to {extract_folder}...")
    download_file(url, zip_path.name, expected_hash)
    
    # Extract
    print(f"Extracting to {extract_path}...")
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Clean up temp ZIP
    zip_path.unlink()
    print(f"{extract_folder} extracted: {extract_path}")
    return extract_path

def verify_hash(path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """Verify file integrity"""
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


