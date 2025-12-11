"""Configuration helpers for tfit.

This module defines the canonical shape of the global configuration dict
used by both the CLI and the Python API. The config is just a plain
dictionary, typically structured like::

    {
        "data_path": "~/tfit-data",
        "hippie": {
            
        },
        "biomart": {
            
        }
        # ...
    }

The CLI can load this from a JSON file, while library users can build
the dict directly in Python.
"""
import json
from pathlib import Path
from typing import Any, Dict
import os
import platformdirs

DEFAULT_CONFIG: Dict[str, Any] = {

}

TOOL_NAME = "tfitpy"
TOOL_AUTHOR = "SVJ"

# Default base directory for all data when no data_path is provided in config.
DEFAULT_DATA_DIR = Path(platformdirs.user_data_dir(TOOL_NAME, TOOL_AUTHOR))
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def expand_path(path_str: str) -> Path:
    """Expand environment variables and ``~`` in a path string.

    Args:
        path_str: Raw path string that may contain environment variables
            or a leading tilde.

    Returns:
        Path: A pathlib.Path object with variables and tilde expanded.
    """
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a global config dict from a JSON file.

    This is a thin helper primarily for CLI usage. It expects the JSON
    file itself to already be in the "global config" format used by the
    rest of the library (i.e. with keys like ``"data_path"``, ``"hippie"``,
    etc.). No further nesting is enforced here.

    Example JSON file content::

        {
          "data_path": "$HOME/.cache/tfit",
          "hippie": {
            "filename": "hippie_ppi.txt"
          }
        }

    Args:
        path: Filesystem path to the JSON config file. May contain
            environment variables or ``~``, which will be expanded.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    cfg_path = expand_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level config JSON must be an object (dict).")

    return data


def blank_config() -> Dict[str, Any]:
    """Return empty config template for user customization."""
    return {
        "data_path": str(DEFAULT_DATA_DIR),  # points to platformdirs default
        # Add module defaults here later as needed
        # "hippie": {},
        # "biomart": {},
        # "stringdb": {}
    }

def save_blank_config(path: str | Path) -> Path:
    """Save blank config template to file."""
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = blank_config()
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    return config_path