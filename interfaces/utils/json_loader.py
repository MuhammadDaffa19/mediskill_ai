# interfaces/utils/json_loader.py
import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# BASE_DIR points to the "interfaces" folder (one level up from this utils module)
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

def _abs_path_for(relative_path: str) -> str:
    """
    Turn a relative interface path (relative to interfaces/) into absolute path.
    Accepts either:
        - "global/global_quickpanel.json"
        - "/full/path/to/file.json" (absolute, returns unchanged)
    """
    if not relative_path:
        raise ValueError("relative_path must be non-empty")

    # if user accidentally passed an absolute path -> return directly
    if os.path.isabs(relative_path):
        return relative_path

    # normalize and join with BASE_DIR
    # allow leading './' or '../' in relative_path
    return os.path.normpath(os.path.join(BASE_DIR, relative_path))

def load_json(relative_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON file located relative to the interfaces/ directory.

    Example:
        load_json("global/global_quickpanel.json")

    Returns parsed JSON (usually a dict). Raises FileNotFoundError or ValueError on issues.
    """
    full_path = _abs_path_for(relative_path)
    if not os.path.exists(full_path):
        logger.error("JSON file not found: %s", full_path)
        raise FileNotFoundError(f"File not found: {full_path}")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.exception("Invalid JSON in %s: %s", full_path, e)
        raise
    except Exception as e:
        logger.exception("Error loading JSON file %s: %s", full_path, e)
        raise
