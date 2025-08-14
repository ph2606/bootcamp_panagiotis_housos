# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

def load_env(dotenv_path: str | None = None) -> bool:
    """
    Load environment variables from .env at repo root by default.
    Returns True if the file was found and loaded, else False.
    """
    default_path = Path(__file__).resolve().parents[1] / ".env"
    return load_dotenv(dotenv_path or default_path)

def get_key(name: str, default=None, required: bool = False):
    """
    Get an environment variable by name.
    If required=True and the variable is missing/empty, raise KeyError.
    """
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise KeyError(f"Missing required env var: {name}")
    return val
