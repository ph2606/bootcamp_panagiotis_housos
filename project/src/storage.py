# project/src/storage.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd

def env_paths(project_root: Path) -> tuple[Path, Path]:
    """Return RAW and PROCESSED dirs using .env (with safe defaults)."""
    raw_rel = os.getenv("DATA_DIR_RAW", "data/raw")
    proc_rel = os.getenv("DATA_DIR_PROCESSED", "data/processed")
    raw = (project_root / raw_rel).resolve()
    proc = (project_root / proc_rel).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    return raw, proc

def _ensure_parent(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

def write_df(df: pd.DataFrame, path: Path) -> Path:
    """Write by suffix: .csv/.csv.gz via to_csv, .parquet via to_parquet (pyarrow/fastparquet)."""
    path = Path(path)
    _ensure_parent(path)
    suf = path.suffix.lower()
    if suf in (".csv", ".gz") or str(path).lower().endswith(".csv.gz"):
        df.to_csv(path, index=False)
    elif suf in (".parquet", ".parq"):
        try:
            df.to_parquet(path, index=False)  # engine auto-detected (pyarrow/fastparquet)
        except Exception as e:
            raise RuntimeError(
                "Parquet write failed. Install a Parquet engine (e.g., `pip install pyarrow`)."
            ) from e
    else:
        raise ValueError(f"Unsupported file suffix for write: {path.suffix}")
    return path

def read_df(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read by suffix; for CSV you may pass parse_dates=['date'] to preserve dtype."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".csv", ".gz") or str(path).lower().endswith(".csv.gz"):
        return pd.read_csv(path, parse_dates=parse_dates)
    elif suf in (".parquet", ".parq"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                "Parquet read failed. Install a Parquet engine (e.g., `pip install pyarrow`)."
            ) from e
    else:
        raise ValueError(f"Unsupported file suffix for read: {path.suffix}")
