# project/src/cleaning.py
from __future__ import annotations
from typing import Iterable, Literal, Tuple, Dict
import numpy as np
import pandas as pd

Numeric = np.number

def fill_missing_median(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fill missing values in numeric columns with the column median.
    Returns (result_df, medians_dict).
    If cols is None, all numeric columns are used.
    """
    work = df if inplace else df.copy()
    if cols is None:
        cols = work.select_dtypes(include=[Numeric]).columns.tolist()

    medians: Dict[str, float] = {}
    for c in cols:
        if c in work.columns:
            med = work[c].median(skipna=True)
            medians[c] = float(med) if pd.notna(med) else np.nan
            work[c] = work[c].fillna(med)
    return work, medians

def drop_missing(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    how: Literal["any","all"] = "any",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Drop rows with missing values.
    - If cols is None: drop rows with ANY missing values across all columns (how='any').
    - If cols is provided: drop rows where selected column(s) have missing values (how ignored; behaves like 'any').
    """
    work = df if inplace else df.copy()
    if cols is None:
        return work.dropna(how=how)
    else:
        return work.dropna(subset=list(cols))

def normalize_data(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    method: Literal["zscore","minmax"] = "zscore",
    stats: Dict[str, Dict[str, float]] | None = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize numeric columns.
    - method='zscore': x' = (x - mean) / std
    - method='minmax': x' = (x - min) / (max - min)
    - If stats is provided, use those parameters (no refit). Otherwise, fit on df and return stats.
    Returns (result_df, stats_dict).
    """
    work = df if inplace else df.copy()
    if cols is None:
        cols = work.select_dtypes(include=[Numeric]).columns.tolist()

    params: Dict[str, Dict[str, float]] = {} if stats is None else stats

    for c in cols:
        if c not in work.columns:
            continue
        s = pd.to_numeric(work[c], errors="coerce")

        if stats is None:
            if method == "zscore":
                mu = float(s.mean(skipna=True))
                sd = float(s.std(skipna=True))
                # avoid divide-by-zero
                sd = sd if sd and sd != 0 else 1.0
                params[c] = {"mean": mu, "std": sd}
            elif method == "minmax":
                mn = float(s.min(skipna=True))
                mx = float(s.max(skipna=True))
                rng = (mx - mn) if (mx - mn) != 0 else 1.0
                params[c] = {"min": mn, "max": mx, "range": rng}

        p = params.get(c, {})
        if method == "zscore":
            mu, sd = p.get("mean", 0.0), p.get("std", 1.0)
            work[c] = (s - mu) / (sd if sd != 0 else 1.0)
        elif method == "minmax":
            mn, rng = p.get("min", 0.0), p.get("range", 1.0)
            work[c] = (s - mn) / (rng if rng != 0 else 1.0)

    return work, params
