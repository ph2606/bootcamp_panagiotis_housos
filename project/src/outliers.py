# project/src/outliers.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> Tuple[pd.Series, Tuple[float, float]]:
    """
    IQR rule: flag values < Q1 - k*IQR or > Q3 + k*IQR.
    Returns (mask, (lower_bound, upper_bound)).
    """
    x = pd.to_numeric(s, errors="coerce")
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lb = q1 - k * iqr
    ub = q3 + k * iqr
    mask = (x < lb) | (x > ub)
    return mask.fillna(False), (lb, ub)

def detect_outliers_zscore(s: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, Tuple[float, float]]:
    """
    Z-score rule: flag values with |(x - mean)/std| > threshold.
    Returns (mask, (mean, std)).
    """
    x = pd.to_numeric(s, errors="coerce")
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=0)) or 1.0
    z = (x - mu) / (sd if sd != 0 else 1.0)
    mask = z.abs() > threshold
    return mask.fillna(False), (mu, sd)

def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> Tuple[pd.Series, Tuple[float, float]]:
    """
    Winsorize by clipping to quantiles [lower, upper].
    Returns (winsorized_series, (q_low, q_high)).
    """
    x = pd.to_numeric(s, errors="coerce")
    ql, qh = np.nanquantile(x, [lower, upper])
    w = x.clip(ql, qh)
    return w, (float(ql), float(qh))
