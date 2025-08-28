# project/src/evaluation.py
from __future__ import annotations
from typing import Callable, Tuple, Dict
import numpy as np
import pandas as pd

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Naive bootstrap on test pairs (i.i.d. assumption). Returns (low, high, samples).
    Note: For time series, this ignores dependence; document this assumption in your notebook.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric(y_true[idx], y_pred[idx]))
    arr = np.array(stats)
    lo = float(np.quantile(arr, alpha/2))
    hi = float(np.quantile(arr, 1 - alpha/2))
    return lo, hi, arr

def parametric_ci_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
) -> Tuple[float, float, np.ndarray]:
    """
    Parametric CI for RMSE assuming residuals ~ N(0, sigma^2).
    Simulates residuals from Normal(0, sigma^2) of same length and builds RMSE distribution.
    """
    rng = np.random.default_rng(seed)
    resid = y_true - y_pred
    n = len(resid)
    sigma = float(np.std(resid, ddof=1)) or 1e-9
    sims = rng.normal(loc=0.0, scale=sigma, size=(n_sim, n))
    rmse_sims = np.sqrt(np.mean(sims**2, axis=1))
    lo = float(np.quantile(rmse_sims, alpha/2))
    hi = float(np.quantile(rmse_sims, 1 - alpha/2))
    return lo, hi, rmse_sims

def subgroup_metric(
    y_true: pd.Series,
    y_pred: pd.Series,
    group: pd.Series,
    metric: Callable[[np.ndarray, np.ndarray], float] = rmse,
) -> pd.DataFrame:
    """
    Compute metric per subgroup (group must align with y_true index).
    Returns a DataFrame: group_value, n, metric.
    """
    out = []
    for g, idx in group.groupby(group).groups.items():
        yt = y_true.loc[idx].to_numpy()
        yp = y_pred.loc[idx].to_numpy()
        out.append({"group": g, "n": len(idx), "metric": metric(yt, yp)})
    return pd.DataFrame(out).sort_values("group").reset_index(drop=True)
