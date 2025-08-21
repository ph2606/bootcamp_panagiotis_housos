# project/src/features.py
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

# ---------- Basics / returns ----------
def add_basic_returns(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """
    Adds: ret (pct_change), log_ret (log return), ret_lag1 (lagged return).
    If adjusted_close not present, falls back to 'close'.
    """
    out = df.copy()
    if price_col not in out.columns:
        price_col = "close"
    out = out.sort_values("date").reset_index(drop=True)
    out["ret"] = out[price_col].pct_change()
    out["log_ret"] = np.log(out[price_col]).diff()
    out["ret_lag1"] = out["ret"].shift(1)
    return out

# ---------- Momentum ----------
def add_momentum_features(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """
    Adds short/medium moving averages and price-vs-MA momentum:
      - ma_5, ma_21
      - mom_5  = close/ma_5 - 1
      - mom_21 = close/ma_21 - 1
      - rsi_14 (simple Wilder-style)
    """
    out = df.copy()
    if price_col not in out.columns:
        price_col = "close"
    out = out.sort_values("date").reset_index(drop=True)
    close = out[price_col]

    out["ma_5"]  = close.rolling(5).mean()
    out["ma_21"] = close.rolling(21).mean()
    out["mom_5"]  = close / out["ma_5"]  - 1.0
    out["mom_21"] = close / out["ma_21"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    return out

# ---------- Volatility ----------
def add_volatility_features(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """
    Adds:
      - vol_21: rolling std of daily returns (21D)
      - range_21: 21D avg of (high-low)/close if columns exist
    Ensures 'ret' exists.
    """
    out = df.copy()
    if price_col not in out.columns:
        price_col = "close"
    out = out.sort_values("date").reset_index(drop=True)
    if "ret" not in out.columns:
        out["ret"] = out[price_col].pct_change()

    out["vol_21"] = out["ret"].rolling(21).std()
    if all(c in out.columns for c in ["high","low","close"]):
        out["range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
        out["range_21"] = out["range"].rolling(21).mean()
    return out

# ---------- Calendar / categorical encodings ----------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar/time features:
      - dow (0=Mon..4=Fri), month (1..12), is_month_end, is_quarter_end
      - one-hot for day-of-week: dow_0..dow_4
    """
    out = df.copy()
    if "date" not in out.columns:
        raise KeyError("Expected a 'date' column (datetime64).")
    dt = pd.to_datetime(out["date"])
    out["dow"] = dt.dt.weekday      # 0..6, equities mostly 0..4
    out["month"] = dt.dt.month
    out["is_month_end"] = dt.dt.is_month_end.astype("Int64")
    out["is_quarter_end"] = dt.dt.is_quarter_end.astype("Int64")
    dummies = pd.get_dummies(out["dow"], prefix="dow", dtype="Int64")
    out = pd.concat([out, dummies], axis=1)
    return out

# ---------- Interactions / targets ----------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple interaction capturing regime effects:
      - ret_x_vol21 = ret * vol_21 (if both present)
    """
    out = df.copy()
    if "ret" in out.columns and "vol_21" in out.columns:
        out["ret_x_vol21"] = out["ret"] * out["vol_21"]
    return out

def add_next_day_targets(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """
    Adds y_next_ret (next-day return) and y_next_up (1 if next-day return > 0).
    """
    out = df.copy()
    if price_col not in out.columns:
        price_col = "close"
    out = out.sort_values("date").reset_index(drop=True)
    ret_today = out[price_col].pct_change()
    out["y_next_ret"] = ret_today.shift(-1)
    out["y_next_up"]  = (out["y_next_ret"] > 0).astype("Int64")
    return out
