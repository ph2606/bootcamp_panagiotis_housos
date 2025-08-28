# project/src/productize.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from dotenv import load_dotenv
import os
from storage import env_paths, read_df

# --------- Feature selection (must match training & inference) ----------
CANDIDATE_FEATURES = [
    "ret","ret_lag1","ret_lag5","ret_mean_5","ret_std_21","ret_z_21",
    "mom_5","mom_21","rsi_14","vol_21","range_21",
    "is_month_end","is_quarter_end"
]

def _latest_dataset(project_root: Path) -> pd.DataFrame:
    """Load the latest engineered dataset if available, else cleaned, else raise."""
    RAW_DIR, PROC_DIR = env_paths(project_root)
    feat = sorted(list(PROC_DIR.glob("asml_features_*.parquet")) + list(PROC_DIR.glob("asml_features_*.csv")))
    clean = sorted(list(PROC_DIR.glob("asml_cleaned_*.parquet")) + list(PROC_DIR.glob("asml_cleaned_*.csv")))
    if feat:
        df = read_df(feat[-1], parse_dates=["date"])
    elif clean:
        df = read_df(clean[-1], parse_dates=["date"])
    else:
        raise FileNotFoundError("No processed dataset found. Run earlier stages first.")
    df = df.sort_values("date").reset_index(drop=True)
    price_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    if "y_next_ret" not in df.columns:
        df["y_next_ret"] = df[price_col].pct_change().shift(-1)
    return df

def build_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    X = df[features].copy()
    y = df["y_next_ret"].copy()
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

def train_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe.fit(X, y)
    return pipe

def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def compute_defaults(X: pd.DataFrame) -> Dict[str, float]:
    # Use training medians as robust defaults for missing fields at inference
    med = X.median(numeric_only=True).to_dict()
    # Ensure every candidate feature has *some* default (0.0 fallback)
    for c in CANDIDATE_FEATURES:
        if c not in med:
            med[c] = 0.0
        elif med[c] is None or (isinstance(med[c], float) and np.isnan(med[c])):
            med[c] = 0.0
    return med

def save_artifacts(project_root: Path, pipe: Pipeline, feature_order: list[str], defaults: Dict[str, float]) -> Path:
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "feature_meta.json"
    joblib.dump(pipe, model_path)
    with open(meta_path, "w") as f:
        json.dump({"features": feature_order, "defaults": defaults}, f, indent=2)
    return model_path

def load_artifacts(project_root: Path) -> Tuple[Pipeline, Dict[str, float], list[str]]:
    model_dir = project_root / "model"
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "feature_meta.json"
    pipe = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text())
    return pipe, meta["defaults"], meta["features"]

def vector_from_payload(payload: Dict, feature_order: list[str], defaults: Dict[str, float]) -> np.ndarray:
    vals = []
    for f in feature_order:
        v = payload.get(f, defaults.get(f, 0.0))
        vals.append(float(v))
    return np.array(vals, dtype=float).reshape(1, -1)

def run_full_training_and_save(project_root: Path) -> Dict:
    load_dotenv(project_root / ".env")
    df = _latest_dataset(project_root)
    X, y = build_Xy(df)

    # time-aware split (80/20) for a metrics snapshot
    split = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split, :], X.iloc[split:, :]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    pipe = train_pipeline(X_tr, y_tr)
    yhat_tr = pipe.predict(X_tr)
    yhat_te = pipe.predict(X_te)

    m_tr = metrics_dict(y_tr.to_numpy(), yhat_tr)
    m_te = metrics_dict(y_te.to_numpy(), yhat_te)
    defaults = compute_defaults(X_tr)
    save_artifacts(project_root, pipe, X_tr.columns.tolist(), defaults)
    return {"train": m_tr, "test": m_te, "n_train": len(X_tr), "n_test": len(X_te)}
