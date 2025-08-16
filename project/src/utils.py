from __future__ import annotations
import numpy as np
import pandas as pd

def get_summary_stats(
    df: pd.DataFrame,
    by: str | None = None,
    agg: str | list[str] | dict[str, str] = "mean",
    keep_na_groups: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    
    # Select numeric columns explicitly (version-safe)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Summary over numeric columns (empty DF if no numeric cols)
    summary = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()

    grouped = None
    if by is not None:
        if by not in df.columns:
            raise KeyError(f"Column '{by}' not found in DataFrame.")
        if numeric_cols:
            # Restrict to group key + numeric columns, then aggregate
            to_agg = df[[by] + numeric_cols].copy()
            grouped = (
                to_agg.groupby(by, dropna=not keep_na_groups)[numeric_cols]
                .agg(agg)
                .reset_index()
            )
        else:
            grouped = None  # nothing numeric to aggregate

    return summary, grouped
