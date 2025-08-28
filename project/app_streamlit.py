# project/app_streamlit.py
from __future__ import annotations
import streamlit as st
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))
from productize import load_artifacts, vector_from_payload

st.title("ASML Next-Day Return — Demo")

pipe, defaults, feature_order = load_artifacts(project_root)

st.write("Enter features (missing fields will use training medians):")
inputs = {}
for f in feature_order:
    # small set shown; rest expandable
    if f in ("ret","ret_lag1","ret_mean_5","ret_std_21","rsi_14"):
        inputs[f] = st.number_input(f, value=float(defaults.get(f, 0.0)))
    else:
        # hide long tail of fields behind an expander
        pass

with st.expander("Other features", expanded=False):
    for f in feature_order:
        if f not in inputs:
            inputs[f] = st.number_input(f, value=float(defaults.get(f, 0.0)))

if st.button("Predict"):
    x = vector_from_payload(inputs, feature_order, defaults)
    pred = float(pipe.predict(x)[0])
    st.metric("Predicted next-day return", f"{pred:.6f}")
