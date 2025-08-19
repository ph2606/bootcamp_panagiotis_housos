# ASML Next-Day Movement Assistant

**Stage:** Problem Framing & Scoping (Stage 01)

# ASML Next-Day Movement Assistant — Tooling Setup

This project is a reproducible scaffold for building a predictive next-day signal for ASML stock. The `project/` folder contains an isolated, organized workspace with `data/`, `notebooks/`, and `src/`, a `.env`-based configuration for secrets/paths, and a sanity-check notebook to verify environment and NumPy. This structure supports subsequent stages (ingestion, preprocessing, modeling, and reporting).

## Problem Statement

ASML stock moves around earnings, guidance, sector flows, and macro prints. The project’s objective is to produce a small, reliable next-day signal for ASML (direction and expected size of the next-day close vs today’s close) that helps decide whether to enter, scale, or wait on a position. This keeps the work decision-linked and avoids a “model first” trap.

**Primary stakeholder/decision owner:** me (student analyst acting as PM). **End user:** me (and a hypothetical mentor). The useful answer is **predictive**: a probability of up/down for the next session and an expected return (bp). The delivered **artifact** will be a notebook-friendly function (`predict_next_day()`), plus a short explainer; the **metric** is out-of-sample directional accuracy and RMSE/MAPE vs naïve and recent-move baselines, at a **daily** decision window.

## Stakeholder & User

- **Who decides?** Student PM (me), reviewing daily before market close or at EOD.
- **Who uses the output?** Me; optionally a mentor reviewing the artifact.
- **Timing & workflow:** Daily refresh; consumed in a research notebook and as a small Python function.

## Useful Answer & Decision

- **Type:** Predictive
- **Decision:** “Enter/scale/wait on ASML for the next session?”
- **Metric:** Directional accuracy and RMSE/MAPE vs naïve/recent-move baselines
- **Artifact:** `project/src/signal.py` (function), demo notebook cell, 1-pager explainer in `project/docs/`

## Assumptions & Constraints

- Public, end-of-day data (prices, volumes, calendar events); educational use; no live trading
- Daily latency acceptable; laptop compute
- Relationships stable enough to beat naïve baselines on rolling OOS windows
- Compliance: no non-public information; this is not investment advice

## Known Unknowns / Risks

- Regime shifts (semis cycle, macro shocks); earnings/event gaps; selection bias
- Leakage around events; will use time-based CV and event masking
- Capacity/costs not modeled at this stage; focus is signal quality



## Cleaning Strategy (Stage 06)

**Goals:** produce a modeling-ready ASML dataset with minimal missingness and standardized numeric scales.

**Functions (in `project/src/cleaning.py`):**

- `fill_missing_median(df, cols=None)`: fills NaNs in numeric columns with median values (robust to outliers).
- `drop_missing(df, cols=None)`: drops rows missing critical columns (when `cols` provided) or any column (when `cols=None`).
- `normalize_data(df, cols=None, method="zscore"|"minmax")`: scales numeric columns; returns normalized DataFrame and parameters.

**Workflow:**

1. Load latest raw CSV from `project/data/raw/`.
2. `drop_missing` on critical fields (`date`, `close`).
3. `fill_missing_median` on remaining numeric columns.
4. `normalize_data` (default z-score).
5. Save cleaned outputs to `project/data/processed/` as timestamped CSV + Parquet.

**Assumptions & Risks:** EOD equity data; educational use; median imputation is appropriate for light missingness; scaling stats should be versioned if used across train/test splits.

## Lifecycle Mapping

Goal → Stage → Deliverable

- Clarify decision & success → **Problem Framing & Scoping (Stage 01)** → Scoping paragraph + persona/memo + `project/` repo skeleton
- Reproducible env → **Tooling Setup** → Conda/venv, `requirements.txt`, `.env`
- Warm-up coding → **Python Fundamentals** → `project/notebooks/hw03_python_fundamentals.ipynb`
- Bring data in → **Data Acquisition/Ingestion** → Ingestion script/notebook; data card
- Keep data → **Data Storage** → Local `project/data/` layout; (optional) parquet cache
- Clean data → **Data Preprocessing** → Cleaning notebook + helpers in `project/src/`
- Identify extremes → **Outlier Analysis** → Outlier report + handling strategy
- Understand structure → **Exploratory Data Analysis** → EDA notebook with visuals
- Create predictors → **Feature Engineering** → Feature notebook; feature list
- Model signal → **Modeling (Time Series/Regression)** → Baselines + model card
- Prove value safely → **Evaluation & Risk Communication** → Metrics vs baselines + risk notes
- Tell the story → **Results Reporting & Stakeholder Communication** → 1-pager + demo notebook
- Make it usable → **Productization #** → `project/src/signal.py` API
- Ship it → **Deployment & Monitoring #** → Batch/cron sketch; basic checks
- Run reliably → **Orchestration & System Design #** → Simple DAG/flow README

## Repo Plan

`project/data/`, `project/src/`, `project/notebooks/`, `project/docs/`; update README each stage; commit at least once per stage with clear messages.
