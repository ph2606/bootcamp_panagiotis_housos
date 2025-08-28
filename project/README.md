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

## Outliers & Risk Assumptions (Stage 07)

**Definitions used**

- **IQR rule (k=1.5):** flag ret < Q1−1.5·IQR or ret > Q3+1.5·IQR.
- **Z-score (|z|>3):** flag extreme standardized moves.
- **Winsorizing (1%/99%):** clip extremes to reduce leverage without dropping rows.

**Why these choices**

- Robust baseline (IQR) + interpretable threshold (Z) cover skew and scale.
- Winsorizing preserves sample size for modeling while dampening tail risk.

**Sensitivity**

- We compare summary stats and an AR(1) fit across: All / Filtered (no outliers) / Winsorized.
- Outputs saved under: `project/data/processed/outliers/`.

**Risks**

- Crash days are real; over-filtering can hide risk.
- Thresholds are assumptions; we monitor the effect and may adjust with domain evidence.



## Feature Engineering (Stage 09)

**Momentum:** `ma_5`, `ma_21`, `mom_5`, `mom_21`, `rsi_14` — trend & overextension (motivated by EDA).
**Volatility:** `vol_21` (rolling std of returns), `range_21` (avg intraday range).
**Calendar:** `dow` one-hots (`dow_0..dow_4`), `is_month_end`, `is_quarter_end`.
**Interaction:** `ret_x_vol21` (captures regime effects).
**Targets:** `y_next_ret`, `y_next_up` (next-day; for labels only).

Implementation: see `project/src/features.py`. Engineered dataset saved under `project/data/processed/` as `asml_features_<timestamp>.(csv|parquet)`.


## Evaluation & Risk Communication (Stage 11)

**Uncertainty:** RMSE CIs via bootstrap (i.i.d.) and Gaussian (parametric) assumptions.
**Scenarios:** Baseline linear vs polynomial feature (still linear in β).
**Subgroups:** Volatility regimes (low/high) — report per-segment RMSE and residual distributions.
**Outputs:** figures under `project/outputs/eval/`, tables under `project/data/processed/` if saved.
**Notes:** Document where conclusions change under assumptions; prefer bootstrap for fat-tailed residuals.


# ASML — Results & Recommendation (Stage 12)

## Executive Summary

- **Headline:** [One sentence: what’s the decision insight? e.g., “Baseline model provides directionally useful signals; accuracy degrades in high volatility.”]
- **Impact:** [1 bullet on use-case / decision window]
- **Risk:** [1 bullet on where it fails or is sensitive]

---

## Key Visuals & Interpretation

### 1) Next-day Return: True vs Predicted (Test) with ~95% Gaussian Band

![True vs Pred with band](images/fig1_pred_vs_true_with_gaussian_band.png)

**Insight:** [1–2 lines: e.g., “Model tracks small moves; tails widen vs band during spikes.”]
**So what:** [decision: e.g., “Use for light tilts; avoid large bets on volatile days.”]
**Assumption note:** Gaussian band may **underestimate** tail risk.

### 2) Scenario Comparison — RMSE with Bootstrap 95% CIs

![Scenario RMSE](images/fig2_scenario_rmse_bootstrap.png)

**Insight:** [e.g., “Adding polynomial of momentum reduced RMSE by ~X%, but within CI overlap.”]
**So what:** [e.g., “Transformation offers marginal benefit; not statistically decisive.”]

### 3) RMSE by Volatility Regime — Baseline

![RMSE by regime](images/fig3_rmse_by_vol_regime.png)

**Insight:** [e.g., “Error ~Y% higher in high-vol regime.”]
**So what:** [e.g., “De-emphasize predictions during high-vol periods or switch to alt model.”]

---

## Sensitivity Summary (Tables)

**Scenario RMSE (Bootstrap CI)** — see `images/table_scenario_rmse.csv`
**Regime RMSE** — see `images/table_regime_rmse.csv`

Brief: [1–2 lines on direction & magnitude of changes from baseline.]

---

## Assumptions & Risks

- **Data/Target:** next-day return; features use only info available at t (no leakage).
- **Uncertainty:** bootstrap treats residuals i.i.d.; real time-dependence may widen true CIs.
- **Model form:** linear in coefficients; polynomial term adds curvature without changing estimator class.
- **Regime sensitivity:** performance degrades in high volatility.
- **Operational:** transaction costs / slippage not modeled here.

---

## Decision Implications (Now What)

- **Use:** [e.g., “Directional tilt when volatility is below median; small position sizing.”]
- **Monitor:** [e.g., “Residuals and RMSE by regime weekly; flag spikes.”]
- **Next steps:** [e.g., “Rolling validation; Ridge/Lasso; event/calendar features; variance modeling (WLS).”]


## Productization (Stage 13)

### How to Run from a Fresh Clone

1. Create/activate a Python env, then:
   ```bash
   pip install -r project/requirements.txt
   ```


---

## Appendix (Optional)

- Metrics (test): R² = [ ], RMSE = [ ], MAE = [ ]
- Links: methodology notebooks (`/notebooks/stage11_eval_risk.ipynb`, `/notebooks/stage10b_timeseries_or_classification.ipynb`)

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
