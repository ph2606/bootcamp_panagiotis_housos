# Stakeholder Memo — ASML Next-Day Movement Assistant (Stage 01)

**Stakeholder (decision owner):** Student PM (me)
**User:** Me (and a mentor reviewing outputs)
**Decision window:** Daily (EOD / pre-open)
**Useful answer:** Predictive; `P(up)` and expected return (bp) for next session

**Problem & why it matters:**
ASML is volatile around events. A small, robust next-day signal can improve entry timing and risk control versus ad-hoc decisions.

**Success metrics:**
Beat naïve and recent-move baselines on held-out windows; track directional accuracy and RMSE/MAPE. Daily latency acceptable.

**Assumptions & constraints:**
Public EOD data only; educational use; laptop compute. Guard against leakage with time-based CV and event masks.

**Risks:**
Regime change; event gaps; overfitting. Monitor rolling performance and retrain cadence.

**Next step:** lock data sources and create a baseline.
