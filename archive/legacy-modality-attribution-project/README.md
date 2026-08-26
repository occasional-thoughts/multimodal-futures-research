# Archive: prior project — Modality-Attribution Robustness

This is the **first version** of this project, before the revision to the current
**Multimodal Futures Strategy Research & Paper-Trading Platform** (see
[../../PHASE_TRACKER.md](../../PHASE_TRACKER.md) and [../../market_driver_map.md](../../market_driver_map.md)
for the active plan).

Kept here for reference, not deleted, because a few things in it are genuinely useful
prior art for the new plan:

- **`backend/scripts/train_tft.py`** — a real, sanity-checked experiment comparing a
  Temporal Fusion Transformer against XGBoost on single-asset and joint multi-asset
  training. Result: TFT performed at chance level (49.3% single-asset, 51.6% joint
  multi-asset directional accuracy) while XGBoost reached 55.9%–83.8% on the same
  assets. Full writeup in `WRITEUP.md` Section 3.3. Relevant to **Phase 15/16** of the
  current plan when choosing the Model 1 (technical-only) architecture — worth
  re-testing on the new regression target and ZN/CL/GC universe rather than assumed,
  but this is a real, not-to-be-ignored prior data point.
- **`backend/app/models/train.py`** — the XGBoost training wrapper used in that
  experiment; a reasonable starting point for one of Phase 15's baselines.
- **`backend/app/fusion/coattention.py`** — a two-stream cross-attention
  implementation (technical ↔ news) designed to keep modality-attribution separable
  for SHAP. The general co-attention *pattern* (query/key/value across two token
  streams) is relevant inspiration for Phase 19, though it will need extending to
  three modalities (technical/macro/news) and to sequence inputs rather than single-day
  snapshot vectors.
- **`backend/app/explain/shap_utils.py`** — SHAP `TreeExplainer` usage pattern,
  relevant to Phase 35.
- **`backend/app/consistency.py`** — the fusion-consistency / regime-sensitivity
  statistical framework (Spearman correlation, bootstrap CI, pre-registered threshold).
  This was specific to the old research question (is the *explanation* stable across
  fusion architectures) and is **not** the current project's research question (does
  *trading performance* improve with richer information) — not reused directly, but the
  general discipline (pre-register thresholds before looking at results, use
  instance-level stats not aggregate means) still applies.

Everything else here — `WRITEUP.md`, the old `README.md`, `main.py`, `pipeline.py`,
`config.py` (old 12-asset equity+futures universe), the old `frontend/`, and the
trained `.joblib` artifacts — reflects the old single-day-snapshot, two-modality
(technical + news), classification-based design and is superseded by the current plan.
