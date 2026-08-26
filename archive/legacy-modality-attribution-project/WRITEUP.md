# Explainable Multimodal Stock Movement Prediction: Modality-Attribution Robustness Across Fusion Strategies and Market Regimes

Project Technical Write-Up (Revised)

## 1. Base Research Paper

"Stock Price Prediction Using FinBERT-Enhanced Sentiment with SHAP Explainability and Differential Privacy" (MDPI Mathematics, Aug 2025)

This paper proposes a multimodal framework that extracts financial sentiment from news headlines using FinBERT (a domain-adapted transformer), fuses it with technical/statistical price indicators, and feeds the combined feature vector into an XGBoost classifier to forecast short-term stock movement. SHAP is used post-hoc to decompose each prediction into additive feature attributions. A secondary contribution adds differential privacy to the pipeline.

A closely related paper, IKNet (2025), extends this by computing SHAP values at the individual news-keyword level rather than only at the aggregate sentiment-score level, showing that specific keywords can outweigh technical indicators in driving predictions.

## 2. Identified Gap / Novelty

Both papers compute SHAP attributions for one fixed fusion architecture, evaluated at one point in time. Neither paper asks:

- **Does modality attribution depend on how the fusion is designed?** If technical and news features are combined differently (simple concatenation vs. weighted fusion vs. attention-based fusion), does SHAP still credit the same modality for the same prediction — or is the "news vs. technical" attribution itself an artifact of the fusion architecture choice, not a stable property of the data?
- **Does modality attribution hold up across market conditions?** A model might lean on news sentiment during calm periods but should arguably lean more on technical momentum during high-volatility regimes (or vice versa) — neither base paper tests whether SHAP attribution shifts accordingly, or whether it stays static regardless of regime (which would suggest the explanation is not actually tracking real market behavior).
- **SHAP alone gives attribution, not actionability.** Neither paper provides counterfactual explanations (e.g., "what minimal sentiment shift would flip today's prediction from Down to Up") — a complementary explanation style shown in other domains (Mothilal et al., 2020) to be more directly interpretable for decision-makers.

**Proposed novelty:** a Modality-Attribution Robustness Framework that (a) trains the same multimodal predictor under three fusion strategies, (b) segments evaluation data by volatility regime, (c) measures whether SHAP's news-vs-technical attribution is consistent across both axes, and (d) adds counterfactual explanations (via DiCE) to complement SHAP's attribution view with minimal-change, actionable explanations. This directly tests whether existing multimodal-XAI stock models' explanations are trustworthy and stable, or fusion-architecture artifacts — a gap neither base paper addresses.

## 3. Proposed Methodology (Revised)

The original plan computed a consistency score across fusion variants and regimes but had five gaps that would have prevented it from actually answering its own question. Each subsection below states the original design, the problem, and the fix.

### 3.1 Data collection

OHLCV price history (Yahoo Finance via `yfinance`) + date-aligned financial news headlines (Kaggle financial news corpus), for 3–5 stocks across different sectors (e.g., one tech, one financial, one energy, one consumer-staples, one healthcare — chosen so volatility regimes aren't confounded with sector).

**Fix — temporal integrity:** All splits must be strictly chronological. Train on an earlier window, validate on the window immediately after, test on the latest window — never a random shuffle-split, since technical indicators and rolling volatility are autocorrelated across adjacent days and a random split leaks future information into training.

### 3.2 Technical feature engineering

RSI, MACD, moving averages, Bollinger Bands, realized volatility — all computed with trailing (backward-looking) windows only, never centered windows, to avoid look-ahead bias.

### 3.3 Numeric backbone: architecture selection (XGBoost vs. Temporal Fusion Transformer)

The technical branch's classifier was not assumed by default — it was chosen by controlled comparison, evaluated on real price data for all 10 assets in the study.

**Candidate evaluated:** the Temporal Fusion Transformer (TFT), an attention-based sequence model reported in the financial-forecasting literature to outperform LSTM/GRU baselines and to offer built-in interpretability (variable-selection weights, temporal attention), making it a plausible deep-learning alternative to gradient boosting for the numeric branch.

**Experiment 1 — single-asset training:** TFT trained independently per ticker (~449 usable trading days, ~22k model parameters) reached **49.3% directional accuracy on AAPL** — at chance level. Consistent with known tabular-data findings that deep sequence models are data-hungry relative to the sample sizes a single equity's daily history provides.

**Experiment 2 — joint multi-asset training:** to rule out data volume as the sole cause, TFT was retrained on all 10 assets pooled into one multi-series dataset (3,156 training rows, ~10× the single-asset data, each ticker sharing model weights as its own group). Result: **51.6% overall directional accuracy** across 670 held-out predictions — still at chance level, with per-ticker accuracy (41.8%–64.2%) falling entirely within the statistical noise band expected around 50% at this sample size (no ticker showed a reliable, reproducible edge).

**Comparison to XGBoost** on the identical assets and split (Section 3.4, single fixed fusion baseline): XGBoost reached 55.9%–83.8% accuracy across the 10 assets, decisively outperforming both TFT variants on 7 of 10 assets. All numbers were independently sanity-checked against ground truth (prediction-to-outcome alignment verified at each step) before being treated as valid.

**Decision:** XGBoost is retained as the numeric-branch classifier. This is reported as a deliberate, evidence-based architecture choice rather than a default — a documented negative result for TFT on this specific data regime (small per-asset sample sizes typical of individual-equity daily data), not a claim that TFT or attention-based sequence models are wrong for stock prediction in general.

### 3.4 News sentiment extraction

FinBERT applied per headline; daily aggregation (mean, or weighted by recency) into a sentiment feature vector. Retain the FinBERT class probabilities (positive/negative/neutral) as a small vector per day, not just a single scalar score — this gives the fusion layer more than one "news" dimension to work with and makes modality-level SHAP aggregation over multiple news columns meaningful rather than trivial (summing `|SHAP|` over a single scalar is just that scalar's own SHAP value).

### 3.4 Fusion strategy variants — redesigned for attribution separability

**Original problem:** the plan specified three fusion variants (concat, weighted, attention) and proposed computing modality-level SHAP by summing `|SHAP|` over "technical" columns vs. "news" columns in each variant's *output* representation. This works for concatenation, but a standard cross-attention layer blends technical and news information into a single fused embedding — once that happens, individual embedding dimensions are no longer identifiably "technical" or "news," and the cross-variant attribution comparison becomes apples-to-oranges. If left uncorrected, the central experiment of the project cannot actually run as intended.

**Fix:** every fusion variant must produce a feature vector whose columns remain traceable to a source modality, so that `Σ|SHAP|` can always be split into a technical group and a news group on the same basis across variants.

- **(a) Concatenation fusion** — `[tech_block | news_block]`. Baseline; trivially separable.
- **(b) Weighted fusion** — `[w_tech · tech_block | w_news · news_block]`, where `w_tech, w_news` are scalar weights learned by a small logistic model trained jointly on the prediction labels (not applied element-wise across mismatched dimensions, and not summed into a single vector — each block keeps its own columns, just rescaled). This preserves both separability and interpretability of the weights themselves as a secondary (coarser) attribution signal to sanity-check SHAP against.
- **(c) Attention-based fusion** — implemented as **two-stream co-attention**, not a single merged embedding: the technical stream attends to the news stream and vice versa, but each stream's *attended output* is kept in its own block. The fused vector is `[attended_tech_block | attended_news_block]`, concatenated before the classifier. This lets attention reweight each modality's features using the other modality as context, without collapsing the two into an inseparable blend.

Classifier: XGBoost, trained independently per fusion variant on next-day up/down labels.

### 3.5 SHAP explanation

Compute SHAP values per prediction, per fusion variant, using `TreeExplainer` (XGBoost is tree-based, so exact SHAP is tractable — no need for the slower model-agnostic kernel explainer). Aggregate into modality-level attribution: `tech_attr = Σ|SHAP|` over technical-block columns, `news_attr = Σ|SHAP|` over news-block columns. Report the **share** `news_attr / (news_attr + tech_attr)` as the primary comparable quantity, since raw magnitudes aren't comparable across models with different output scales.

### 3.6 Volatility regime segmentation

Split the test period into calm vs. volatile windows using realized volatility (rolling 20-day std. dev. of returns), thresholded at the sample median (or top/bottom tercile if the middle is to be discarded as ambiguous). Computed using only trailing data.

**Fix — leakage guard:** insert a short embargo gap (a few trading days) between the regime-labeling window and the train/test boundary, and between calm/volatile windows themselves where they abut, so a rolling-volatility calculation never reads across the boundary it's used to define.

### 3.7 Attribution consistency scoring — redefined to be statistically meaningful

**Original problem:** "rank correlation / cosine similarity of modality attribution" was proposed between the mean attribution profile of each (fusion × regime) cell — but each cell reduces to a single 2-dimensional point (tech share, news share). Correlating two aggregate scalars across a handful of cells has no statistical power and no natural significance test.

**Fix:** compute consistency at the **instance level**, using the per-test-instance news-attribution share, not the cell mean:

- **Fusion-consistency** (does architecture change the story?): for each regime, Spearman-correlate the per-instance news-share vector between each pair of fusion variants, matched on the same test instances. High correlation ⇒ architecture-independent attribution; low or negative ⇒ architecture is driving the explanation.
- **Regime-sensitivity** (does the story change with market conditions, as it plausibly should?): for each fusion variant, compare the *distribution* of news-share between calm and volatile regimes (Mann–Whitney U test, plus the effect size). A real difference here is not a failure — it would be evidence the explanation tracks actual market behavior; no difference across regimes despite very different market conditions is itself a finding worth reporting as (weak) evidence the explanation is static/uninformative.
- Report bootstrap confidence intervals on all shares and correlations (resampling test instances), since 3–5 stocks × 3 fusion variants × 2 regimes gives thin per-cell sample sizes — point estimates alone would overstate certainty.

**Fix — pre-registered decision rule:** state the interpretation threshold *before* running the experiment, not after seeing the numbers. Adopt: attribution is **architecture-robust** within a regime if the fusion-consistency Spearman correlation is ≥ 0.7 across all three fusion-variant pairs; if any pair falls below that (or a pair's correlation sign flips against the others), treat the "news vs. technical" story as an artifact of fusion-architecture choice for that regime. This turns the consistency score from a descriptive number into an answer to the paper's actual question.

### 3.8 Counterfactual generation

For representative predictions (e.g., cases near the decision boundary in each fusion × regime cell), use DiCE to generate the minimal feature change that would flip the prediction.

**Fix:** XGBoost is not differentiable, so DiCE must run in **genetic-algorithm or random-search mode** (`dice_ml.Dice(..., method="genetic")`), not the gradient-based mode most tutorials default to. Constrain the search to plausible feature ranges (FinBERT sentiment aggregate bounded to its natural range, RSI to [0, 100], etc.) via DiCE's feature range/permitted-range config, so the generated counterfactual is an *actionable* change ("sentiment would need to shift from -0.2 to +0.15") rather than a value that could never occur in real data.

### 3.9 Evaluation

Report, per fusion variant: accuracy, and the modality attribution share by regime. Report the fusion-consistency and regime-sensitivity statistics from 3.7 against the pre-registered threshold, with bootstrap CIs. Report qualitative counterfactual case studies per fusion variant. Test across all 3–5 stocks and report whether the robust/artifact conclusion is consistent across stocks (a conclusion that only holds for one stock is a much weaker claim than one that generalizes).

## 4. Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| DL framework | PyTorch, HuggingFace Transformers |
| Sentiment model | FinBERT (pretrained) |
| Fusion layer | Custom PyTorch module (concat / weighted / two-stream co-attention variants) |
| Classifier | XGBoost |
| Explainability (attribution) | SHAP (`TreeExplainer`) |
| Explainability (counterfactual) | DiCE (genetic-algorithm mode) |
| Data sources | Yahoo Finance (`yfinance`), Kaggle financial news datasets |
| Volatility regime labeling | Realized volatility / rolling std. dev. of returns, with embargo gaps |
| Stats | SciPy (Spearman, Mann–Whitney U), bootstrap CIs |
| Experiment tracking | Weights & Biases or MLflow |
| Serving / demo | FastAPI backend + web frontend (live interactive demo) |

## 5. Reference Papers

1. Stock Price Prediction Using FinBERT-Enhanced Sentiment with SHAP Explainability and Differential Privacy, MDPI Mathematics, 2025. (base paper)
2. IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators, arXiv, 2025.
3. Gu, W., Li, S., Wang, Z., Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis, ICCBDC 2024.
4. Multimodal Stock Price Prediction (tweets + news + financial metrics), ResearchGate, 2025.
5. Araci, D., FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, arXiv, 2019.
6. Mothilal, R. K., Sharma, A., Tan, C., Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations (DiCE), FAT* 2020.
7. Lundberg, S. & Lee, S., A Unified Approach to Interpreting Model Predictions (SHAP), NeurIPS 2017.

## 6. Flowchart

```
 Historical OHLCV data          Date-aligned financial news headlines
        |                                    |
        v                                    v
 Technical indicators                FinBERT sentiment extraction
 (RSI, MACD, MA, volatility,         (per headline -> daily aggregate,
  trailing windows only)              class-probability vector, not scalar)
        |                                    |
        +-------------------+----------------+
                             |
                             v
              +-------------------------------------+
              |   Fusion strategy (x3 runs)          |
              |  (a) Concatenation                   |
              |  (b) Weighted (per-block scalars)    |
              |  (c) Two-stream co-attention          |
              |      (blocks stay separable)         |
              +-------------------------------------+
                             |
                             v
                  Train XGBoost classifier
                  (per fusion variant, chronological split)
                             |
                             v
                  Predict next-day up/down
                             |
                             v
       SHAP attribution (TreeExplainer) -> tech vs news share
                             |
                             v
        Segment test set by volatility regime
        (calm window / volatile window, embargo gaps)
                             |
                             v
     Instance-level fusion-consistency (Spearman, per pair)
     + regime-sensitivity (Mann-Whitney U) + bootstrap CIs
                             |
                             v
     Apply pre-registered threshold -> robust vs artifact verdict
                             |
                             v
        Generate counterfactual explanations (DiCE, genetic mode,
        range-constrained) for representative predictions
                             |
                             v
   Output: Accuracy per fusion variant + Attribution Consistency
   Report (with verdict) + Counterfactual case studies
```

## 7. Algorithm (Revised)

```
Input: price data P, news headlines N, FinBERT model F,
       fusion strategies Φ = {concat, weighted, coattention},
       classifier C, SHAP explainer E, counterfactual generator CF,
       consistency threshold τ = 0.7

1.  tech_features ← compute_indicators(P)              # trailing windows only
2.  for each day d:
3.      sentiment_d ← aggregate_class_probs([F(h) for h in N[d]])   # vector, not scalar
4.
5.  split ← chronological_split(P)                      # train / val / test, no shuffling
6.
7.  for each strategy φ in Φ:
8.      fused_φ ← φ(tech_features, sentiment)            # each fused_φ keeps tech/news
                                                          # blocks traceable to source
9.      C_φ ← train(C, fused_φ[split.train], labels[split.train])
10.     for each test instance x in split.test:
11.         pred_φ[x] ← C_φ.predict(x)
12.         shap_φ[x] ← E(C_φ, x)                        # TreeExplainer
13.         tech_attr_φ[x] ← Σ|shap_φ[x]| over tech-block columns
14.         news_attr_φ[x] ← Σ|shap_φ[x]| over news-block columns
15.         news_share_φ[x] ← news_attr_φ[x] / (news_attr_φ[x] + tech_attr_φ[x])
16.
17. regimes ← segment_by_volatility(P, embargo=k_days)   # calm / volatile
18.
19. # Fusion-consistency: instance-level, per regime
20. for each regime r:
21.     for each pair (φ_i, φ_j) in Φ:
22.         consistency[r][φ_i, φ_j] ← spearman(news_share_φi[x in r], news_share_φj[x in r])
23.
24. # Regime-sensitivity: per fusion variant
25. for each strategy φ in Φ:
26.     regime_sensitivity[φ] ← mann_whitney_u(news_share_φ[calm], news_share_φ[volatile])
27.
28. CIs ← bootstrap(news_share, consistency, regime_sensitivity, n_resamples=2000)
29.
30. verdict ← "architecture-robust" if min(consistency[r][:,:]) ≥ τ for all r
             else "architecture-artifact"
31.
32. for representative test instances x* (near each fusion x regime decision boundary):
33.     cf[x*] ← CF.generate(C_best, x*, method="genetic", permitted_ranges=feature_bounds)
34.
35. return {pred_φ}, {accuracy_φ}, {news_share_φ}, consistency, regime_sensitivity,
           CIs, verdict, cf
```

## 8. Notes for Scoping (4-month timeline)

- Fusion strategy (b)/(c) and the counterfactual layer are the most implementation-heavy pieces — prioritize getting strategy (a) fully working end-to-end first, then extend.
- Volatility-regime segmentation is cheap to compute (rolling std. dev.) — do this early so evaluation code is ready before all three fusion variants finish training.
- Testing on 3–5 stocks across different sectors (not just one) is what elevates this from a single-experiment result to a generalizable finding.
- Decide the consistency threshold (τ = 0.7 above) and the regime-sensitivity test *before* looking at results, and keep that decision recorded — the credibility of the "robust vs. artifact" conclusion depends on not moving the goalposts after seeing the numbers.
- The Kaggle news corpus is historical only; a live demo (Section 9) additionally needs a live/recent headline source, which is a separate concern from the historical corpus used for training.

## 9. Live Interactive Demo (App)

A FastAPI backend + web frontend that lets a user pick a trained stock and a test-period date, and see: the prediction from each fusion variant, the SHAP tech/news attribution split for that instance, which regime that date falls in, and a DiCE-generated counterfactual ("headline sentiment would need to shift from X to Y to flip this prediction"). This is a presentation layer over the pipeline in Sections 3–7, not a replacement for the offline experiment — model training and the consistency report are computed offline; the app serves precomputed model artifacts plus on-demand SHAP/DiCE for the instance the user selects.
