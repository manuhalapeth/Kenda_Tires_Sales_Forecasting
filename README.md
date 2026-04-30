# Kenda Tires & Wheels: ATW Sales Forecasting

This repository contains the full data science workflow built for **Kenda Tires and Wheels** to forecast ATW (All-Terrain & Winter) product revenue. The project evolved from a single-series aggregate demand forecast to a production-ready, item-level forecasting system covering 1,368 SKUs across 50 months of transaction history. The final system forecasts monthly revenue per item three months out, includes a confidence grading layer, and is reconciled to a top-down aggregate benchmark.

---

## File Structure

```
Kenda/
├── SalesData_ATW.xlsx                 # Raw sales data (196,426 rows, 2022–2026)
│
├── analyze_sales.ipynb                # Raw EDA notebook (no outputs)
├── analyze_sales_output.ipynb         # Initial EDA + aggregate forecasting (with outputs)
├── improved_forecasting.ipynb         # Leak-free aggregate forecasting
├── item_forecasting.ipynb             # First item-level forecast 
├── item_forecasting_improved.ipynb    # Final item-level forecast (production)
│
├── item_forecasts_production.csv      # Output from item_forecasting.ipynb
├── item_forecasts_improved.csv        # Output from item_forecasting_improved.ipynb
│
├── make_improved_notebook.py          # Script used to generate improved_forecasting.ipynb
├── Kenda_Tires_Blog                   # Blog writeup on the project
│
├── Data_Viz_Kenda.png                 # EDA visualization
├── Data_Viz_Updated.png               # Updated data visualization
├── DV_P_2.png                         # Additional data visualization
├── forecast_dashboard.png             # Forecast comparison dashboard
└── forecast_final.png                 # Final forecast chart (dark-mode publication quality)
```

---

## Notebook Walkthrough

### 1. `analyze_sales_output.ipynb` — Initial EDA & Aggregate Forecasting

This was the starting point. The notebook performs a comprehensive EDA on the raw sales data (196,426 rows, 10 columns) and then builds ten forecasting models against monthly aggregated revenue: Naive (Last Value), Seasonal Naive, Simple Moving Average, Weighted Moving Average, Simple Exponential Smoothing, Holt-Winters (additive and multiplicative), ARIMA, SARIMA, Linear Regression, Random Forest, XGBoost, and Prophet with external regressors.

**Why it was superseded: Target Leakage**

The ML models (Linear Regression, Random Forest, XGBoost, Prophet) were fed features derived from the same month being predicted. Specifically:

- **COGS (Cost of Goods Sold)**: This figure is only settled after the transactions of month T have occurred. It is not available before the month begins — it is a byproduct of the sale, not a predictor of it.
- **QtyShipped and NumTransactions**: These are counts of events that happened *during* month T, not before it. Including them as input features means the model is effectively told how busy the month is before being asked to predict how busy the month will be.
- **NumCustomers and NumItems**: Similarly, knowing how many unique customers or SKUs were active in month T requires seeing all of month T's data.

Because these features encode the answer, the models reported inflated accuracy during training and cross-validation. A model using COGS to predict Sales is, for practical purposes, predicting a lagged accounting identity rather than learning a real forward-looking signal. In deployment the model would have nothing to plug into those inputs, so the reported metrics were meaningless.

---

### 2. `improved_forecasting.ipynb` — Leak-Free Aggregate Forecasting

This notebook was rebuilt from scratch to fix the data leakage. The core rule enforced throughout: **when forecasting month T, only features from months ≤ T-1 are permitted as inputs**.

Key accomplishments:

- **Clean feature engineering**: Lag features (lag_1 through lag_12), rolling mean and standard deviation (shift-first before rolling to avoid peeking), Fourier terms encoding annual and semi-annual seasonality, and a linear trend index.
- **Walk-forward cross-validation**: Instead of a single 80/20 split, an expanding-window walk-forward CV simulates real deployment. The model is trained on everything up to time T and predicts T+1, repeated across all available folds.
- **Broader model comparison**: Added LightGBM and a stacking ensemble (RF + XGBoost + LightGBM) in addition to all prior models.
- **Best result**: The Optimised Ensemble (RF + XGBoost + LightGBM) achieved **MAPE 8.58%** on the 10-month held-out test set (May 2025 – Feb 2026), compared to a naïve last-value baseline at 40.67%.
- **Honest stacking**: CV-tuned ensemble weights are derived from walk-forward out-of-sample predictions, not from the test set, no leakage in the ensemble layer either.
- **Publication-quality chart** (`forecast_final.png`) showing the full model comparison with confidence bands.

---

### 3. `item_forecasting.ipynb` — First Item-Level Forecast (v4, Superseded)

When the client moved from wanting a single aggregate revenue number to wanting an item to item monthly forecast, this notebook was built. It extended the aggregate approach to 1,368 individual items using a global XGBoost model with Optuna Bayesian hyperparameter tuning, EWM features, and a Forecast Confidence Score system.

**Issues that led to the improved version**

- **Seasonal average for intermittent items**: The 664 items with sparse sales history (< 60% of months active) were forecast by taking the mean revenue for the same calendar month from prior years. This is simple and interpretable but ignores trend, cross-item signals, and recent momentum. For fast-growing or newly-active items, the seasonal mean anchors to old, irrelevant history.
- **No cross-series features**: Each item was forecast in isolation. Product-line-level or total-market-level signals (e.g., the entire 0003 product line accelerating) were invisible to the model.
- **Proportional reconciliation**: After generating item-level forecasts, the notebook scaled all items proportionally so they sum to the Holt-Winters aggregate target. This is statistically suboptimal because it treats all items as equally uncertain — a high-revenue item with a confident forecast gets the same adjustment as a noisy low-revenue item.
- **Only 28 features**: The feature set, while effective for the aggregate case, did not capture quantity dynamics, implicit price signals, customer breadth trends, or warehouse concentration at the item level.
- **Revenue-weighted MAPE plateau**: Regular items achieved ~79% revenue-weighted MAPE; the overall number was ~123%. The gap between regular and intermittent performance was large and unaddressed.

---

### 4. `item_forecasting_improved.ipynb` — Final Item-Level Forecast (Production)

This is the final deliverable. It addresses every limitation of v4 with production-grade techniques drawn from the M5 Forecasting Competition methodology.

**What it accomplishes**

| Component | Detail |
|---|---|
| Regular items (704) | LightGBM with Optuna Bayesian tuning (50 trials), 70 M5-style features |
| Intermittent items (664) | CrostonSBA + AutoETS ensemble via `statsforecast` |
| Top-100 items | Optional blending with Amazon Chronos foundation model (70/30 LGB/Chronos) |
| Feature count | 70 features: multi-resolution rolling (2–24 months), 5 EWM alphas, quantity lags, implicit price, customer breadth, warehouse breadth, product-line cross-series lags, total-market cross-series lags |
| Reconciliation | MinT (minimum trace, WLS diagonal) — statistically optimal hierarchical reconciliation |
| Forecast horizon | 3 months forward: Mar / Apr / May 2026 |
| Confidence grading | Score 0–100, graded A/B/C/D per item based on walk-forward CV MAPE |
| Output | `item_forecasts_improved.csv` with per-item forecasts, confidence grade, tier, method |

**Key results (test set: May 2025 – Feb 2026)**

| Metric | Improved model | v4 baseline | Improvement |
|---|---|---|---|
| Revenue-weighted MAPE (all) | **47.56%** | 123.69% | +76 pp |
| Revenue-weighted MAPE (regular) | **42.76%** | 78.86% | +36 pp |
| Revenue-weighted MAPE (intermittent) | **61.86%** | n/a (no dedicated model) | — |
| Portfolio aggregate MAPE (monthly) | **10.6%** | ~19%+ | vs HW alone at 19.3% |
| Median absolute error per item-month | $957 | $1,304 (seasonal naive) | **35.3% lower** |

**Why it's a success**

The model was statistically validated. A Wilcoxon signed-rank test (n=7,898 item-months, one-sided H1: improved errors < naive errors) returned **p = 1.90 × 10⁻¹¹⁸**, providing overwhelming statistical evidence that the model's per-item absolute errors are significantly smaller than a seasonal naive baseline. 

The portfolio-level aggregate MAPE of **10.6%** means that when you sum up all item-level forecasts for a given month, the total is within ~10% of actual total revenue. This is comparable to a dedicated aggregate time-series model (Holt-Winters at 19.3%) while also providing granular per-item breakdowns. This is something no aggregate model can offer.

The confidence grading system is actionable. Items with grade A (confidence score ≥ 75) had a median test MAPE of ~46% in validation; items with grade D had ~70%+ median MAPE. This stratification lets inventory planners apply tight buffers to A-grade items and manual review to D-grade items, optimising working capital allocation.

---

## Limitations and Mitigations

### 1. High unweighted MAPE on intermittent items (~185%)

**Why it happens**: The 664 intermittent items sell sporadically, months with zero revenue are common. When the actual revenue is $50 and the forecast is $300, MAPE is 500%. A handful of such extreme errors dominate the unweighted average, even if the revenue impact is negligible. This is a structural property of intermittent demand, not a model failure.

**Mitigation**: Use revenue-weighted MAPE as the primary business metric (47.56% overall). The unweighted number is misleading because a $50/month SKU and a $500,000/month SKU receive equal weight. For operational decisions, sort by `Total_Q2_2026` descending, items with genuine revenue at stake are almost all grade A or B. Long-term: collect more data for intermittent items, or reclassify very-low-volume SKUs as non-forecastable and handle via safety stock only.

### 2. Short training history (50 months, 4 years)

**Why it happens**: The sales data begins January 2022. Many ML models and time-series techniques benefit from 5–10+ years of history to reliably learn multi-year seasonality, business cycles, and product lifecycle curves. With 50 months the model sees at most four complete seasonal cycles.

**Mitigation**: As each month passes, retrain the production models on the expanding dataset. The lag_24 and roll_24 features already prepare for this — they will start delivering signal once 24 months of item-level history exist. Adding external signals (industry tyre sales indices, weather patterns, raw rubber prices) can partially substitute for historical depth.

### 3. Recursive forecasting error accumulation

**Why it happens**: Mar 2026 is predicted using actual history. Apr 2026 is predicted using actual history + the Mar 2026 prediction. May 2026 uses history + Mar + Apr predictions. Each step introduces a small error that compounds forward. By May, the features `lag_1` and `lag_2` are already predictions, not observations.

**Mitigation**: For a 3-month horizon the accumulation is modest. The MinT reconciliation provides a partial correction by anchoring the item-level sum to the independently-forecast Holt-Winters total. Going forward, refreshing forecasts monthly (rolling 1-step-ahead) instead of committing to a fixed 3-month batch will keep compounding minimal.

### 4. Cold-start problem for new SKUs

**Why it happens**: Items that enter the catalogue after the training window have no lag features, no rolling means, and no EWM history, all of which are the most important features in the model. The model cannot be directly applied to them.

**Mitigation**: Use product-line average revenue as a proxy for the first 3–6 months of a new SKU's life, then transition to the ensemble model once sufficient history accumulates. The confidence scoring system will automatically assign new items a D grade until their walk-forward CV MAPE stabilises.

### 5. No exogenous macroeconomic features

**Why it happens**: The current model is relies entirely on internal factor: it only looks at Kenda's own historical sales. External shocks (raw material price spikes, tariff changes, competitor launches, fuel price trends affecting ATV/off-road activity) are invisible to it.

**Mitigation**: Add a small set of leading indicators as additional features: US rubber commodity price index (monthly lag), consumer confidence index, and relevant weather indices for off-road season timing. These are publicly available and straightforward to pipe in via an automated data pull.

---

## Scalability: n8n Workflow Integration

The forecasting system is currently a series of Jupyter notebooks executed manually. To run this in production on a recurring basis, updating forecasts every month as new sales data arrives, an **n8n** automation workflow is the recommended path forward. n8n is an open-source workflow automation tool that can orchestrate data pipelines, run scripts, and push outputs to business tools without requiring custom infrastructure.

**Proposed monthly pipeline**

```
[Schedule Trigger]
       │  First Monday of each month
       ▼
[HTTP Request Node]
       │  Pull latest SalesData_ATW.xlsx from SharePoint / Google Drive / ERP export
       ▼
[Python Script Node (via Execute Command)]
       │  Run item_forecasting_improved.ipynb via papermill
       │  papermill item_forecasting_improved.ipynb output.ipynb -p TRAIN_END "YYYY-MM"
       ▼
[Read Binary File Node]
       │  Pick up item_forecasts_improved.csv from disk
       ▼
[Spreadsheet File Node / Google Sheets Node]
       │  Push CSV to shared Google Sheet or SharePoint Excel for planner access
       ▼
[Send Email / Slack Message Node]
       │  Send summary: "Q2 2026 forecasts updated — total 3-month outlook: $X.XM"
       │  Attach top-20 items by forecast revenue
       ▼
[IF Node — anomaly check]
       │  If any month's total forecast deviates > 20% from prior forecast:
       ▼
[Alert Node]
       │  Notify analyst team for manual review before distribution
```

**Why n8n specifically**

- **Self-hostable**: Can run on a small VM (2 vCPU, 4 GB RAM is sufficient for this pipeline), so Kenda's sales data never leaves their environment.
- **No-code connectors**: Native nodes for Google Sheets, Slack, email, HTTP, and file I/O mean no glue code is needed to connect the notebook output to downstream consumers.
- **Parameterised notebooks via Papermill**: The `TRAIN_END` parameter can be injected at runtime so the same notebook always trains on the full available history without manual edits.
- **Retry and error handling**: If the data pull fails or the model crashes, n8n handles retries and routes failures to an alert channel — the pipeline self-monitors.
- **Extensibility**: As the system matures, the same workflow can be extended to write forecasts directly into an ERP or inventory management system via API calls, eliminating the manual CSV handoff entirely.

**Scaling to more items or categories**

The LightGBM model is a global model. It trains on all items simultaneously, so adding more SKUs or product lines does not require retraining separate models. The primary scaling consideration is compute time for the Optuna hyperparameter search (currently ~5 minutes for 50 trials). For monthly retraining where hyperparameters are locked in and only the model weights are refreshed, a full retrain takes under 2 minutes on a standard laptop. The n8n scheduled trigger can be set with a 30-minute timeout, giving ample headroom.

---

## Output Files

### `item_forecasts_production.csv`
Generated by `item_forecasting.ipynb` (v4). Contains per-item forecasts for Mar/Apr/May 2026 with proportional reconciliation and the original Forecast Confidence Score system.

### `item_forecasts_improved.csv`
Generated by `item_forecasting_improved.ipynb` (final). The production deliverable. Columns:

| Column | Description |
|---|---|
| `ItemCode` | SKU identifier |
| `2026-03`, `2026-04`, `2026-05` | MinT-reconciled revenue forecast per month |
| `Total_Q2_2026` | 3-month total |
| `item_pl` | Product line |
| `tier` | Revenue tier (Tier_A through Tier_E) |
| `is_intermittent` | Whether item used CrostonSBA/AutoETS vs LightGBM |
| `ForecastMethod` | `ensemble_lgb` or `intermittent_stats` |
| `confidence_score` | 0–100 numeric score |
| `confidence_grade` | A (trust), B (normal buffer), C (extra buffer), D (manual review) |

---

*Data: Kenda Tires ATW sales records, January 2022 – February 2026.*
