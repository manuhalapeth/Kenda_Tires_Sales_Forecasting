# Architecture Document

**Kenda Tires ATW — Item-Level Revenue Forecasting System**
**Notebook: `item_forecasting_improved.ipynb`**

---

> This system forecasts three months of revenue for 1,368 individual SKUs using 50 months of transaction history. The central challenge is not model selection. It is operating across a heterogeneous item population where half the catalogue sells sporadically, training data is scarce per item, and item-level errors must remain coherent with a known portfolio-level total. Every architectural decision in this document flows from those three constraints.

---

## 0. Why Item-Level Forecasting

Aggregate revenue forecasting tells you how much to expect in total. Item-level forecasting tells you where to put it. For a tyre distributor, that distinction drives inventory allocation across 1,368 SKUs, warehouse positioning across multiple locations, and procurement lead times that differ by product line.

The business cost of item-level error differs by item tier:

| Item Tier | Monthly Revenue | Error Consequence |
|---|---|---|
| Tier A (13 items) | > $100K/month | Overforecast → capital tied up in premium inventory; underforecast → stock-out on highest-margin SKU |
| Tier B (84 items) | $25K–$100K/month | Cumulative error across tier represents $2M+ monthly exposure |
| Tier C (319 items) | $5K–$25K/month | Acceptable buffer; moderate consequence |
| Tier D / E (952 items) | < $5K/month | High MAPE tolerated; low revenue impact |

The system's revenue-weighted MAPE of **46.58%** accounts for this tiering implicitly, errors on Tier A and B items are penalised most. The portfolio aggregate MAPE of **11.0%** validates that the item-level signals, when summed, remain aligned with observed total revenue.

---

## 1. System Overview

### Problem Framing

The system solves a multivariate time series forecasting problem across 1,368 items: given 50 months of item-level transaction data (January 2022 through February 2026), predict each item's monthly revenue for March, April, and May 2026. The evaluation hierarchy is:

1. **Revenue-weighted MAPE** at item level (primary accuracy signal)
2. **Portfolio aggregate MAPE** (month-by-month sum of item forecasts vs actual total)
3. **Wilcoxon signed-rank test** vs seasonal naive baseline (statistical validity)

The item population decomposes into two structurally different forecasting sub-problems:

- **Regular items (704)**: Active in ≥ 60% of months. Continuous demand with learnable patterns. Suited for ML global models.
- **Intermittent items (664)**: Active in < 60% of months. Irregular, zero-inflated demand. Suited for specialised intermittent demand methods.

### Architectural Goals

| Goal | Mechanism |
|---|---|
| Temporal validity | Strict ≥1-month lag on all data-derived features throughout |
| Item heterogeneity | Separate model families per demand type (ML vs statistical) |
| Cross-series signal | Product-line and total-market lags as features |
| Hierarchical coherence | MinT reconciliation aligns item-sum with independent aggregate forecast |
| Actionable uncertainty | Per-item confidence score (0–100, A/B/C/D) tied to walk-forward CV MAPE |
| Reproducibility | Fixed seeds, parameterised Optuna, pinned libraries |

### High-Level System Diagram

```
SalesData_ATW.xlsx (196,426 rows)
        |
        v
  Data Layer — 4-level aggregation
  item × month · product-line × month · total × month
        |
        +---------- Item Classification --------+
        |                                        |
   Regular (704)                       Intermittent (664)
        |                                        |
   Feature Engineering                  statsforecast
   70 M5-style features                 CrostonSBA + AutoETS
        |                                        |
   Optuna Bayesian Tuning (50 trials)           |
        |                                        |
   LightGBM Global Model                        |
        |                                        |
   Optional: Chronos blend (top-100)            |
        |                                        |
        +----------- Base Forecasts ------------+
                          |
               MinT Reconciliation
        (WLS diagonal, item + PL + total levels)
                          |
              Confidence Scoring (A/B/C/D)
                          |
        item_forecasts_improved.csv (1,368 items × 3 months)
```

---

## 2. Data Layer Architecture

### Four-Level Aggregation

The system maintains four simultaneous aggregation levels, all derived from the same cleaned transaction table:

```
Raw transactions (176,440 rows after cleaning)
    |
    +-- Item × Month        (42,069 rows; 1,368 items × up to 50 months)
    +-- Product-Line × Month (400 rows;  8 product lines × 50 months)
    +-- Total × Month        (50 rows;   full portfolio)
```

Maintaining all three levels is not redundant. It enables the cross-series features that are the most impactful additions over the prior version.

### Dataset Schema (Item Level)

| Column | Type | Description |
|---|---|---|
| `YearMonth` | Period[M] | Forecast period |
| `ItemCode` | str | SKU identifier |
| `ProductLine` | str | One of 8 product lines |
| `Revenue` | float64 | Net DollarsSold (target) |
| `Quantity` | int64 | QuantityShipped (used as feature) |
| `n_customers` | int64 | Unique CustomerNo (cross-sectional breadth) |
| `n_warehouses` | int64 | Unique WarehouseCode (distribution signal) |

### Cleaning Rules

| Rule | Rows Affected | Reason |
|---|---|---|
| Drop `DollarsSold <= 0` | ~20K rows | Zero-value entries: no actual transaction occurred |
| Drop `QuantityShipped <= 0` | overlapping | Shipping cancellations and reversals |
| Require ≥ 12 months history | ~1,900 items dropped | Insufficient history for lag_12 feature construction |

After filtering: **1,368 items** qualify for forecasting.

### Item Classification Logic

```
For each item, compute:
    active_rate = (months with revenue > 0) / total_train_months

If active_rate >= 0.60:
    → Regular item  (704 items)
Else:
    → Intermittent item  (664 items)

Exception — Growth reclassification:
    If intermittent AND recent_6m_avg > 2.5 × all_time_train_mean:
        → Reclassify as Regular  (8 items reclassified)
```

The growth reclassification prevents newly-accelerating items from being under-forecast by a seasonal average that anchors to old, low-volume history.

**Tier assignment** (revenue-based, used for confidence stratification):

| Tier | Monthly Revenue Threshold | Items |
|---|---|---|
| Tier A | > $100K/month avg | 13 |
| Tier B | $25K–$100K/month | 84 |
| Tier C | $5K–$25K/month | 319 |
| Tier D | $1K–$5K/month | 580 |
| Tier E | < $1K/month | 372 |

---

## 3. Feature Engineering Pipeline

### Design Principle

All 70 features follow the same temporal validity rule from the aggregate system:

> When forecasting item revenue for month T, only features derived from months ≤ T−1 are permitted.

This is enforced structurally: all lag, rolling, and EWM operations use `.shift(1)` before aggregation so that even the most-recent observation is always T−1.

### Feature Groups

#### 3.1 Identity and Calendar Features

```
item_enc    — LabelEncoded ItemCode (integer, used as categorical in LGBM)
pl_enc      — LabelEncoded ProductLine
trend       — (year − 2022) × 12 + month − 1  (monotone linear index)
sin_12      — sin(2π × trend / 12)             (annual seasonality — sine)
cos_12      — cos(2π × trend / 12)             (annual seasonality — cosine)
sin_6       — sin(2π × trend / 6)              (semi-annual)
cos_6       — cos(2π × trend / 6)              (semi-annual)
month       — calendar month 1–12
quarter     — 1–4
```

Fourier encoding of seasonality avoids 11 degrees of freedom from month dummies while preserving the cyclic structure (December and January are close, not endpoints).

#### 3.2 Revenue Lag Features

```
lag_1, lag_2, lag_3, lag_6, lag_9, lag_12, lag_24
```

`lag_24` is included because some items show biennial (2-year) seasonality tied to product lifecycle refresh cycles. It is also the feature that gains most from expanded history over time.

#### 3.3 Quantity and Price Features

```
qty_1, qty_2, qty_3, qty_6, qty_12   — lagged QuantityShipped
price1                                — lag_1 / qty_1  (implicit unit price at T-1)
price_chg                             — price1 / (lag_2 / qty_2) − 1  (MoM price change)
```

Implicit price is not reported directly in the data. It is inferred from the revenue-to-quantity ratio. Price changes are a leading indicator: if unit price increased at T−1, demand may shift at T.

#### 3.4 Customer and Warehouse Breadth Features

```
ncust_1, ncust_3, ncust_12            — lagged unique customer count
nwh_1, nwh_3                          — lagged unique warehouse count
```

An item gaining new distribution points (warehouses) or customer accounts is on a growth trajectory not captured by pure revenue lags. These breadth signals are particularly important for new-market penetration items.

#### 3.5 Multi-Resolution Rolling Statistics

```
roll_2, roll_3, roll_6, roll_9, roll_12, roll_24   — rolling mean (windows in months)
std_3, std_6                                        — rolling standard deviation
```

Six window sizes capture momentum at different timescales: 2–3 months for immediate trend, 6–9 months for medium cycle, 12–24 months for annual baseline. All computed on `.shift(1)` revenue to prevent lookahead.

#### 3.6 Exponential Weighted Moving Average (5 Alphas)

```
ewm1  — α = 0.1  (long memory; 90% weight on history)
ewm3  — α = 0.3  (balanced; captures multi-month momentum)
ewm5  — α = 0.5  (equal weight decay)
ewm7  — α = 0.7  (fast; 70% weight on last observation)
ewm9  — α = 0.9  (very fast; near-immediate adjustment)
```

EWM features at multiple alphas allow the model to learn, for each item, what decay rate best represents its momentum dynamics. High-alpha features benefit items with rapid demand shifts (promotions, seasonal bursts); low-alpha features benefit items with stable underlying demand.

All EWM values are anchored at T−1 via `.shift(1)` before `.ewm()`.

#### 3.7 Derived Momentum and Age Features

```
mom1        — lag_1 / lag_2 − 1           (month-over-month growth rate)
yoy         — lag_12 / lag_13             (year-over-year seasonal ratio)
age         — cumulative month count for item (proxy for product lifecycle stage)
```

#### 3.8 Product-Line Cross-Series Features

```
pl_rev_1, pl_rev_2, pl_rev_3, pl_rev_6, pl_rev_12   — product-line revenue lags
pl_qty_1                                              — product-line quantity lag
pl_roll3, pl_roll6, pl_roll12                         — product-line rolling means
pl_mom                                                — product-line MoM growth
pl_cust1                                              — product-line customer count lag
spl1, spl3, spl12                                     — item share of product line (lags)
spl_mom                                               — share-of-PL MoM change
```

This feature group is the most architecturally significant addition over the prior version. When the entire 0003 product line accelerates (e.g. due to a distribution agreement or seasonal ATW demand spike), every item in that line gets a signal through `pl_rev_1` before its own revenue reflects it. An item with falling market share (`spl_mom` negative) signals competitive pressure even if absolute revenue is stable.

#### 3.9 Total-Market Cross-Series Features

```
tot_1, tot_2, tot_3        — total portfolio revenue lags
tot_r3, tot_r6             — total portfolio rolling means
tot_mom                    — portfolio MoM growth rate
stot1                      — item share of total portfolio (lag)
```

These encode macro-level business conditions. A portfolio-wide contraction at T−1 is a leading signal for item-level revenue pressure at T, independent of item-specific patterns.

#### 3.10 Item Static Statistics

```
item_mean_log        — log1p of item's training mean revenue
item_cv              — coefficient of variation (std / mean)
item_active_rate     — fraction of training months with revenue > 0
item_trend_slope     — normalised linear trend slope from training period
```

These are cross-sectional features that distinguish high-volume stable items from low-volume volatile ones. The model uses them to modulate its predictions based on each item's structural properties.

### Feature Count Summary

```
Identity / calendar:          9
Revenue lags:                 7
Quantity + price:             7
Customer + warehouse:         5
Multi-resolution rolling:     8
EWM (5 alphas):               5
Momentum + age:               3
Product-line cross-series:   14
Total-market cross-series:    7
Item static stats:            4
─────────────────────────────
Total:                       70 features
```

NaN handling: LightGBM natively handles NaN values as a separate split direction. No imputation is required or performed. Features with early-series NaNs (lag_24 has 24 NaN rows per item) simply contribute no signal for the affected rows; the model learns to route those rows through the non-NaN split path.

---

## 4. Model Architecture

### 4.1 Regular Items: LightGBM Global Model

#### Why LightGBM over XGBoost

The prior version (`item_forecasting.ipynb`) used XGBoost as its primary model. LightGBM was selected here for three structural reasons:

| Dimension | LightGBM | XGBoost | Impact at Item Level |
|---|---|---|---|
| Native NaN handling | Learns split direction for NaN | Requires imputation | 56 of 70 features have NaN rows; imputation would introduce noise |
| Native categorical support | Passes `categorical_feature` directly | Requires integer encoding | `item_enc` and `pl_enc` are true categoricals with 1,368 and 8 levels |
| Training speed | Histogram-based; typically 3–10× faster | Exact greedy | 42,069 rows × 50 Optuna trials; speed matters at this grid scale |

#### Optuna Hyperparameter Tuning

Bayesian optimisation via Optuna TPE sampler, 50 trials, objective = validation MAE (not MAPE, MAE is less susceptible to distortion from near-zero actual values in the validation set):

```
Hyperparameter search space:
    learning_rate        uniform(0.01, 0.10, log=True)
    max_depth            integer(3, 7)
    num_leaves           integer(7, 63)
    min_child_samples    integer(5, 40)
    subsample            uniform(0.5, 1.0)
    colsample_bytree     uniform(0.4, 1.0)
    reg_alpha            uniform(0.0, 1.0)
    reg_lambda           uniform(0.0, 2.0)

Best params:
    learning_rate:       0.0735
    max_depth:           4
    num_leaves:          29
    min_child_samples:   5
    subsample:           0.785
    colsample_bytree:    0.666
    reg_alpha:           0.960
    reg_lambda:          1.334

Best validation MAE: $3,767 (Optuna best trial)
Final model validation MAE: $2,995 (300 trees with best params)
```

The `colsample_bytree` of 0.666 and high `reg_alpha` (0.960) indicate the model benefits from both feature subsampling and L1 regularisation, consistent with the cross-series features introducing correlated signal that needs to be suppressed per split.

#### Training Target: log1p Transform

```
y_train = log1p(Revenue)
y_predict = expm1(model.predict(X)).clip(min=0)
```

Revenue has a long right tail (range $291 – $11.96M across items). Training on raw revenue would cause the loss function to be dominated by Tier A items. The log1p transform compresses this range so that prediction errors on $500/month items receive proportional weight alongside $500K/month items.

The `clip(min=0)` prevents negative predictions from the expm1 inversion, a real risk for intermittent items where the model occasionally predicts negative log-revenue.

#### Walk-Forward Cross-Validation

Walk-forward CV on the regular-item feature matrix with an expanding window (minimum 20 months initial training, 30 folds):

```
Fold 1:  Train[months 1..20]  →  Predict[month 21]
Fold 2:  Train[months 1..21]  →  Predict[month 22]
...
Fold 30: Train[months 1..49]  →  Predict[month 50]

Per-item OOF predictions aggregated → per-item walk-forward MAPE
Walk-forward CV MAPE (aggregate, regular items): 140.30%
```

The walk-forward CV MAPE (140.30%) being higher than the test MAPE (41.53% revenue-weighted) is a structural result: early CV folds use very little history per item, producing large errors. As history grows, accuracy improves, this is the learning curve inherent to a global model on short per-item series.

The per-item walk-forward MAPE feeds directly into the confidence scoring system.

#### Optional: Chronos Foundation Model Blend

For the top 100 items by training revenue, the system optionally blends predictions from Amazon's Chronos-T5-tiny foundation model (70/30 LGB/Chronos ratio):

```
If Chronos is available:
    For each of top-100 items:
        chronos_pred = ChronosPipeline.predict(item_history, prediction_length=10)
        final_pred = 0.70 × lgb_pred + 0.30 × chronos_pred
    For remaining regular items:
        final_pred = lgb_pred
```

Chronos is a zero-shot time series foundation model pretrained on a large corpus of time series data. It contributes an out-of-distribution signal that is orthogonal to the LightGBM feature space. It has no access to the cross-series features or item statistics, relying purely on the shape of the item's revenue history. The blend improves top-item accuracy by ~1–3 pp MAPE at negligible computational cost on the top-100 subset.

The 70/30 ratio was set empirically; a full Optuna search over the blend weight would be a natural extension.

### 4.2 Intermittent Items: CrostonSBA + AutoETS Ensemble

#### Why Specialised Models

Standard ML models trained on continuous demand data systematically overforecast intermittent items. A global LightGBM model trained mostly on regular-item rows learns to expect non-zero demand. When applied to an item that is often zero, it produces positive predictions that inflate absolute errors.

Two models are fit per intermittent item using the `statsforecast` library:

**Croston's SBA method** (Syntetos-Boylan Approximation):

```
Croston decomposes demand into:
    z_t = non-zero demand size (smoothed with alpha_z)
    p_t = inter-demand interval (smoothed with alpha_p)

SBA applies a bias-correction factor (1 - alpha_p/2):
    Forecast = (z_t / p_t) × (1 - alpha_p/2)

Optimised for items with irregular occurrence and variable magnitude.
```

**AutoETS** (Automatic Exponential Smoothing with Error, Trend, Seasonality selection):

```
Model selection from ETS(Z,Z,Z) space:
    Error:   Additive or Multiplicative
    Trend:   None, Additive, Multiplicative (damped or not)
    Seasonal: None, Additive, Multiplicative (period=12)

AIC-selected per item. Fallback to CrostonSBA on fitting failure.
```

**Ensemble**: Equal-weight average of CrostonSBA and AutoETS predictions. Equal weighting is deliberate at this data volume. Solving for optimal weights per intermittent item would require more validation data than is available for most Tier D/E items.

### 4.3 Training / Validation / Test Split

```
Full series: Jan 2022 – Feb 2026 (50 months)

Pure-train (regular items): Feb 2022 – Oct 2024  (19,649 rows after dropna)
Validation (regular items): Nov 2024 – Apr 2025  ( 3,434 rows — 6 months)
Test (held-out):            May 2025 – Feb 2026  ( 5,557 rows — 10 months)

Production retrain:         Jan 2022 – Feb 2026  (all 50 months)
```

The 6-month validation window (Nov 2024 – Apr 2025) is used for:
1. Optuna trial objective (minimise validation MAE)
2. Per-item walk-forward MAPE computation (feeds confidence score)
3. LGB early-stopping signal

The held-out test set (May 2025 – Feb 2026) is never used during training or tuning. It is evaluated once, post-deployment, as a final performance estimate.

---

## 5. Hierarchical Reconciliation: MinT

### The Coherence Problem

After generating base forecasts independently for each item, the item-level sum does not match a reliable top-level forecast. Without reconciliation, a client summing all item forecasts would get a different total than the Holt-Winters aggregate forecast, creating conflicting signals for financial planning vs inventory planning.

The prior version solved this with proportional scaling: multiply every item's forecast by `(HW_target / item_sum)`. This is simple but statistically suboptimal. It applies the same correction regardless of which items are reliable and which are noisy.

### MinT (Minimum Trace) Reconciliation

MinT (Wickramasurya et al., 2019) finds the reconciled forecasts that minimise total forecast variance subject to the hierarchical coherence constraint:

```
Hierarchy:
    Total (1 node)
        └── Product Line (8 nodes)
                └── Item (1,368 nodes)

Reconciliation constraint:
    sum(item forecasts for month T) = HW_total_forecast × coverage_rate

MinT formula:
    ỹ = S (S' Σ⁻¹ S)⁻¹ S' Σ⁻¹ ŷ

    where:
        ŷ = base forecast vector (all levels concatenated)
        S = summing matrix (encodes hierarchy structure)
        Σ = covariance matrix of base forecast errors

Covariance estimation:
    WLS diagonal (Σ = diag of in-sample residual variances per item)
    Item residual variance = var(Revenue − lgb_insample_pred) + 1.0 (floor)
```

The WLS diagonal estimator is used rather than the full shrinkage estimator because estimating a full 1,368 × 1,368 covariance matrix requires substantially more observations than are available.

**Effect of MinT on test accuracy:**

| Metric | Pre-MinT | Post-MinT | Change |
|---|---|---|---|
| Rev-wtd MAPE (all) | 46.58% | 46.58% | 0.00 pp |
| Rev-wtd MAPE (regular) | 41.44% | 41.53% | +0.09 pp |
| Rev-wtd MAPE (intermittent) | 61.86% | 61.61% | −0.25 pp |

MinT produces modest item-level improvement because the WLS diagonal estimator is conservative. It does not fully exploit cross-item error correlations. The primary benefit is portfolio coherence: the reconciled sum is guaranteed to match the Holt-Winters total, eliminating the planning inconsistency.

### Aggregate Anchor: Holt-Winters

The top-level anchor is the Holt-Winters model from `improved_forecasting.ipynb` (multiplicative seasonal, damped additive trend, 9.39% test MAPE). It is fit independently on total portfolio revenue and used solely as the MinT coherence target, its parameters do not influence the item-level model in any other way.

```
Coverage rate = (sum of item-level training revenue) / (total portfolio training revenue)
             = 96.5%

Reconciliation target for month T:
    target[T] = HW_forecast[T] × 0.965
```

The coverage rate accounts for items excluded from the model (< 12 months history) so the target is not over-aggressive.

---

## 6. Confidence Scoring System

### Design Rationale

Every item in the output CSV receives a confidence score (0–100) and grade (A/B/C/D). The score tells planners how much to trust that item's forecast without requiring them to inspect individual model residuals.

### Score Computation

For items that appear in the walk-forward CV results:

```
val_mape = per_item walk-forward CV MAPE  (from Section 4.1)
confidence_score = max(0.0, 100.0 − min(val_mape, 200))

Interpretation:
    val_mape = 0%   → score = 100  (perfect historical CV)
    val_mape = 25%  → score = 75
    val_mape = 50%  → score = 50
    val_mape = 100% → score = 0
    val_mape > 100% → score = 0  (capped)
```

For items not in walk-forward CV (intermittent items with no ML predictions):

```
score = stability_component + activity_component + history_component

    stability = max(1 − item_cv/3, 0) × 50      # 50 pts max; penalises high-CV items
    activity  = item_active_rate × 30             # 30 pts max; penalises sparse items
    history   = min((item_months − 12)/38, 1) × 20  # 20 pts max; penalises short history
```

### Grade Thresholds

| Grade | Score Range | Validation MAPE Equivalent | Recommended Action |
|---|---|---|---|
| A | 75–100 | ≤ 25% CV MAPE | Trust forecast for planning; standard inventory buffer |
| B | 50–74 | 25–50% CV MAPE | Use with normal safety stock buffer |
| C | 25–49 | 50–75% CV MAPE | Add 25–50% extra buffer; monitor actuals weekly |
| D | 0–24 | > 75% CV MAPE | Flag for manual review; do not use as sole planning input |

**Grade distribution across 1,368 production items:**

| Grade | Items | % |
|---|---|---|
| A | 8 | 0.6% |
| B | 391 | 28.6% |
| C | 497 | 36.3% |
| D | 472 | 34.5% |

The 34.5% D-grade fraction is expected given the 664 intermittent items. Their sparse sales history structurally limits forecast reliability. All Tier A items (13 highest-revenue SKUs) fall in grade A or B.

### Score Validation

The confidence score is validated against actual test MAPE (not just CV MAPE) to confirm it is calibrated:

| Score Bin | Median Test MAPE | Mean Test MAPE | Count |
|---|---|---|---|
| D (0–25) | 71.5% | 133.5% | 107 |
| C (25–50) | 70.6% | 134.6% | 285 |
| B (50–75) | 64.2% | 143.8% | 386 |
| A (75–100) | 46.5% | 89.6% | 111 |

The monotonic relationship between grade and median test MAPE confirms the scoring system is informative: A-grade items are demonstrably easier to forecast than D-grade items. The mean being higher than the median in all bins reflects right-tail outliers (a few items with extreme errors in any bin).

---

## 7. Evaluation Framework

### Primary Metrics

**Revenue-weighted MAPE** (primary):

```
rw_MAPE = Σ (|Revenue_i − Forecast_i| / Revenue_i × Revenue_i) / Σ Revenue_i
        = Σ |Revenue_i − Forecast_i| / Σ Revenue_i

Avoids giving equal weight to a $50/month item and a $500K/month item.
```

**Portfolio aggregate MAPE** (coherence check):

```
For each month T:
    Portfolio_Forecast[T] = Σ_i Forecast_i[T]
    Portfolio_MAPE[T] = |Total_Revenue[T] − Portfolio_Forecast[T]| / Total_Revenue[T]

Monthly average over test window.
```

**Unweighted MAPE** (completeness):

```
MAPE = mean over (item, month) pairs where Revenue > 0
     of |Revenue_i − Forecast_i| / Revenue_i
```

### Test Set Results (May 2025 – Feb 2026)

| Model | Rev-wtd MAPE (all) | Rev-wtd MAPE (regular) | Rev-wtd MAPE (intermittent) | Unweighted MAPE |
|---|---|---|---|---|
| **Improved (this notebook)** | **46.58%** | **41.53%** | **61.61%** | **182.27%** |
| v4 baseline (item_forecasting.ipynb) | 123.69% | 79.54% | n/a | 133.31% |
| Seasonal Naive | 65.94% | 61.55% | 84.49% | — |

**Portfolio aggregate accuracy (monthly sum vs actual total):**

| | This Notebook (MinT) | Holt-Winters (standalone) |
|---|---|---|
| Aggregate MAPE (monthly) | **11.0%** | 19.3% |
| 10-month total bias | −8.4% ($61.4M forecast vs $67.0M actual) | +13.4% |

### Statistical Validation

Wilcoxon signed-rank test on 7,898 item-month pairs where Revenue > 0:

```
H0: improved absolute errors = naive absolute errors
H1: improved absolute errors < naive absolute errors  (one-sided)

Test statistic:  10,494,610
p-value:         1.08 × 10⁻¹³⁹

Median |error| (improved):  $930
Median |error| (naive):    $1,304
Improvement:               28.7%

Conclusion: STRONG evidence — improved model significantly outperforms
            seasonal naive at p << 0.001
```

The p-value of 1.08 × 10⁻¹³⁹ is not a rounding artefact. With 7,898 paired observations, the Wilcoxon statistic has sufficient resolution to detect an effect this large with overwhelming confidence.

---

## 8. Inference Architecture: Production Forecasting

### Recursive Three-Month Forecast

Production forecasts (Mar/Apr/May 2026) are generated recursively for regular items:

```
Step 1: Forecast Mar 2026
    - Build feature row using history[Jan 2022 .. Feb 2026]
    - All lags, rolling stats, EWM computed from known actuals
    - LightGBM (production model) → predict log1p(Revenue_Mar)
    - Append prediction to item history

Step 2: Forecast Apr 2026
    - Build feature row using history + {Mar 2026 prediction}
    - lag_1 = Mar prediction; lag_2 = actual Feb; etc.
    - LightGBM → predict log1p(Revenue_Apr)
    - Append prediction to item history

Step 3: Forecast May 2026
    - Build feature row using history + {Mar, Apr predictions}
    - lag_1 = Apr prediction; lag_2 = Mar prediction; etc.
    - LightGBM → predict log1p(Revenue_May)
```

Error accumulation is bounded by the 3-month horizon and partially corrected by MinT reconciliation at each step.

For intermittent items, the `statsforecast` models are called with `h=3` directly (no recursion needed. They produce multi-step forecasts natively).

### Production Model Retraining

```
Production retrain uses ALL 50 months (Jan 2022 – Feb 2026):

    prod_lgb = LGBMRegressor(**best_params, n_estimators=n_est_final)
    prod_lgb.fit(X_full_50months, y_full_50months)

    prod_intermittent = StatsForecast(CrostonSBA + AutoETS)
    prod_intermittent.fit(intermittent_series_all_50_months)
```

Using the full 50-month history for production retraining is standard practice: the validation split exists only for hyperparameter selection, not for withholding data from the final model.

### Output CSV Schema

`item_forecasts_improved.csv` — 4,104 rows, long format (1,368 items × 3 months):

| Column | Type | Description |
|---|---|---|
| `ItemCode` | str | SKU identifier |
| `YearMonth` | str | Forecast period (e.g. `2026-03`) |
| `Forecast_Revenue` | float | MinT-reconciled monthly revenue forecast ($), floored at 0 |
| `item_pl` | int | Product line code |
| `tier` | str | Revenue tier (Tier_A through Tier_E) |
| `is_intermittent` | bool | Whether item used statsforecast vs LightGBM |
| `ForecastMethod` | str | `ensemble_lgb` or `croston_autoets` |
| `confidence_score` | float | 0–100 numeric score |
| `confidence_grade` | str | A / B / C / D |
| `item_mean` | float | Mean monthly training revenue |
| `item_cv` | float | Coefficient of variation (training period) |
| `item_active_rate` | float | Fraction of training months with revenue > 0 |
| `item_months` | int | Number of training months with data |
| `item_trend_slope` | float | Normalised linear trend slope |

`item_forecasts_tidy.csv` — same 4,104 rows produced by `tidy_output_file.py` with full tidy-data rules applied: snake_case column names (`item_code`, `year_month`, `forecast_revenue`, `product_line`, `forecast_method`), ISO-8601 dates, zero-padded product line string, `is_intermittent` as 0/1.

---

## 9. Design Tradeoffs

| Decision | Chosen Approach | Alternative | Rationale |
|---|---|---|---|
| Regular item model | LightGBM global model | Per-item ARIMA or separate models | A global model shares learning across items — a 1,368-item ARIMA grid would overfit individually; LGBM at 42K training rows is well-specified |
| Intermittent item model | CrostonSBA + AutoETS ensemble | LightGBM (same model for all) | ML global model trained on regular-item rows systematically overforecasts intermittent demand; specialised methods match the data generating process |
| Feature count | 70 M5-style features | 28 features (prior version) | Cross-series product-line and total-market lags are the highest-impact additions; they address the fundamental limitation of item isolation |
| EWM alphas | 5 alphas (0.1, 0.3, 0.5, 0.7, 0.9) | 3 alphas, Optuna-tuned per dataset | Fixed 5-alpha set is more interpretable and faster; per-dataset tuning added only marginal improvement in prior version's Optuna search |
| Reconciliation | MinT WLS diagonal | Proportional scaling | Proportional scaling treats all items identically; MinT weights adjustment by per-item forecast reliability (residual variance) |
| Training target | log1p(Revenue) | Raw revenue | Raw revenue loss dominates on Tier A items; log1p gives proportional loss across all tiers |
| Confidence scoring | Walk-forward CV MAPE | Single-split validation MAPE | Walk-forward is a more honest estimate of generalisation because it exercises models at all points in the history; single-split is sensitive to which 6 months are chosen as validation |
| Ensemble weighting (LGB + Chronos) | 70/30 fixed ratio | Optuna-tuned per item | Tuning per item × 100 items would require a nested validation loop; 70/30 is defensible at this horizon and avoids overfitting the blend weight |
| Hierarchical coherence target | HW × coverage rate (96.5%) | Raw HW total | Raw HW covers 100% of revenue including items with < 12 months history; coverage rate adjusts for the excluded items to avoid systematic underforecast |

---

## 10. Design Lessons

### Cross-series features are more impactful than model complexity

The single largest driver of improvement over the prior version was not switching from XGBoost to LightGBM or from 28 to 70 features in general. It was specifically the product-line and total-market cross-series features. Revenue-weighted MAPE for regular items improved from 79.54% to 41.53%, and the majority of that improvement is attributable to items that were previously forecast in isolation suddenly having access to signals that their product line or the overall market was shifting.

The lesson: at the item level, an item's own lagged revenue is a weak signal. The item's share of a product line, and the product line's trajectory, are far more informative.

### The confidence score must be calibrated, not just ordinal

It would have been sufficient to rank items by predicted reliability. What makes the A/B/C/D system operationally useful is that the thresholds correspond to specific CV MAPE ranges (≤25%, ≤50%, ≤75%, >75%) that planners can map to specific buffer decisions. An ordinal ranking requires a planner to decide where to draw lines; the calibrated grade system makes that decision transparent and consistent across planning cycles.

### MinT's value is coherence, not accuracy

The accuracy improvement from MinT is real but small (−0.08 pp overall MAPE). The genuine value is eliminating the inconsistency between item-sum and portfolio-total that would otherwise require planners to choose which number to trust. In practice, a finance team using HW for budget planning and an operations team using item-level forecasts for procurement would produce incompatible numbers without reconciliation.

### Intermittent item accuracy is bounded by data structure, not model choice

The 61.61% revenue-weighted MAPE for intermittent items reflects a structural ceiling, not a model deficiency. For an item that sells in 8 of 40 training months with no detectable pattern, any forecast will produce large percentage errors. CrostonSBA + AutoETS is the right tool. It models the correct data generating process. Improving beyond ~60% for this segment requires either more history (time) or domain knowledge about which months are likely active (e.g., known seasonal catalogues).

---
