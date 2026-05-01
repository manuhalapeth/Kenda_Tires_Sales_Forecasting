"""
tidy_output_file.py
Input:  item_forecasts_improved.csv
Output: item_forecasts_tidy.csv

Tidy-data rules applied
-----------------------
1. All column names snake_case.
2. year_month is a proper ISO-8601 date (first day of the month).
3. product_line is a zero-padded 4-digit string code, categorical, not numeric.
4. is_intermittent is 0/1 integer (CSV has no native bool type).
5. forecast_revenue is floored at 0 and rounded to 2 dp.
6. Columns ordered: identifiers → measure → categorical descriptors → numeric metadata.
7. Rows sorted by item_code, year_month.
"""

import pandas as pd

INPUT  = "item_forecasts_improved.csv"
OUTPUT = "item_forecasts_tidy.csv"

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)

# ── 1. standardise column names to snake_case ─────────────────────────────────
df = df.rename(columns={
    "ItemCode":         "item_code",
    "YearMonth":        "year_month",
    "Forecast_Revenue": "forecast_revenue",
    "ForecastMethod":   "forecast_method",
    "item_pl":          "product_line",
})

# ── 2. year_month → ISO-8601 date (first of month) ───────────────────────────
df["year_month"] = pd.to_datetime(df["year_month"] + "-01").dt.strftime("%Y-%m-%d")

# ── 3. product_line: int code → zero-padded 4-digit string ───────────────────
df["product_line"] = df["product_line"].astype(int).astype(str).str.zfill(4)

# ── 4. is_intermittent: bool → 0/1 ───────────────────────────────────────────
df["is_intermittent"] = df["is_intermittent"].astype(int)

# ── 5. forecast_revenue: floor at 0, round to 2 dp ───────────────────────────
df["forecast_revenue"] = df["forecast_revenue"].clip(lower=0).round(2)

# ── 6. column order ───────────────────────────────────────────────────────────
col_order = [
    # identifiers
    "item_code",
    "year_month",
    # measure
    "forecast_revenue",
    # categorical descriptors
    "product_line",
    "tier",
    "is_intermittent",
    "forecast_method",
    "confidence_grade",
    # numeric metadata
    "confidence_score",
    "item_mean",
    "item_cv",
    "item_active_rate",
    "item_months",
    "item_trend_slope",
]
df = df[col_order]

# ── 7. sort ───────────────────────────────────────────────────────────────────
df = df.sort_values(["item_code", "year_month"]).reset_index(drop=True)

# ── save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print()
print("Column types:")
print(df.dtypes.to_string())
print()
print("Sample (first 6 rows):")
print(df.head(6).to_string(index=False))
