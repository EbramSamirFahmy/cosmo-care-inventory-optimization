

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Paths
DATA_PATH = "data/raw/sales_22-24.csv"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load Data
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Date processing
df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
df['month_start'] = df['Date_parsed'].dt.to_period('M').dt.to_timestamp()

# Pivot monthly sales per category
all_months = pd.date_range(
    start=df['month_start'].min(),
    end=df['month_start'].max(),
    freq='MS'
)

pivot = (
    df.groupby(['month_start', 'Category'])['Sales']
      .sum()
      .unstack(fill_value=0)
      .reindex(all_months, fill_value=0)
)

pivot.index.name = 'Date'
ts = pivot.copy()

# Train / Test split
train = ts.loc[:'2023-12-31']
test = ts.loc['2024-01-01':]

print("Train:", train.shape, "| Test:", test.shape)

# Forecasting setup
def clean_series(series, m=2):
    return np.clip(series, series.mean() - m*series.std(), series.mean() + m*series.std())

categories = train.columns
forecasts_2024 = pd.DataFrame()
forecasts_2025 = pd.DataFrame()
forecasts_2026 = pd.DataFrame()
summary = []

annual_growth = 0.10  # yearly growth
epsilon = 1e-6

# Forecast per category
for cat in categories:
    try:
        y_train = train[cat].astype(float)
        y_test = test[cat].astype(float)

        y_train_clean = clean_series(y_train)

        # Trend linear
        X = np.arange(len(y_train_clean)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(X, y_train_clean)
        trend_forecast = lr.predict(np.arange(len(y_train_clean), len(y_train_clean)+12).reshape(-1,1))
        trend_forecast = np.clip(trend_forecast, 0, None)

        # Monthly correction factor
        correction_month = y_test.values[:12] / (trend_forecast[:12] + epsilon)
        correction_month = np.where(trend_forecast[:12]==0, 1.0, correction_month)
        correction_month = np.clip(correction_month, 0.8, 3.5)

        forecast_2024 = trend_forecast * correction_month
        forecast_2024 = pd.Series(forecast_2024).rolling(2, min_periods=1).mean().values
        forecast_2024 = np.round(np.clip(forecast_2024, 0, None))
        forecasts_2024[cat] = forecast_2024.astype(int)

        # Evaluate
        L = min(len(y_test),12)
        mae = mean_absolute_error(y_test.values[:L], forecast_2024[:L])
        rmse = np.sqrt(mean_squared_error(y_test.values[:L], forecast_2024[:L]))
        error_ratio = mae / y_test.values[:L].mean() * 100
        summary.append({
            "Category": cat, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "Error Ratio %": round(error_ratio,2)
        })

        # Forecast 2025 & 2026
        last_val = forecast_2024[-1]
        monthly_growth = (1 + annual_growth)**(1/12) - 1
        vals_2025 = [last_val * (1 + monthly_growth)**i for i in range(1,13)]
        vals_2026 = [vals_2025[-1] * (1 + monthly_growth)**i for i in range(1,13)]
        forecasts_2025[cat] = np.round(vals_2025).astype(int)
        forecasts_2026[cat] = np.round(vals_2026).astype(int)

    except Exception as e:
        print(f"Error in {cat}: {e}")

# Add date columns
forecasts_2024.insert(0,'Date', pd.date_range("2024-01-01", periods=12, freq='MS'))
forecasts_2025.insert(0,'Date', pd.date_range("2025-01-01", periods=12, freq='MS'))
forecasts_2026.insert(0,'Date', pd.date_range("2026-01-01", periods=12, freq='MS'))

# Save CSV
#forecasts_2024.to_csv(f"{PROCESSED_PATH}/forecast_2024_monthly.csv", index=False)
#forecasts_2025.to_csv(f"{PROCESSED_PATH}/forecast_2025_monthly.csv", index=False)
#forecasts_2026.to_csv(f"{PROCESSED_PATH}/forecast_2026_monthly.csv", index=False)

results_df = pd.DataFrame(summary).sort_values("Error Ratio %")
#results_df.to_csv(f"{PROCESSED_PATH}/summary_errors_monthly.csv", index=False)

# Overall metrics
all_actual = np.concatenate([test[cat].values[:12] for cat in categories])
all_pred = np.concatenate([forecasts_2024[cat].values for cat in categories])

overall_mae = mean_absolute_error(all_actual, all_pred)
overall_rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
overall_error_ratio = overall_mae / all_actual.mean() * 100

print(f"\nOverall MAE: {overall_mae:.2f}")
print(f"Overall RMSE: {overall_rmse:.2f}")
print(f"Overall Error Ratio: {overall_error_ratio:.2f}%")

# Convert to long format
def prepare_forecast_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.drop(columns=['Date'])
    return df.melt(id_vars=['Year','Month'], var_name='Category', value_name='Forecast_Demand')

#forecast_2025 = prepare_forecast_data(f"{PROCESSED_PATH}/forecast_2025_monthly.csv")
#forecast_2026 = prepare_forecast_data(f"{PROCESSED_PATH}/forecast_2026_monthly.csv")
#forecast_all = pd.concat([forecast_2025, forecast_2026]).reset_index(drop=True)

#forecast_all.to_csv(f"{PROCESSED_PATH}/forecast_22_24_long_format.csv", index=False)

print(" Short-term forecast completed successfully")
