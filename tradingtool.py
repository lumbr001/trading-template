import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load the dataset
file_path = "/Nifty Financial Services Historical Data.csv"
df = pd.read_csv(file_path)

# Data Cleaning
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
numeric_cols = ["Price", "Open", "High", "Low"]
for col in numeric_cols:
    df[col] = df[col].str.replace(",", "").astype(float)
df["Change %"] = df["Change %"].str.replace("%", "").astype(float) / 100

df.sort_values(by="Date", inplace=True)
df.set_index("Date", inplace=True)

# ARIMA Model Training
model = ARIMA(df["Price"], order=(5,1,0))
model_fit = model.fit()

# Forecasting for Next 30 Trading Days with Confidence Intervals
forecast_result = model_fit.get_forecast(steps=30)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI

# Generate future dates from the forecast result's index
future_dates = forecast.index

# Plot the Predictions with Confidence Intervals
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Price"], label="Historical Prices", color="blue")
plt.plot(future_dates, forecast, label="Predicted Prices", color="red", linestyle="dashed")
plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label="95% Confidence Interval")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Nifty Financial Services Price Prediction (ARIMA) with Confidence Intervals")
plt.legend()
plt.grid()
plt.show()

# Display Forecasted Values with Confidence Intervals
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": forecast,
    "Lower Bound": conf_int.iloc[:, 0],
    "Upper Bound": conf_int.iloc[:, 1]
})
print(forecast_df)
