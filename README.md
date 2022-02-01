* Time Series Analysis 

import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Currency pair exchange rates for CAD/JPY
cad_jpy_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
cad_jpy_df.head()

# Trim the dataset to begin on January 1st, 1990
cad_jpy_df = cad_jpy_df.loc["1990-01-01":, :]
cad_jpy_df.head()

# Plot just the "Price" column from the dataframe:
cad_jpy_df.Price.plot(figsize=(20,10))

import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the exchange rate price into two separate series:
cad_jpy_noise, cad_jpy_trend = sm.tsa.filters.hpfilter(cad_jpy_df["Price"])

# Create a dataframe of just the exchange rate price, and add columns for "noise" and "trend" series from above:
cad_jpy_HP = pd.DataFrame()
cad_jpy_HP["Price"] = cad_jpy_df["Price"]
cad_jpy_HP["noise"] = cad_jpy_noise
cad_jpy_HP["trend"] = cad_jpy_trend
cad_jpy_HP.head()

# Plot the Exchange Rate Price vs. the Trend for 2015 to the present
cad_jpy_PvsT = cad_jpy_HP.loc["2015-01-01" :, :]
cad_jpy_PvsT = cad_jpy_PvsT.drop(columns=["noise"])
cad_jpy_PvsT.plot(figsize=(20,10),title='Price vs. Trend').get_figure().savefig('Price vs. Trend.png')

# Plot the Settle Noise
cad_jpy_noise.plot(figsize=(20,10), title='Noise').get_figure().savefig('noise.png')

# Create a series using "Price" percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (cad_jpy_df[["Price"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()

import statsmodels.api as sm

# Estimate and ARMA model using statsmodels (use order=(2, 1))
from statsmodels.tsa.arima_model import ARMA
model_arma = ARMA(returns.values, order=(2,1))

# Fit the model and assign it to a variable called results
results_arma = model_arma.fit()

# Output model summary results:
results_arma.summary()

# Plot the 5 Day Returns Forecast
pd.DataFrame(results_arma.forecast(steps=5)[0]).plot(title="5 Day Returns Forecast").get_figure().savefig('arma.png')

from statsmodels.tsa.arima_model import ARIMA

# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))
from statsmodels.tsa.arima_model import ARIMA
model_arima = ARIMA(cad_jpy_df['Price'], order=(5, 1, 1))

# Fit the model
results_arima = model_arima.fit()

# Output model summary results:
results_arima.summary()

# Plot the 5 Day Price Forecast
pd.DataFrame(results_arima.forecast(steps=5)[0]).plot(title="5 Day Future Price Forecast").get_figure().savefig('arima.png')

import arch as arch
from arch import arch_model

# Estimate a GARCH model:
model = arch_model(returns.Price, mean="Zero", vol="GARCH", p=2, q=1)

# Fit the model
res = model.fit(disp="off")

# Summarize the model results
res.summary()

# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day

# Create a 5 day forecast of volatility
forecast_horizon = 5

# Start the forecast using the last_day calculated above
forecasts = res.forecast(start='2020-06-04', horizon=forecast_horizon)
forecasts

# Annualize the forecast
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()

# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
final.head()

# Plot the final forecast
final.plot()
