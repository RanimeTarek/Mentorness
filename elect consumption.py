import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Load the dataset (replace with your dataset path)
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates={'DateTime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False)

# Clean and preprocess the dataset
# Handle missing values
data.replace('?', np.nan, inplace=True)
data = data.dropna()

# Convert DateTime column to datetime format and set as index
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M:%S')
data.set_index('DateTime', inplace=True)

# Select relevant columns
data = data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

# Resample daily to average values
data = data.resample('D').mean()

# Plot daily electricity consumption
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Global_active_power'], marker='o', linestyle='-')
plt.title('Daily Electricity Consumption')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kW)')
plt.grid(True)
plt.show()

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train['Global_active_power'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate forecast
mse = mean_squared_error(test['Global_active_power'], forecast)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# Plot forecast vs. actual
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Global_active_power'], label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast vs. Actual Electricity Consumption')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.grid(True)
plt.show()

