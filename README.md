# Global Weather Forecasting Project

## Overview

This project involves forecasting future temperatures based on past global daily temperature data. The dataset is processed, analyzed, and modeled using statistical methods to predict future temperatures. Two models are explored: ARIMA and SARIMA.

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Check and Make Data Stationary](#4-check-and-make-data-stationary)
5. [Split the Data into Training and Testing Sets](#5-split-the-data-into-training-and-testing-sets)
6. [Build and Train the ARIMA Model](#6-build-and-train-the-arima-model)
7. [Make Predictions and Evaluate the Model](#7-make-predictions-and-evaluate-the-model)
8. [Forecast Future Temperatures](#8-forecast-future-temperatures)
9. [SARIMA Model](#9-sarima-model)
   - [Fit the SARIMA Model](#91-fit-the-sarima-model)
   - [Make Predictions and Evaluate the Model](#92-make-predictions-and-evaluate-the-model)
   - [Forecast Future Temperatures](#93-forecast-future-temperatures)

## 1. Data Collection

The dataset is loaded using Pandas from a CSV file.

```python
import pandas as pd

# Load the dataset
file_path = '/content/GlobalWeatherRepository.csv'
data = pd.read_csv(file_path)

# Create DataFrame
df = pd.DataFrame(data)
```

## 2. Data Preprocessing

### Steps:

1. Convert the `'last_updated'` column to datetime format.
2. Check for and handle missing values using forward fill.

```python
# Convert 'last_updated' column to datetime
df['last_updated'] = pd.to_datetime(df['last_updated'])

# Handle missing values (example: fill with forward fill method)
df.fillna(method='ffill', inplace=True)
```

## 3. Exploratory Data Analysis (EDA)

### Steps:

1. Plot temperature over time for a specific country (e.g., India).
2. Visualize the distribution of temperature using Seaborn.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the temperature over time for a specific country (example: 'India')
country = 'India'
country_data = df[df['country'] == country]

plt.figure(figsize=(15, 6))
plt.plot(country_data['last_updated'], country_data['temperature_celsius'], label=country)
plt.xlabel('Date')
plt.ylabel('temperature_celsius')
plt.title(f'Temperature over Time for {country}')
plt.legend()
plt.show()

# Distribution of temperature
sns.histplot(df['temperature_celsius'], kde=True)
plt.title('Temperature Distribution')
plt.show()
```

## 4. Check and Make Data Stationary

### Steps:

1. Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.
2. Apply differencing to make the data stationary if necessary.

```python
from statsmodels.tsa.stattools import adfuller

# Function to perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is non-stationary.")

# Perform ADF test on the original data
adf_test(df['temperature_celsius'])

# If the data is non-stationary, make it stationary
df_diff = df['temperature_celsius'].diff().dropna()

# Reindex df_diff with the 'last_updated' column, excluding the first date
df_diff.index = df['last_updated'][1:]
```

## 5. Split the Data into Training and Testing Sets

### Steps:

1. Select a specific country (e.g., India) and set the `'last_updated'` column as the index.
2. Split the data into training (80%) and testing (20%) sets.

```python
from sklearn.model_selection import train_test_split

# Split the data
train_size = int(len(country_data) * 0.8)
train_data, test_data = country_data[:train_size], country_data[train_size:]
```

## 6. Build and Train the ARIMA Model

### Steps:

1. Fit an ARIMA model on the training data.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit the model
model = ARIMA(train_data['temperature_celsius'], order=(5, 1, 0))
model_fit = model.fit()
```

## 7. Make Predictions and Evaluate the Model

### Steps:

1. Forecast temperatures for the testing data period.
2. Evaluate the model using Mean Squared Error (MSE).
3. Plot the predictions against actual data.

```python
from sklearn.metrics import mean_squared_error

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))

# Evaluate the model
mse = mean_squared_error(test_data['temperature_celsius'], predictions)
```

## 8. Forecast Future Temperatures

### Steps:

1. Forecast temperatures for the next 30 days and plot the results.

```python
# Forecast future temperatures
future_steps = 30  # Example: Forecasting for the next 30 days
future_forecast = model_fit.forecast(steps=future_steps)
```

## 9. SARIMA Model

### Steps:

1. Fit a Seasonal ARIMA (SARIMA) model on the training data.
2. Make predictions and evaluate the model using MSE.
3. Forecast future temperatures and visualize the results.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit the SARIMA model
sarima_model = SARIMAX(train_data['temperature_celsius'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
```
