# WeatherPredictor

## Overview

**WeatherPredictor** is a comprehensive time series analysis project aimed at forecasting future temperatures based on historical data. Utilizing advanced statistical models such as SARIMA, this project provides insights and predictions for temperature trends across various global cities.

## Features

- **Data Preprocessing:** Efficient handling of missing values, and converting date columns to appropriate datetime format.
- **Exploratory Data Analysis (EDA):** Visualization of temperature trends over time for different countries.
- **Stationarity Testing:** Using the Augmented Dickey-Fuller (ADF) test to check the stationarity of the time series data.
- **ARIMA and SARIMA Modeling:** Applying ARIMA and Seasonal ARIMA models to forecast future temperatures.
- **Evaluation Metrics:** Calculating Mean Squared Error (MSE) to evaluate model performance.
- **Future Forecasting:** Predicting future temperature trends for a specified period.

## Project Structure

- `data/`: Directory containing the dataset (e.g., `GlobalWeatherRepository.csv`).
- `notebooks/`: Jupyter notebooks for data analysis and model development.
- `scripts/`: Python scripts for data preprocessing, modeling, and evaluation.
- `visualizations/`: Directory for storing generated plots and graphs.

## Getting Started

### Prerequisites

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/WeatherPredictor.git
cd WeatherPredictor
pip install -r requirements.txt
```

### Usage

1. **Load and Preprocess Data:**

   ```python
   import pandas as pd

   # Load the dataset
   file_path = '/content/GlobalWeatherRepository.csv'
   data = pd.read_csv(file_path)

   # Display the first few rows of the dataset
   data.head()

   # Create DataFrame
   df = pd.DataFrame(data)

   df.info()
   df.describe()

   # Convert 'last_updated' column to datetime
   df['last_updated'] = pd.to_datetime(df['last_updated'])

   # Check for missing values
   print(df.isnull().sum())

   # Handle missing values (example: fill with forward fill method)
   df.fillna(method='ffill', inplace=True)

   # Verify the changes
   df.head()
   df.info()
   ```

2. **Exploratory Data Analysis:**

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

3. **Train-Test Split:**

   ```python
   from sklearn.model_selection import train_test_split

   # Assuming we are forecasting for one specific country
   country = 'India'
   country_data = df[df['country'] == country]

   # Set the 'last_updated' column as the index
   country_data.set_index('last_updated', inplace=True)

   # Split the data
   train_size = int(len(country_data) * 0.8)
   train_data, test_data = country_data[:train_size], country_data[train_size:]

   print(train_data.shape, test_data.shape)
   ```

4. **ARIMA Modeling and Forecasting:**

   ```python
   from statsmodels.tsa.arima.model import ARIMA

   # Fit the model
   model = ARIMA(train_data['temperature_celsius'], order=(5, 1, 0))
   model_fit = model.fit()

   # Print the model summary
   print(model_fit.summary())

   # Make predictions
   predictions = model_fit.forecast(steps=len(test_data))

   # Evaluate the model
   from sklearn.metrics import mean_squared_error

   mse = mean_squared_error(test_data['temperature_celsius'], predictions)
   print(f'Mean Squared Error: {mse}')

   # Plot the predictions
   plt.figure(figsize=(15, 6))
   plt.plot(train_data['temperature_celsius'], label='Training Data')
   plt.plot(test_data['temperature_celsius'], label='Test Data')
   plt.plot(test_data.index, predictions, label='Predictions', color='red')
   plt.xlabel('Date')
   plt.ylabel('temperature_celsius')
   plt.title(f'Temperature Predictions for {country}')
   plt.legend()
   plt.show()

   # Forecast future temperatures
   future_steps = 30  # Example: Forecasting for the next 30 days
   future_forecast = model_fit.forecast(steps=future_steps)

   # Plot the future forecast
   plt.figure(figsize=(15, 6))
   plt.plot(country_data['temperature_celsius'], label='Historical Data')
   plt.plot(pd.date_range(start=country_data.index[-1], periods=future_steps, freq='D'), future_forecast, label='Future Forecast', color='green')
   plt.xlabel('Date')
   plt.ylabel('temperature_celsius')
   plt.title(f'Future Temperature Forecast for {country}')
   plt.legend()
   plt.show()
   ```

5. **SARIMA Modeling and Forecasting:**

   ```python
   from statsmodels.tsa.statespace.sarimax import SARIMAX

   # Fit the SARIMA model
   sarima_model = SARIMAX(train_data['temperature_celsius'],
                          order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 12))
   sarima_fit = sarima_model.fit(disp=False)

   # Print the model summary
   print(sarima_fit.summary())

   # Make predictions
   sarima_predictions = sarima_fit.predict(start=test_data.index[0], end=test_data.index[-1], dynamic=False)

   # Evaluate the model
   sarima_mse = mean_squared_error(test_data['temperature_celsius'], sarima_predictions)
   print(f'SARIMA Mean Squared Error: {sarima_mse}')

   # Plot the predictions
   plt.figure(figsize=(15, 6))
   plt.plot(train_data['temperature_celsius'], label='Training Data')
   plt.plot(test_data['temperature_celsius'], label='Test Data')
   plt.plot(test_data.index, sarima_predictions, label='SARIMA Predictions', color='red')
   plt.xlabel('Date')
   plt.ylabel('temperature_celsius')
   plt.title(f'Temperature Predictions for {country}')
   plt.legend()
   plt.show()

   # Forecast future temperatures
   future_steps = 30  # Example: Forecasting for the next 30 days
   sarima_forecast = sarima_fit.get_forecast(steps=future_steps)
   sarima_forecast_index = pd.date_range(start=test_data.index[-1], periods=future_steps, freq='D')
   sarima_forecast_values = sarima_forecast.predicted_mean

   # Plot the future forecast
   plt.figure(figsize=(15, 6))
   plt.plot(country_data['temperature_celsius'], label='Historical Data')
   plt.plot(sarima_forecast_index, sarima_forecast_values, label='SARIMA Future Forecast', color='green')
   plt.xlabel('Date')
   plt.ylabel('temperature_celsius')
   plt.title(f'Future Temperature Forecast for {country}')
   plt.legend()
   plt.show()
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
