# stock_forecasting_project.py

# ---------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Load dataset
file_path = 'sp500.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Check for missing values
df = df[['Close']].dropna()

# ADF Test for stationarity
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

# Differencing if not stationary
if adf_test(df['Close']) > 0.05:
    df['Close_diff'] = df['Close'].diff().dropna()
else:
    df['Close_diff'] = df['Close']

# ---------------------------
# 2. EDA & VISUALIZATION
# ---------------------------
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(10,4))
plt.plot(df['Close'])
plt.title("S&P 500 Closing Price")
plt.show()

result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
result.plot()
plt.show()

plot_acf(df['Close_diff'].dropna())
plot_pacf(df['Close_diff'].dropna())
plt.show()

# ---------------------------
# 3. TRAIN-TEST SPLIT FOR VALIDATION
# ---------------------------
split_ratio = 0.9
split_index = int(len(df) * split_ratio)

train = df.iloc[:split_index]
test = df.iloc[split_index:]
forecast_horizon = len(test)

# ---------------------------
# 4. ARIMA MODEL
# ---------------------------
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(train['Close'], order=(5,1,0))
arima_result = arima_model.fit()
forecast_arima = arima_result.forecast(steps=forecast_horizon)

plt.figure()
plt.plot(df['Close'], label='Historical')
plt.plot(forecast_arima.index, forecast_arima, label='ARIMA Forecast')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

# ---------------------------
# 5. SARIMA MODEL
# ---------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(train['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.forecast(steps=forecast_horizon)

plt.figure()
plt.plot(df['Close'], label='Historical')
plt.plot(sarima_forecast.index, sarima_forecast, label='SARIMA Forecast')
plt.legend()
plt.title("SARIMA Forecast")
plt.show()

# ---------------------------
# 6. PROPHET MODEL
# ---------------------------
from prophet import Prophet

train_prophet = train.reset_index()[['Date', 'Close']]
train_prophet.columns = ['ds', 'y']

prophet_model = Prophet()
prophet_model.fit(train_prophet)

future = prophet_model.make_future_dataframe(periods=forecast_horizon)
forecast = prophet_model.predict(future)

forecast_prophet = forecast[['ds', 'yhat']].set_index('ds').iloc[-forecast_horizon:]

prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

# ---------------------------
# 7. LSTM MODEL
# ---------------------------
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Close']])

train_scaled = df_scaled[:split_index]
test_scaled = df_scaled[split_index - 60:]

X_train, y_train = [], []
for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

X_test, y_test = [], []
for i in range(60, len(test_scaled)):
    X_test.append(test_scaled[i-60:i, 0])
    y_test.append(test_scaled[i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_preds_scaled = model.predict(X_test)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

# ---------------------------
# 8. MODEL EVALUATION & COMPARISON
# ---------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse

mae_arima, rmse_arima = evaluate_model(test['Close'], forecast_arima)
mae_sarima, rmse_sarima = evaluate_model(test['Close'], sarima_forecast)
mae_prophet, rmse_prophet = evaluate_model(test['Close'], forecast_prophet['yhat'])
mae_lstm, rmse_lstm = evaluate_model(y_test_actual, lstm_preds)

results = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
    'MAE': [mae_arima, mae_sarima, mae_prophet, mae_lstm],
    'RMSE': [rmse_arima, rmse_sarima, rmse_prophet, rmse_lstm]
})

print(results)

# ---------------------------
# 9. DEPLOYMENT (Optional)
# ---------------------------
# Save all model results and predictions for Streamlit/Flask app

# ---------------------------
# END OF PIPELINE
# ---------------------------
