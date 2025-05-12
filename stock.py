import math
import numpy as np
import pandas as pd
import yfinance as yf  
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import requests  

# ⚙️ Disable GPU if not available
tf.config.set_visible_devices([], 'GPU')

# 📌 Function to fetch USD to INR exchange rate
def get_usd_to_inr():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    try:
        response = requests.get(url, timeout=5)  # Set timeout
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
        data = response.json()
        return data.get("rates", {}).get("INR", 83)  # Use 83 if INR not found
    except requests.RequestException:
        print("⚠️ Warning: Could not fetch live exchange rate. Using fallback rate of 83 INR/USD.")
        return 83


# 📌 Get today's date
today = date.today()
date_today = today.strftime("%Y-%m-%d")
date_start = '2022-01-01'  # 🔄 Updated start date to 2022

# 📌 Get stock data
stock = input("Enter stock name: ").strip().upper()  
symbol = stock  

try:
    df = yf.download(symbol, start=date_start, end=date_today, auto_adjust=False)

    if df.empty:  # Check if data is empty
        print(f"⚠️ No data found for {symbol}. Please check the stock ticker!")
        exit()

    print("\n🔍 Debug: Raw Downloaded DataFrame:")
    print(df.head())

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1) 

    # ✅ Rename columns safely
    expected_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    if len(df.columns) == len(expected_columns):  
        df.columns = expected_columns

    # Drop NaN values
    df = df.dropna(subset=['Close'])

    # ✅ Show last 5 trading days in descending order
    print("\n📊 Recent 5 Trading Days (Descending Order):")
    print(df.tail(5).iloc[::-1])  # Reverse order

except Exception as e:
    print(f"⚠️ Error fetching stock data: {e}")
    exit()

# 📊 Plot stock price data
plt.figure(figsize=(16, 6))
plt.title(f'{stock} from {date_start} to {date_today}', fontsize=16)
plt.plot(df['Close'], color='#039dfc', label=stock, linewidth=1.0)
plt.fill_between(df.index, 0, df['Close'], color='#b9e1fa')
plt.ylabel('Stock Price', fontsize=12)
plt.legend([stock], fontsize=12)
plt.show()

# 🔄 Prepare data for LSTM
train_df = df[['Close']]
data_unscaled = train_df.values
train_data_length = math.ceil(len(data_unscaled) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_unscaled)

sequence_length = 50
train_data_len = math.ceil(len(data_scaled) * 0.8)

train_data = data_scaled[:train_data_len]
test_data = data_scaled[train_data_len - sequence_length:]

# 📌 Function to create training/testing sequences
def partition_dataset(sequence_length, dataset):
    x, y = [], []
    for i in range(sequence_length, len(dataset)):
        x.append(dataset[i - sequence_length:i])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

# 📌 Split into training and testing sets
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# 📌 Reshape data for LSTM
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# 🔥 Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Define EarlyStopping before fitting
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Compile the model BEFORE training
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("\n⏳ Training LSTM Model...")
history = model.fit(x_train, y_train, batch_size=16, epochs=50, callbacks=[early_stop], verbose=1)
print("\n✅ Training Complete! Proceeding to Predictions...")

# 📌 Predict on test set
y_pred_scaled = model.predict(x_test)

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Check for NaN values
if np.isnan(y_pred_scaled).any():
    print("⚠️ Warning: NaN values detected in predictions! Re-training the model...")
    history = model.fit(x_train, y_train, batch_size=16, epochs=50, callbacks=[early_stop], verbose=1)
    y_pred_scaled = model.predict(x_test)  # Retry prediction

if np.isnan(y_pred_scaled).any():
    print("❌ Model still producing NaN predictions. Please check dataset and hyperparameters!")
    exit()


# Convert predictions back to original scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# 📌 Error metrics
mae = mean_absolute_error(y_test_unscaled, y_pred)
mape = np.mean(np.abs((y_test_unscaled - y_pred) / y_test_unscaled)) * 100
mdape = np.median(np.abs((y_test_unscaled - y_pred) / y_test_unscaled)) * 100

print(f'\n📊 Model Performance:')
print(f'🔹 Mean Absolute Error (MAE): {mae:.2f}')
print(f'🔹 Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
print(f'🔹 Median Absolute Percentage Error (MDAPE): {mdape:.2f}%')

# 📊 Visualize predictions
train = train_df[:train_data_length]
valid = train_df[train_data_length:].copy()
valid['Predictions'] = y_pred
valid['Difference'] = valid['Predictions'] - valid['Close']

plt.figure(figsize=(16, 8))
plt.title("Predictions vs Actual Prices", fontsize=20)
plt.plot(train['Close'], color="#039dfc", linewidth=1.0)
plt.plot(valid['Predictions'], color="#E91D9E", linewidth=1.0)
plt.plot(valid['Close'], color="black", linewidth=1.0)
plt.legend(["Train", "Predictions", "Actual"], loc="upper left")
plt.bar(valid.index, valid['Difference'], width=0.8, color=['#2BC97A' if x >= 0 else '#C92B2B' for x in valid['Difference']])
plt.show()

# 🔮 Predict next day's closing price
last_days_scaled = scaler.transform(df[['Close']].tail(sequence_length).values)
X_test = np.array([last_days_scaled]).reshape((1, sequence_length, 1))

pred_price = scaler.inverse_transform(model.predict(X_test))[0, 0]
price_today = df['Close'].iloc[-1]
percent_change = ((pred_price - price_today) / price_today) * 100

# 💰 Convert prediction to INR
usd_to_inr = get_usd_to_inr()
pred_price_inr = pred_price * usd_to_inr
price_today_inr = price_today * usd_to_inr

# 🏆 Final Output
print("\n📈 **Stock Price Prediction in INR**")
print(f"📅 Date: {today}")
print(f"🔹 Actual Close Price: {price_today:.2f} USD ({price_today_inr:.2f} INR)")
print(f"🔮 Predicted Close Price (Next Day): {pred_price:.2f} USD ({pred_price_inr:.2f} INR)")
print(f"📊 Expected Change: {percent_change:+.2f}%")

