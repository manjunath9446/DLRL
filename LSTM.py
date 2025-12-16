# -*- coding: utf-8 -*-
"""
Refactored LSTM for Stock Price Prediction
Created on Mon Dec 15 2025

Description:
    Implementation of LSTM using the Keras Functional API.
    Simulates Stock Price prediction (Time Series).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras import Model, Input # Changed: Functional API imports
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Data Generation (Stock Price Theme)
# Note: Replaced CSV loading with synthetic data generation so the code runs standalone.
def generate_stock_data(n_points=144):
    x = np.linspace(0, 50, n_points)
    # Sine wave + Linear Trend + Random Noise
    trend = x * 0.5
    seasonality = np.sin(x) * 10
    noise = np.random.normal(0, 2, n_points)
    y = 50 + trend + seasonality + noise
    return y

# Generate the data
dataset_values = generate_stock_data(200)
dataset_values = dataset_values.reshape(-1, 1).astype("float32")

# Visualization of raw data
plt.figure(figsize=(10,6))
plt.plot(dataset_values)
plt.xlabel("Days")
plt.ylabel("Stock Price ($)")
plt.title("Synthetic Stock Price History")
plt.show()

# 2. Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset_values)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# Helper function to create sliding window dataset
# Changed: Refactored the loop logic into a reusable function
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 10 # Same as 'time_stamp' in previous code
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
# Logic maintained from original: (Samples, 1, Look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# 3. Model Build (Functional API)

def build_functional_lstm(input_shape):
    # Changed: Explicit Input Layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Changed: LSTM Layer defined functionally
    # Note: '10' is the number of units (neurons)
    x = LSTM(10, activation='tanh', name='lstm_layer')(inputs)
    
    # Changed: Dense Output Layer defined functionally
    outputs = Dense(1, name='output_layer')(x)
    
    # Changed: Model instantiation
    model = Model(inputs=inputs, outputs=outputs, name="Stock_Predictor_LSTM")
    return model

# Create the model
# input_shape is (1, 10) based on the reshape logic above
model = build_functional_lstm(input_shape=(1, look_back))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train
history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)

# 4. Predictions & Evaluation
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY_orig = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_orig = scaler.inverse_transform([testY])

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY_orig[0], trainPredict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = math.sqrt(mean_squared_error(testY_orig[0], testPredict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')


# 5. Plotting Results

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# Logic: Start after training data + look_back offset
# The index calculation ensures alignment with the original dataset
start_idx = len(trainPredict) + (look_back * 2) + 1
end_idx = len(dataset) - 1

# Safety check for indices to prevent crash on small datasets
if start_idx < len(dataset):
    # Adjust slice size to match testPredict length
    slice_len = min(len(testPredict), len(dataset) - start_idx)
    testPredictPlot[start_idx : start_idx + slice_len, :] = testPredict[:slice_len]

# Final Plot
plt.figure(figsize=(10,6))
plt.plot(scaler.inverse_transform(dataset), label="Actual Stock Price", color='gray', alpha=0.5)
plt.plot(trainPredictPlot, label="Train Prediction", color='blue')
plt.plot(testPredictPlot, label="Test Prediction", color='red')
plt.title("Stock Price Prediction: LSTM (Functional API)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.show()