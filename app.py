# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:56:34 2024

@author: Admin
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pickle

#open model in read binary mode
load = open('model.pkl','rb')
model =pickle.load(load)

# Streamlit app title
st.title('Stock Price Prediction Dashboard')

# User input for stock symbol, forex symbol, and date range
stock_symbol = st.text_input('Enter Stock Symbol :', 'AAPL')
forex_symbol = st.text_input('Enter Forex Symbol :', 'USDINR=X')
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

# Fetch stock data
st.write(f'Fetching data for {stock_symbol} and {forex_symbol}...')
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
forex_data = yf.download(forex_symbol, start=start_date, end=end_date)

# Create new features like moving averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

# Drop rows with missing values
stock_data = stock_data.dropna()

# Merge stock and forex data
merged_data = stock_data.merge(forex_data[['Close']], left_index=True, right_index=True, suffixes=('', '_forex'))
merged_data['Forex'] = merged_data['Close_forex']

# Features for the model
features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200', 'Forex']
target = 'Close'

# Split the data into X and y
X = merged_data[features]
y = merged_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
st.write("Predicted Stock Prices:", y_pred)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')

# Plot the actual vs predicted prices
st.subheader('Actual vs Predicted Stock Prices')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test.values, label='Actual')
ax.plot(y_pred, label='Predicted')
ax.legend()
st.pyplot(fig)
