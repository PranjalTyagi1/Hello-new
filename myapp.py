import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Data
def load_data():
    uploaded_file = st.file_uploader("Upload Tesla Stock CSV", type="csv")
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Simple model to predict Close price using the previous day's Open, High, Low prices
def train_model(data):
    data['Previous_Open'] = data['Open'].shift(1)
    data['Previous_High'] = data['High'].shift(1)
    data['Previous_Low'] = data['Low'].shift(1)
    data.dropna(inplace=True)

    X = data[['Previous_Open', 'Previous_High', 'Previous_Low']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    return model, mse

def predict_next_day(model, data):
    latest_data = data.iloc[-1]
    X_new = [[latest_data['Open'], latest_data['High'], latest_data['Low']]]
    return model.predict(X_new)[0]

st.title("Tesla Stock Price Predictor")

data = load_data()

if data is not None:
    st.write(data.head())

    model, mse = train_model(data)

    st.write(f"Model Mean Squared Error: {mse:.2f}")

    days_to_predict = st.slider('Number of days to predict', 1, 30, 1)
    predictions = []

    for _ in range(days_to_predict):
        prediction = predict_next_day(model, data)
        predictions.append(prediction)
        new_row = {'Open': prediction, 'High': prediction, 'Low': prediction, 'Close': prediction}
        data = data.append(new_row, ignore_index=True)  # Fixed this line
    
    st.write(f"Predictions for next {days_to_predict} days:")
    st.write(predictions)

st.write("Disclaimer: Predictions are for informational purposes only and not financial advice.")
