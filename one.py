import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Data
@st.cache
def load_data():
    data = pd.read_csv(r'C:\Users\anvik\Desktop\tsla.csv')
    return data

data = load_data()

# Sidebar
st.sidebar.header('Parameters')

# Main
st.write("# Tesla Stock Price Predictor")
st.write("### Data Overview")
st.write(data.head())

# Splitting the data
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Displaying predictions
st.write("### Predictions")
st.write(pd.DataFrame({
    'Actual Close Price': y_test,
    'Predicted Close Price': y_pred
}))

# Model Evaluation
st.write("### Model Evaluation")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# Visualization
st.write("### Stock Price Chart")
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label="Actual Close Price")
plt.title("Tesla Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(plt)

# User Input for Predictions
st.write("### Get Custom Predictions")
open_val = st.number_input("Enter Open Price", min_value=0.0)
high_val = st.number_input("Enter High Price", min_value=0.0)
low_val = st.number_input("Enter Low Price", min_value=0.0)
volume_val = st.number_input("Enter Volume", min_value=0.0)
user_input = np.array([open_val, high_val, low_val, volume_val]).reshape(1, -1)
user_pred = model.predict(user_input)

st.write(f"Predicted Close Price: ${user_pred[0]:.2f}")
