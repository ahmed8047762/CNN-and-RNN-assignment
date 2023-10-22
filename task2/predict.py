# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import joblib
import json

# Function to download historical stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess stock data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

# Flask application for prediction
app = Flask(__name__)

# Load pre-trained model and scaler
scaler = joblib.load('task2/scaler.pkl')
model = load_model('task2/stock_prediction_model.h5')

@app.route('/predict/<string:data>', methods=['GET'])
def predict(data):
    # Parse the input data from the URL string (data) and convert it to a numpy array
    input_data = np.array(json.loads(data))
    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
    scaled_input_data = scaler.transform(input_data[0])
    prediction = model.predict(scaled_input_data.reshape(1, scaled_input_data.shape[0], scaled_input_data.shape[1]))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    # Run Flask application
    app.run(debug=True, port=5000)
