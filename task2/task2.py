# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, f1_score
import joblib

# Function to download historical stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess stock data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

def create_sequences(data, sequence_length, threshold=0.5):
    sequences = []
    target = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        # Convert continuous closing price to binary label based on threshold
        label = 1 if data[i + sequence_length][3] > threshold else 0
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)


# Function to build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the LSTM model
def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    predictions = model.predict(X_test)
    binary_predictions = (predictions > threshold).astype(int)
    mse = mean_squared_error(y_test, predictions)
    accuracy = accuracy_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    return mse, accuracy, recall, f1



if __name__ == "__main__":
    # Download historical stock data
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Preprocess data
    scaled_data, scaler = preprocess_data(stock_data)

    # Create sequences for LSTM model
    sequence_length = 10
    X, y = create_sequences(scaled_data, sequence_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the LSTM model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse, accuracy, recall, f1 = evaluate_model(model, X_test, y_test, threshold=0.5)
    print(f"Mean Squared Error on Test Data: {mse}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Save the model and scaler
    joblib.dump(scaler, 'task2/scaler.pkl')
    model.save('task2/stock_prediction_model.h5')
