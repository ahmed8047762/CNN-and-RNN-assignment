# Image and Time Series Data Analysis

## Overview

This technical report encompasses three distinct tasks: Skin Cancer Classification using CNN, Stock Price Prediction using LSTM, and Facial Expression Recognition using CNN. The project demonstrates the versatility of machine learning techniques applied to medical image analysis, financial time series data, and facial expression recognition.

## Tasks Overview

### Task 1: Skin Cancer Classification using CNN

#### Approach

- **Preprocessing:** Resized and normalized skin lesion images for consistent input.
- **Model Architecture:** Designed a CNN for feature extraction and cancer type classification.
- **Training Process:** Trained the model using categorical cross-entropy loss and Adam optimizer, monitoring training progress to prevent overfitting.
- **Evaluation Metrics:** Assessed model performance using accuracy, precision, recall, and F1-score on the test dataset.

### Task 2: Stock Price Prediction using LSTM

#### Approach

- **Data Preprocessing:** Scaled historical stock data using Min-Max scaling.
- **Model Architecture:** Constructed an LSTM model to capture temporal patterns in stock data.
- **Training Process:** Trained the LSTM model on a subset of the data, evaluating performance using Mean Squared Error (MSE).

### Task 3: Facial Expression Recognition using CNN

#### Approach

- **Image Augmentation:** Employed techniques to increase the diversity of the training dataset.
- **Model Architecture:** Designed a CNN to extract features from facial expressions and classify emotions.
- **Training Process:** Trained the model using augmented images, evaluating performance on the validation dataset.

## Conclusion

The completion of tasks, spanning skin cancer classification, stock price prediction, and facial expression recognition, highlights the adaptability of machine learning in various domains. Convolutional Neural Networks (CNNs) proved effective in medical image analysis, LSTM networks captured temporal patterns in financial data, and CNNs demonstrated the ability to discern complex emotional cues. These results underscore the power of machine learning in addressing real-world challenges.
