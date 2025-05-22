# stock-price-prediction-lstm
 Note: This notebook was adapted from an open-source LSTM stock prediction tutorial.  
> I fixed the date labels on the x-axis and added detailed cell-by-cell explanations to reflect my understanding.
>  Fixed NaN Issue During Training
> Cleaned Up the Model Architecture

ong-Term Stock Price Prediction Using LSTM
This project uses a Long Short-Term Memory (LSTM) neural network to predict the long-term stock price of a company based on its historical stock data.

🔧 Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
TensorFlow/Keras
Jupyter Notebook
Yahoo Finance API

📂 Dataset
Collected using yfinance for TCS stock.
Includes:
Date
Open, High, Low, Close,Volume
Data stored in CSV format.

📊 Features
100-day and 200-day Moving Averages
Normalized closing prices
Train-Test split: 70-30
Model: LSTM Neural Network
Evaluation: R² Score, Plot of Actual vs Predicted

📌 Key Steps
Load and visualize the stock data
Calculate moving averages
Preprocess and normalize data
Build and train LSTM model
Predict and evaluate

🧠 Outcome
Predicts the trend of stock prices
Demonstrates how deep learning can be applied to financial time-series data

