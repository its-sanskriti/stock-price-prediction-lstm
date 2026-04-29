# Stock Price Prediction using LSTM

A deep learning project that predicts long-term stock prices using an LSTM 
neural network trained on 15 years of TCS historical data.

🚀 Live Demo: https://huggingface.co/spaces/Sanskritiii/lstm-stock-predictor

## Results
- RMSE: ₹84 on stock trading ₹2500–₹4300
- MAE: 2% average error
- R² Score: 0.9563

## What makes this project solid
- No data leakage — scaler fitted only on training data, 
  transform applied separately on test
- Proper train/test split before sliding window creation
- Stacked LSTM architecture with Dropout regularization
- Fully deployed as interactive web app on HuggingFace Spaces

## Tech Stack
Python, TensorFlow/Keras, Streamlit, yfinance, 
scikit-learn, Pandas, NumPy, Matplotlib

## Model Architecture
- 4 stacked LSTM layers (50, 60, 80 units) with Dropout
- 100-day lookback window
- Trained on 70% of data from 2010–2026
- MinMaxScaler normalization (fit on train only)

## Key Steps
1. Fetch TCS stock data via yfinance API
2. Preprocess — scale, split, create sliding windows
3. Train stacked LSTM with dropout regularization
4. Evaluate with RMSE, MAE, R² score
5. Deploy interactive app on HuggingFace Spaces

## Run Locally
pip install -r requirements.txt
streamlit run app.py
