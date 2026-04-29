import numpy as np
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = yf.download('TCS.NS', start='2010-01-01', auto_adjust=True)
close = df['Close'].values.reshape(-1, 1)

split = int(len(close) * 0.70)
train_close = close[:split]

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_close)

X_train, y_train = [], []
for i in range(100, len(train_scaled)):
    X_train.append(train_scaled[i-100:i])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(60, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

model.save('tcs_lstm_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Done — model and scaler saved locally')