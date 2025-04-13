#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Binance 클라이언트 초기화
client = Client(api_key, api_secret)

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')

# Binance 클라이언트 초기화 (이 부분이 누락되었습니다)
client = Client(api_key, api_secret)

# 모델 파라미터 설정
SYMBOL = 'XRPUSDT'
MODEL_PATH = 'xrp_lstm.h5'
SEQ_LENGTH = 60

# LSTM 모델 초기화
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 과거 데이터 1000개 이상 수집
klines = client.get_historical_klines(SYMBOL, '1m', '1000 minutes ago UTC')
prices = [float(k[4]) for k in klines]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1,1))

X_train, y_train = [], []
for i in range(SEQ_LENGTH, len(scaled_prices)):
    X_train.append(scaled_prices[i-SEQ_LENGTH:i, 0])
    y_train.append(scaled_prices[i, 0])
    
X_train = np.array(X_train).reshape(-1, SEQ_LENGTH, 1)
y_train = np.array(y_train)

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32)
save_model(model, MODEL_PATH)

print(f"모델이 성공적으로 학습되었으며 {MODEL_PATH}에 저장되었습니다.")
print(f"학습에 사용된 데이터 포인트: {len(prices)}개")
print(f"학습 샘플 수: {len(X_train)}개")

