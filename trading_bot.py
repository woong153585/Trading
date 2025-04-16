#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import json
import numpy as np
from binance import Client, BinanceSocketManager
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model
from tensorflow.keras.layers import LSTM, Dense
import requests
import logging
import pandas as pd

TELEGRAM_TOKEN = '7907965642:AAGcnrc8iKgY7cHYcwEFgVEuY5iUQp7ySto'
CHAT_ID = '@wooooooong'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram 알림 오류: {str(e)}")
        return False
        
logging.basicConfig(level=logging.INFO)
logging.info("Trading bot is starting...")

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')

# Binance 클라이언트 초기화 - 이 부분이 누락되었습니다
client = Client(api_key, api_secret)

# 글로벌 설정
SYMBOL = 'XRPUSDT'
QUANTITY = 1
MODEL_PATH = 'xrp_lstm.h5'
SEQ_LENGTH = 60  # LSTM 입력 시퀀스 길이

price_data = pd.DataFrame(columns=['timestamp', 'price'])
current_position = 0

# LSTM 모델 초기화
# trading_bot.ipynb 파일의 initialize_model 함수 수정
def initialize_model():
    if os.path.exists(MODEL_PATH):
        # MSE 손실 함수 직접 정의
        import tensorflow as tf
        def mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
            
        return load_model(MODEL_PATH, custom_objects={'mse': mse})
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


model = initialize_model()
scaler = MinMaxScaler(feature_range=(0, 1))

# 데이터 전처리
def preprocess_data(data):
    if len(data) < SEQ_LENGTH:
        print(f"경고: 충분한 데이터가 없습니다. 현재 {len(data)}개, 필요한 개수: {SEQ_LENGTH}")
        return np.array([]), None
    
    try:
        # 데이터 확인 로그 추가
        print(f"전처리 데이터 크기: {len(data)}")
        
        # 데이터 형태 변환
        data_values = data.values.reshape(-1, 1)
        scaled = scaler.fit_transform(data_values)
        
        X = []
        for i in range(SEQ_LENGTH, len(scaled)):
            X.append(scaled[i-SEQ_LENGTH:i, 0])
        
        # 결과 확인
        if len(X) == 0:
            print("경고: 시퀀스를 생성할 수 없습니다")
            return np.array([]), None
            
        return np.array(X), scaler
    except Exception as e:
        print(f"데이터 전처리 오류: {str(e)}")
        return np.array([]), None

# 실시간 예측
def predict_price():
    global price_data
    
    try:
        if len(price_data) < SEQ_LENGTH:
            print(f"가격 데이터 부족: {len(price_data)}/{SEQ_LENGTH}")
            return None
            
        # 최신 SEQ_LENGTH개 데이터만 사용
        data_window = price_data['price'].iloc[-SEQ_LENGTH:].copy()
        print(f"예측용 데이터 윈도우 크기: {len(data_window)}")
            
        # 임시 방편: 직접 올바른 크기의 X 생성
        window_values = data_window.values.reshape(-1, 1)
        scaled_window = scaler.fit_transform(window_values)
            
        # 올바른 차원으로 직접 변환
        X = np.array([scaled_window.flatten()])
        X = X.reshape((1, SEQ_LENGTH, 1))
            
        prediction = model.predict(X, verbose=0)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        print(f"예측 가격: {predicted_price}")
        return predicted_price
    except Exception as e:
        print(f"예측 오류: {str(e)}")
        return None

# 거래 전략
def trading_strategy(current_price):
    predicted_price = predict_price()
    
    if predicted_price is None:
        print("예측 불가: 데이터 부족. HOLD 상태 유지.")
        return 'HOLD'
    
    print(f"현재 가격: {current_price}, 예측 가격: {predicted_price}")
    
    if predicted_price > current_price * 1.005:  # 0.5% 상승 예측
        return 'BUY'
    elif predicted_price < current_price * 0.995:  # 0.5% 하락 예측
        return 'SELL'
    return 'HOLD'

# 거래 실행
def execute_order(client, decision):
    global current_position
    try:
        if decision == 'BUY' and current_position < 5:
            order = client.create_order(
                symbol=SYMBOL,
                side='BUY',
                type='MARKET',
                quantity=QUANTITY
            )
            current_position += QUANTITY
            log_trade('BUY', float(order['fills'][0]['price']), QUANTITY)
        elif decision == 'SELL' and current_position > 0:
            order = client.create_order(
                symbol=SYMBOL,
                side='SELL',
                type='MARKET',
                quantity=QUANTITY
            )
            current_position -= QUANTITY
            log_trade('SELL', float(order['fills'][0]['price']), QUANTITY)
    except Exception as e:
        print(f"주문 실행 오류: {str(e)}")

# 거래 기록
def log_trade(action, price, quantity):
    entry = {
        'timestamp': int(time.time() * 1000),
        'action': action,
        'price': price,
        'quantity': quantity
    }
    with open('trade_log.json', 'a') as f:
        json.dump(entry, f)
        f.write('\n')

# 메인 함수
def main():
    client = Client(api_key, api_secret)
    global price_data
    
    # 초기 과거 데이터를 넉넉하게 로드 (최소 120개)
    print("과거 데이터 로드 중...")
    klines = client.get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1MINUTE, "120 minutes ago UTC")
    
    # 데이터 확인
    print(f"가져온 과거 데이터: {len(klines)}개")
    
    # 충분한 데이터가 있는지 확인
    if len(klines) < SEQ_LENGTH:
        print(f"충분한 과거 데이터를 가져올 수 없습니다. 더 많은 데이터가 필요합니다.")
        return
    
    # 데이터프레임 초기화 및 데이터 로드
    price_data = pd.DataFrame(columns=['timestamp', 'price'])
    for k in klines:
        price_data.loc[len(price_data)] = [k[0], float(k[4])]
    
    print(f"초기 데이터 {len(price_data)}개 로드 완료")
    
    # 데이터 충분한지 확인
    if len(price_data) < SEQ_LENGTH:
        print("데이터가 부족합니다. 프로그램을 종료합니다.")
        return
    
    # 폴링 방식으로 거래 실행
    while True:
        try:
            # 1분마다 최신 가격 데이터 가져오기
            latest_kline = client.get_klines(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)
            price = float(latest_kline[0][4])  # 종가
            timestamp = latest_kline[0][0]     # 타임스탬프
            
            # 데이터프레임에 추가
            price_data.loc[len(price_data)] = [timestamp, price]
            print(f"현재 가격: {price}, 누적 데이터: {len(price_data)}개")
            
            # 데이터가 충분할 때만 거래 결정
            if len(price_data) >= SEQ_LENGTH + 10:  # 여유있게 10개 더 확보
                decision = trading_strategy(price)
                print(f"거래 결정: {decision}")
                if decision != 'HOLD':
                    execute_order(client, decision)
            else:
                print(f"데이터 수집 중... {len(price_data)}/{SEQ_LENGTH + 10}개")
            
            # 대기 시간
            time.sleep(60)
                
        except Exception as e:
            print(f"메인 루프 오류: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()
 

