import pandas as pd
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from catboost import CatBoostClassifier
import joblib
import os

MODEL_PATH = "model/catboost_tradebot_pro.pkl"

def make_label(df, lookahead=12, threshold=0.002):
    df = df.copy()
    df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
    df['label'] = 1  # Hold
    df.loc[df['future_return'] > threshold, 'label'] = 2  # Buy
    df.loc[df['future_return'] < -threshold, 'label'] = 0  # Sell
    return df

def main():
    all_features = []
    all_labels = []
    for symbol in SYMBOLS:
        candles = get_latest_candles(symbol, limit=3000)
        news = get_latest_news(symbol, hours=365*24)
        if candles is None or candles.empty:
            continue
        candles = make_label(candles)
        print("Label distribution for", symbol)
        print(candles['label'].value_counts())

        if not news.empty:
            news['published_at'] = pd.to_datetime(news['published_at'])

        for i in range(len(candles)-12):
            candle_slice = candles.iloc[max(0, i-99):i+1]
            candle_time = pd.to_datetime(candles.iloc[i]['timestamp'], unit='s')
            news_slice = news[news['published_at'] <= candle_time]
            features = build_features(candle_slice, news_slice, symbol)
            # خلاصه فاندامنتال
            fund_keys = [k for k in features.columns if 'news' in k]
            fund_score = abs(features[fund_keys]).sum() if fund_keys else 0
            if i % 300 == 0:
                print(f"[{symbol}][{i}] fund_score={fund_score:.2f}")
            all_features.append(features.iloc[0])
            all_labels.append(candles.iloc[i]['label'])

    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)
    print("Training samples:", len(X))
    model = CatBoostClassifier(verbose=100, iterations=700, learning_rate=0.05, depth=7)
    model.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
