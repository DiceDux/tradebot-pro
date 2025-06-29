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
        for i in range(len(candles)-12):
            candle_slice = candles.iloc[max(0, i-99):i+1]  # 100 کندل تا لحظه i
            news_slice = news[news['published_at'] <= candles.iloc[i]['timestamp']]
            features = build_features(candle_slice, news_slice, symbol)
            all_features.append(features.iloc[0])
            all_labels.append(candles.iloc[i]['label'])

    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)
    print("Training samples:", len(X))
    model = CatBoostClassifier(verbose=100, iterations=700, learning_rate=0.05, depth=7)
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)

if __name__ == "__main__":
    main()