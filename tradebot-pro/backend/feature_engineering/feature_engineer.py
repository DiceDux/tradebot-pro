import pandas as pd
import numpy as np

def build_features(candles_df, news_df, symbol):
    # --- مهندسی فیچر تکنیکال ---
    # مثال: EMA, RSI, ATR, MACD, Bollinger, OBV, VWAP و ...
    features = {}
    if candles_df is not None and not candles_df.empty:
        # EMA مثال
        features['ema20'] = candles_df['close'].ewm(span=20).mean().values[-1]
        features['ema50'] = candles_df['close'].ewm(span=50).mean().values[-1]
        # RSI مثال
        delta = candles_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        features['rsi'] = 100 - (100 / (1 + rs.values[-1]))
        # ... سایر فیچرهای تکنیکال به همین شکل اضافه شود

    # --- مهندسی فیچر فاندامنتال و اخبار ---
    if news_df is not None and not news_df.empty:
        # میانگین sentiment_score خبرها
        features['news_sentiment_mean'] = news_df['sentiment_score'].mean() if 'sentiment_score' in news_df else 0.0
        features['news_count'] = len(news_df)
    else:
        features['news_sentiment_mean'] = 0.0
        features['news_count'] = 0

    # --- جمع‌بندی ---
    features_df = pd.DataFrame([features])
    return features_df