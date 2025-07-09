import pandas as pd
import numpy as np
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import train_model
import os

def make_label(candles, threshold=0.03, future_steps=12):
    closes = candles['close'].values
    labels = []
    for i in range(len(closes) - future_steps):
        future_window = closes[i+1:i+1+future_steps]
        current = closes[i]
        max_future = future_window.max()
        min_future = future_window.min()
        if (max_future - current) / current > threshold:
            labels.append(2)  # Buy
        elif (current - min_future) / current > threshold:
            labels.append(0)  # Sell
        else:
            labels.append(1)  # Hold
    # آخرین future_steps ردیف لیبل ندارد، حذفشان کن
    candles = candles.iloc[:-future_steps].copy()
    candles['label'] = labels
    return candles

all_features = []
all_labels = []
all_cols = set()

for symbol in SYMBOLS:
    candles = get_latest_candles(symbol, limit=3000)
    news = get_latest_news(symbol, hours=365*24)
    if candles is None or candles.empty:
        print(f"[{symbol}] Candles is empty!")
        continue
    candles = make_label(candles)
    print(f"Label distribution for {symbol}\n{candles['label'].value_counts()}")

    if not news.empty:
        news['published_at'] = pd.to_datetime(news['published_at'])

    for i in range(len(candles)):
        candle_slice = candles.iloc[max(0, i-99):i+1]
        candle_time = pd.to_datetime(candles.iloc[i]['timestamp'], unit='s')
        news_slice = news[news['published_at'] <= candle_time] if not news.empty else pd.DataFrame()
        if i % 200 == 0:
            print(f"[{symbol}][{i}] News count in window: {len(news_slice)}")
        features = build_features(candle_slice, news_slice, symbol)
        features_clean = {}
        for col in features.columns:
            val = features[col].values[0] if hasattr(features[col], "values") else features[col]
            if isinstance(val, (pd.Series, np.ndarray)):
                val = float(val[0]) if len(val) > 0 else 0.0
            try:
                val = float(val)
            except Exception:
                val = 0.0
            features_clean[col] = val

        all_cols.update(features_clean.keys())
        all_features.append(features_clean)
        all_labels.append(candles.iloc[i]['label'])

        if i % 250 == 0 or i == len(candles) - 1:
            techs = {k: v for k, v in features_clean.items() if not k.startswith("news")}
            funds = {k: v for k, v in features_clean.items() if k.startswith("news")}
            print(f"[{symbol}][{i}] TECH: " + " | ".join([f"{k}={v:.2f}" for k, v in list(techs.items())[:4]]) +
                  " | ... | FUND: " + " | ".join([f"{k}={v:.2f}" for k, v in list(funds.items())[:4]]) + " | ...")

use_cols = sorted(list(all_cols))
X = pd.DataFrame([{col: f.get(col, 0.0) for col in use_cols} for f in all_features])
y = np.array(all_labels)

# لاگ کلیدی برای بررسی سایز داده و توزیع لیبل
print(f"Shape of X: {X.shape}, y: {y.shape}, label dist: {np.unique(y, return_counts=True)}")
print(f"Training samples: {len(X)}")
print("All feature columns:", X.columns.tolist())
print("Sample row:", X.iloc[0].to_dict())

tech_cols = [c for c in X.columns if not c.startswith("news")]
fund_cols = [c for c in X.columns if c.startswith("news")]
print(f"Number of technical features: {len(tech_cols)}")
print(f"Number of fundamental features: {len(fund_cols)}")

# پاک کردن مدل قبلی (در صورت وجود)
import glob
for f in glob.glob("model/catboost_tradebot_pro.pkl") + glob.glob("model/catboost_features.pkl"):
    try:
        os.remove(f)
        print(f"Deleted old model file: {f}")
    except Exception as e:
        print(f"Failed to delete {f}: {e}")

model = train_model(X, y)
