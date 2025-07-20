import pandas as pd
import numpy as np
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from feature_engineering.feature_selector import auto_select_features
from model.catboost_model import train_model
from model.catboost_model import FEATURES_PATH
import joblib
import os

# منطق تستی و تضمینی برای فعال شدن buy/sell
def make_label(candles, news_df=None, threshold=0.001, future_steps=6, past_steps=15):
    closes = candles['close'].values
    highs = candles['high'].values
    lows = candles['low'].values
    labels = []
    for i in range(past_steps, len(closes) - future_steps):
        current = closes[i]
        future_high = highs[i+1:i+future_steps+1].max()
        future_low = lows[i+1:i+future_steps+1].min()
        buy_cond = (future_high - current) / current > threshold
        sell_cond = (current - future_low) / current > threshold
        if buy_cond and not sell_cond:
            labels.append(2)
        elif sell_cond and not buy_cond:
            labels.append(0)
        else:
            labels.append(1)
    candles = candles.iloc[past_steps:-(future_steps)].copy()
    candles['label'] = labels
    return candles

all_features = []
all_labels = []
all_cols = set()

for symbol in SYMBOLS:
    candles = get_latest_candles(symbol, limit=None)
    news = get_latest_news(symbol, hours=None)
    if candles is None or candles.empty:
        print(f"[{symbol}] Candles is empty!")
        continue
    candles = make_label(candles, news, threshold=0.001)
    label_counts = candles['label'].value_counts()
    print(f"Label distribution for {symbol}\n{label_counts}")

    # اگر فقط یک کلاس بود، از ادامه آموزش صرف‌نظر کن
    if label_counts.nunique() == 1:
        print(f"[{symbol}] Only one class present in the labels, skipping training for this symbol.")
        continue

    if not news.empty:
        news['published_at'] = pd.to_datetime(news['published_at'])

    for i in range(len(candles)):
        candle_slice = candles.iloc[max(0, i-99):i+1]
        if len(candle_slice) < 20:
            continue
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

print(f"Shape of X: {X.shape}, y: {y.shape}, label dist: {np.unique(y, return_counts=True)}")
print(f"Training samples: {len(X)}")
print("All feature columns:", X.columns.tolist())
print("Sample row:", X.iloc[0].to_dict())

tech_cols = [c for c in X.columns if not c.startswith("news")]
fund_cols = [c for c in X.columns if c.startswith("news")]
print(f"Number of technical features: {len(tech_cols)}")
print(f"Number of fundamental features: {len(fund_cols)}")

import glob
for f in glob.glob("model/catboost_tradebot_pro.pkl") + glob.glob("model/catboost_features.pkl"):
    try:
        os.remove(f)
        print(f"Deleted old model file: {f}")
    except Exception as e:
        print(f"Failed to delete {f}: {e}")

model = train_model(X, y)
if model is not None:
    feature_names = joblib.load(FEATURES_PATH)
    auto_select_features(model, feature_names, top_n=30)
else:
    print("Training failed due to lack of label diversity.")