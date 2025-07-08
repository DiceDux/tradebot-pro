import pandas as pd
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import train_model
import numpy as np

def make_label(candles, threshold=0.015):
    labels = []
    for i in range(len(candles) - 12):
        future = candles.iloc[i+1:i+13]
        price_now = candles.iloc[i]['close']
        price_max = future['high'].max()
        price_min = future['low'].min()
        if price_max >= price_now * (1 + threshold):
            labels.append(2)  # buy
        elif price_min <= price_now * (1 - threshold):
            labels.append(0)  # sell
        else:
            labels.append(1)  # hold
    candles = candles.iloc[:-12].copy()
    candles['label'] = labels
    return candles

all_features = []
all_labels = []
all_cols = set()

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

    for i in range(len(candles)):
        candle_slice = candles.iloc[max(0, i-99):i+1]
        candle_time = pd.to_datetime(candles.iloc[i]['timestamp'], unit='s')
        news_slice = news[news['published_at'] <= candle_time]
        features = build_features(candle_slice, news_slice, symbol)
        # جمع همه فیچرها (ستون‌ها را جمع کن)
        all_cols.update(features.columns)
        all_features.append(features)
        all_labels.append(candles.iloc[i]['label'])

# لیست کامل ستون‌ها برای همه فیچرها
use_cols = sorted(list(all_cols))

# همه فیچرها را به DataFrame تبدیل کن و ستون‌های ناقص را با 0 پر کن
X = pd.DataFrame([{col: f[col] if col in f else 0 for col in use_cols} for f in all_features])
y = np.array(all_labels)
print(f"Training samples: {len(X)}")
print("All feature columns:", X.columns.tolist())
print("Sample row:", X.iloc[0].to_dict())

# (نمایش خلاصه از فیچرهای تکنیکال و فاندامنتال)
tech_cols = [c for c in X.columns if not c.startswith("news")]
fund_cols = [c for c in X.columns if c.startswith("news")]
print(f"Number of technical features: {len(tech_cols)}")
print(f"Number of fundamental features: {len(fund_cols)}")

model = train_model(X, y)
