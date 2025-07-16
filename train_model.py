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

def make_label(candles, news_df=None, threshold=0.03, future_steps=12, past_steps=12):
    closes = candles['close'].values
    volumes = candles['volume'].values
    labels = []
    for i in range(past_steps, len(closes) - future_steps):
        future_window = closes[i+1:i+1+future_steps]
        current = closes[i]
        max_future = future_window.max()
        min_future = future_window.min()
        future_vol = volumes[i+1:i+1+future_steps].mean()
        current_vol = volumes[i]
        past_window = closes[i-past_steps:i]
        past_max = past_window.max()
        past_min = past_window.min()
        # شوک خبری
        shock = 0
        if news_df is not None and not news_df.empty:
            t0 = candles.iloc[i]['timestamp']
            if 'ts' not in news_df.columns and 'published_at' in news_df.columns:
                news_df = news_df.copy()
                news_df['ts'] = pd.to_datetime(news_df['published_at']).astype(int) // 10**9
            shock_news = news_df[(news_df['ts'] <= t0) & (news_df['ts'] > t0 - 3600*6)]
            if not shock_news.empty and 'sentiment_score' in shock_news:
                shock = shock_news['sentiment_score'].astype(float).abs().mean()
        # منطق پیشرفته: ترکیب روند، حجم، شوک خبری، رفتار گذشته و متوسط احساسات کل اخبار
        fund_score = 0
        if news_df is not None and not news_df.empty and 'sentiment_score' in news_df:
            fund_score = news_df['sentiment_score'].astype(float).mean()
        buy_cond = (
            (max_future - current) / current > threshold and
            current > past_max and
            (future_vol > current_vol * 1.05 or shock > 0.2 or fund_score > 0.1)
        )
        sell_cond = (
            (current - min_future) / current > threshold and
            current < past_min and
            (future_vol > current_vol * 1.05 or shock > 0.2 or fund_score < -0.1)
        )
        if buy_cond:
            labels.append(2)  # Buy
        elif sell_cond:
            labels.append(0)  # Sell
        else:
            labels.append(1)  # Hold
    candles = candles.iloc[past_steps:-(future_steps)].copy()
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
        if len(candle_slice) < 20:  # یا هر عددی که برای محاسبه اندیکاتورهای پیچیده لازم است (مثلاً 20 برای ADX و BB و ...)
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
feature_names = joblib.load(FEATURES_PATH)
auto_select_features(model, feature_names, top_n=30)  # هر تعدادی که خواستی
