import pandas as pd
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import train_model
import numpy as np

def make_label(candles, threshold=0.015):
    # 0: sell, 1: hold, 2: buy
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
use_cols = None

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

        # فقط فیچرهای بازه‌ای را به مدل بده:
        cur_use_cols = [c for c in features.columns if not (
            c in ['news_count', 'news_sentiment_mean', 'news_sentiment_std', 'news_pos_count', 'news_neg_count', 'news_latest_sentiment', 'news_content_len']
        )]
        if use_cols is None:
            use_cols = cur_use_cols
        all_features.append(features[cur_use_cols].values[0])
        all_labels.append(candles.iloc[i]['label'])

X = pd.DataFrame(all_features, columns=use_cols)
y = np.array(all_labels)
print(f"Training samples: {len(X)}")
model = train_model(X, y)
print("Model trained and saved to model/catboost_tradebot_pro.pkl")

# نمایش feature importance
feature_names = X.columns if hasattr(X, 'columns') else [f"f{i}" for i in range(X.shape[1])]
importances = model.get_feature_importance()
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 15 Feature Importances:")
for idx in sorted_idx[:15]:
    print(f"{feature_names[idx]} : {importances[idx]:.4f}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.bar([feature_names[i] for i in sorted_idx[:15]], importances[sorted_idx[:15]])
    plt.xticks(rotation=45)
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
except Exception as e:
    print("Plotting feature importances failed:", e)