import requests
from utils.config import DB_CONFIG
import pymysql
import datetime

def fetch_candles_binance(symbol, interval="4h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        candles = []
        for row in data:
            candles.append({
                "symbol": symbol,
                "timestamp": int(row[0] // 1000),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })
        return candles
    except Exception as e:
        print(f"[fetch_candles_binance] Error: {e}")
        return []

def fetch_news_online(symbol, limit=10):
    # این تابع باید با API خبری دلخواه پر شود! (نمونه فرضی)
    url = f"https://api.yournewsprovider.com/crypto_news?symbol={symbol}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()  # فرض بر این که دیتا در قالب لیست است
        news = []
        for n in data:
            news.append({
                "symbol": symbol,
                "published_at": n["published_at"],
                "sentiment_score": n.get("sentiment_score", 0),
                "content": n.get("content", ""),
                "title": n.get("title", "")
            })
        return news
    except Exception as e:
        print(f"[fetch_news_online] Error: {e}")
        return []
