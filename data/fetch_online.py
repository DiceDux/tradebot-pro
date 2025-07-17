import requests
from utils.config import DB_CONFIG
import pymysql
import datetime

def fetch_candles_binance(symbol, interval="4h", limit=200):
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

def save_candles_to_db(candles):
    if not candles: return
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            for c in candles:
                query = """
                    INSERT IGNORE INTO candles (symbol, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (c["symbol"], c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[save_candles_to_db] Error: {e}")

def fetch_news_newsapi(symbol, limit=25, api_key="..."):
    # توجه: باید api_key خودت رو از https://newsapi.org بگیری و اینجا بذاری
    # نمادهای جهانی (BTCUSDT → bitcoin) باید map شوند
    symbol_map = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        # بقیه نمادها رو هم اضافه کن
    }
    q = symbol_map.get(symbol, symbol)
    url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize={limit}&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        news = []
        for n in data.get("articles", []):
            news.append({
                "symbol": symbol,
                "published_at": n["publishedAt"],  # ISO format
                "sentiment_score": 0,  # اگر تحلیل احساسات داری اینجا بگذار، وگرنه 0
                "content": n.get("content", ""),
                "title": n.get("title", "")
            })
        return news
    except Exception as e:
        print(f"[fetch_news_newsapi] Error: {e}")
        return []

def save_news_to_db(news):
    if not news: return
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            for n in news:
                query = """
                    INSERT IGNORE INTO news (symbol, published_at, sentiment_score, content, title)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (n["symbol"], n["published_at"], n.get("sentiment_score",0), n.get("content",""), n.get("title","")))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[save_news_to_db] Error: {e}")
