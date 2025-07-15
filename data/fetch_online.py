import requests
from utils.config import DB_CONFIG
import pymysql

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

def save_candles_to_db(candles):
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

# candles = fetch_candles_binance("BTCUSDT")
# save_candles_to_db(candles)
