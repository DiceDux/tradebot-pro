import pymysql
import pandas as pd
from utils.config import DB_CONFIG
from data.fetch_online import fetch_candles_binance

def get_latest_candles(symbol, limit=200):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        if limit is None:
            query = "SELECT * FROM candles WHERE symbol=%s ORDER BY timestamp DESC"
            df = pd.read_sql(query, conn, params=[symbol])
        else:
            query = "SELECT * FROM candles WHERE symbol=%s ORDER BY timestamp DESC LIMIT %s"
            df = pd.read_sql(query, conn, params=[symbol, limit])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    finally:
        conn.close()

def save_candles_to_db(candles):
    if isinstance(candles, pd.DataFrame):
        candles = candles.to_dict(orient="records")
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
    finally:
        conn.close()

def keep_last_200_candles(symbol):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM candles WHERE symbol=%s AND timestamp NOT IN (
                    SELECT timestamp FROM (
                        SELECT timestamp FROM candles WHERE symbol=%s ORDER BY timestamp DESC LIMIT 200
                    ) t
                )
            """, (symbol, symbol))
            conn.commit()
    finally:
        conn.close()
