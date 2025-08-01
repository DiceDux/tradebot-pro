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
    """
    فقط بررسی تعداد کندل‌ها بدون حذف هیچ داده‌ای
    """
    try:
        from utils.config import DB_CONFIG
        import pymysql
        
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles WHERE symbol = %s", (symbol,))
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Symbol {symbol} has {count} historical candles in database")
        return count
        
    except Exception as e:
        print(f"Error in keep_last_200_candles: {e}")
        return 0