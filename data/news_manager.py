import pymysql
import pandas as pd
from utils.config import DB_CONFIG

def get_latest_news(symbol, hours=800):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        if hours is None:
            query = "SELECT * FROM news WHERE symbol=%s ORDER BY published_at DESC"
            df = pd.read_sql(query, conn, params=[symbol])
        else:
            query = "SELECT * FROM news WHERE symbol=%s AND published_at >= NOW() - INTERVAL %s HOUR ORDER BY published_at DESC"
            df = pd.read_sql(query, conn, params=[symbol, hours])
        df = df.sort_values("published_at").reset_index(drop=True)
        return df
    finally:
        conn.close()

def save_news_to_db(news):
    if isinstance(news, pd.DataFrame):
        news = news.to_dict(orient="records")
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
    finally:
        conn.close()

def get_news_for_range(symbol, start_ts, end_ts):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        query = "SELECT * FROM news WHERE symbol=%s AND UNIX_TIMESTAMP(published_at) BETWEEN %s AND %s ORDER BY published_at ASC"
        df = pd.read_sql(query, conn, params=[symbol, start_ts, end_ts])
        return df
    finally:
        conn.close()