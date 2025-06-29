import pandas as pd
from sqlalchemy import create_engine
from utils.config import DB_CONFIG, NEWS_HOURS

def get_engine():
    url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

def get_latest_news(symbol, hours=NEWS_HOURS):
    engine = get_engine()
    query = """
        SELECT * FROM news
        WHERE symbol=%s AND published_at >= NOW() - INTERVAL %s HOUR
        ORDER BY published_at DESC
    """
    df = pd.read_sql(query, engine, params=(symbol, hours))
    return df