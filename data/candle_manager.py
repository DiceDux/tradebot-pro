import pandas as pd
from sqlalchemy import create_engine
from utils.config import DB_CONFIG, CANDLE_LIMIT

def get_engine():
    url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

def get_latest_candles(symbol, limit=CANDLE_LIMIT):
    engine = get_engine()
    query = """
        SELECT * FROM candles
        WHERE symbol=%s
        ORDER BY timestamp DESC
        LIMIT %s
    """
    df = pd.read_sql(query, engine, params=(symbol, limit))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
