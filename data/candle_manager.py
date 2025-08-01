"""
مدیریت کننده داده‌های کندل‌استیک
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import mysql.connector
from utils.config import DB_CONFIG

logger = logging.getLogger("candle_manager")

def get_latest_candles(symbol, limit=None):
    """
    دریافت آخرین کندل‌ها از دیتابیس
    
    Args:
        symbol: نماد ارز
        limit: محدودیت تعداد کندل‌ها (اگر None باشد، همه کندل‌ها برگردانده می‌شوند)
        
    Returns:
        DataFrame: داده‌های کندل‌استیک
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # استفاده از تمام داده‌های موجود اگر limit مشخص نشده باشد
        if limit is None:
            query = """
            SELECT * FROM candles 
            WHERE symbol = %s 
            ORDER BY timestamp ASC
            """
            cursor.execute(query, (symbol,))
        else:
            query = """
            SELECT * FROM candles 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            cursor.execute(query, (symbol, limit))
            
        rows = cursor.fetchall()
        
        # بازگرداندن به ترتیب زمانی اگر limit تعیین شده باشد
        if limit is not None:
            rows.reverse()
            
        if not rows:
            logger.warning(f"No candle data found for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        
        # تبدیل timestamp به datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # مرتب‌سازی بر اساس زمان
        df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} candles for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading candle data for {symbol}: {e}")
        return pd.DataFrame()
        
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
