"""
مدیریت کننده داده‌های خبری
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import mysql.connector
from utils.config import DB_CONFIG

logger = logging.getLogger("news_manager")

def get_latest_news(symbol, limit=None):
    """
    دریافت آخرین اخبار از دیتابیس
    
    Args:
        symbol: نماد ارز
        limit: محدودیت تعداد اخبار (اگر None باشد، همه اخبار برگردانده می‌شوند)
        
    Returns:
        DataFrame: داده‌های خبری
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # استفاده از تمام داده‌های موجود اگر limit مشخص نشده باشد
        if limit is None:
            query = """
            SELECT * FROM news 
            WHERE symbol = %s 
            ORDER BY published_at ASC
            """
            cursor.execute(query, (symbol,))
        else:
            query = """
            SELECT * FROM news 
            WHERE symbol = %s 
            ORDER BY published_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (symbol, limit))
            
        rows = cursor.fetchall()
        
        # بازگرداندن به ترتیب زمانی اگر limit تعیین شده باشد
        if limit is not None:
            rows.reverse()
            
        if not rows:
            logger.warning(f"No news data found for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        
        # تبدیل published_at به datetime
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # مرتب‌سازی بر اساس زمان
        df = df.sort_values('published_at')
        
        logger.info(f"Loaded {len(df)} news items for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading news data for {symbol}: {e}")
        return pd.DataFrame()
        
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
