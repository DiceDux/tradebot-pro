"""
مدیریت ذخیره‌سازی و بازیابی فیچرها از دیتابیس
"""
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# تنظیمات دیتابیس
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tradebot-pro",
    "port": 3306
}

class FeatureDatabase:
    def __init__(self):
        """مدیریت دسترسی به فیچرهای ذخیره شده در دیتابیس"""
        self.conn = None
        self.connect()
        
    def connect(self):
        """اتصال به دیتابیس"""
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def get_latest_features(self, symbol):
        """دریافت آخرین مقادیر فیچرها برای یک نماد"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.connect()
                
            cursor = self.conn.cursor(dictionary=True)
            
            # دریافت آخرین timestamp
            cursor.execute(
                "SELECT MAX(timestamp) as latest_ts FROM live_features WHERE symbol = %s",
                (symbol,)
            )
            result = cursor.fetchone()
            
            if not result or not result['latest_ts']:
                return pd.DataFrame()
                
            latest_ts = result['latest_ts']
            
            # دریافت تمام فیچرهای مربوط به آخرین timestamp
            cursor.execute(
                "SELECT feature_name, feature_value FROM live_features WHERE symbol = %s AND timestamp = %s",
                (symbol, latest_ts)
            )
            features = cursor.fetchall()
            
            cursor.close()
            
            # تبدیل به DataFrame
            if features:
                features_dict = {item['feature_name']: item['feature_value'] for item in features}
                return pd.DataFrame([features_dict])
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting latest features: {e}")
            return pd.DataFrame()
    
    def get_features_history(self, symbol, feature_names, hours=24):
        """
        دریافت تاریخچه مقادیر فیچرها
        
        Args:
            symbol: نماد مورد نظر
            feature_names: لیست نام فیچرها
            hours: تعداد ساعت‌های گذشته
            
        Returns:
            DataFrame با تاریخچه فیچرها
        """
        try:
            if not self.conn or not self.conn.is_connected():
                self.connect()
                
            cursor = self.conn.cursor(dictionary=True)
            
            # زمان شروع بازه
            start_ts = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            # تبدیل لیست فیچرها به رشته برای استفاده در کوئری
            features_str = ', '.join([f"'{name}'" for name in feature_names])
            
            query = f"""
            SELECT timestamp, feature_name, feature_value
            FROM live_features
            WHERE symbol = %s AND feature_name IN ({features_str}) AND timestamp >= %s
            ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (symbol, start_ts))
            records = cursor.fetchall()
            
            cursor.close()
            
            # تبدیل به DataFrame محوری
            if records:
                df = pd.DataFrame(records)
                pivot_df = df.pivot(index='timestamp', columns='feature_name', values='feature_value')
                pivot_df.index = pd.to_datetime(pivot_df.index, unit='s')
                return pivot_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting features history: {e}")
            return pd.DataFrame()
    
    def get_available_features(self, symbol):
        """دریافت لیست تمام فیچرهای موجود برای یک نماد"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.connect()
                
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT DISTINCT feature_name FROM live_features WHERE symbol = %s",
                (symbol,)
            )
            
            features = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            return features
            
        except Exception as e:
            print(f"Error getting available features: {e}")
            return []
    
    def cleanup_old_features(self, days=7):
        """پاک کردن فیچرهای قدیمی از دیتابیس"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.connect()
                
            cursor = self.conn.cursor()
            
            # زمان قطع
            cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
            
            cursor.execute(
                "DELETE FROM live_features WHERE timestamp < %s",
                (cutoff_ts,)
            )
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            cursor.close()
            
            print(f"Deleted {deleted_count} old feature records")
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up old features: {e}")
            return 0
            
    def close(self):
        """بستن اتصال دیتابیس"""
        if self.conn:
            self.conn.close()