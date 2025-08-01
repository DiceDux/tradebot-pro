"""
مدیریت دیتابیس برای ذخیره و بازیابی فیچرها و سیگنال‌های معاملاتی
"""
import mysql.connector
import pandas as pd
from datetime import datetime
import numpy as np
import logging
import os
import json

# تنظیم لاگر
logger = logging.getLogger("feature_database")
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/database.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class FeatureDatabase:
    """کلاس مدیریت دیتابیس برای ذخیره و بازیابی فیچرها"""
    
    def __init__(self, config_file='config/db_config.json'):
        """مقداردهی اولیه با پیکربندی از فایل"""
        self.config_file = config_file
        self.db_config = self._load_config()
        self._initialize_database()
        
    def _load_config(self):
        """بارگیری تنظیمات از فایل پیکربندی یا استفاده از مقادیر پیش‌فرض"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # تنظیمات پیش‌فرض
                config = {
                    'host': 'localhost',
                    'user': 'root',
                    'password': '',
                    'database': 'tradebot-pro'
                }
                # ساخت دایرکتوری config اگر وجود ندارد
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                # ذخیره تنظیمات پیش‌فرض
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                return config
        except Exception as e:
            logger.error(f"Error loading database config: {e}")
            # برگشت تنظیمات پیش‌فرض در صورت خطا
            return {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'tradebot-pro'
            }
        
    def _connect(self):
        """اتصال به دیتابیس"""
        return mysql.connector.connect(**self.db_config)
        
    def _initialize_database(self):
        """ایجاد جداول مورد نیاز در دیتابیس"""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            # ایجاد جدول فیچرهای زنده
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_features (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    features JSON NOT NULL,
                    UNIQUE KEY symbol_timestamp (symbol, timestamp)
                )
            """)
            
            # ایجاد جدول سیگنال‌های معاملاتی
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_signals (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    decision TINYINT NOT NULL,
                    confidence FLOAT NOT NULL,
                    sell_probability FLOAT NOT NULL,
                    hold_probability FLOAT NOT NULL,
                    buy_probability FLOAT NOT NULL
                )
            """)
            
            # ایجاد جدول تاریخچه معاملات
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    action ENUM('BUY', 'SELL') NOT NULL,
                    price DECIMAL(18,8) NOT NULL,
                    amount DECIMAL(18,8) NOT NULL,
                    total_value DECIMAL(18,8) NOT NULL,
                    profit_loss DECIMAL(18,8) NULL,
                    success BOOLEAN NOT NULL DEFAULT TRUE,
                    notes TEXT NULL
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def save_features(self, symbol, features_dict):
        """
        ذخیره فیچرها در دیتابیس
        
        Args:
            symbol: نماد معاملاتی
            features_dict: دیکشنری فیچرها
        
        Returns:
            bool: آیا عملیات موفقیت‌آمیز بود یا خیر
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            # تبدیل مقادیر numpy به float برای سریالایز شدن به JSON
            for key, value in features_dict.items():
                if isinstance(value, np.number):
                    features_dict[key] = float(value)
            
            # تبدیل به JSON
            features_json = json.dumps(features_dict)
            
            # درج یا آپدیت رکورد
            query = """
                INSERT INTO live_features (symbol, timestamp, features)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE features = VALUES(features)
            """
            
            now = datetime.now()
            cursor.execute(query, (symbol, now, features_json))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving features for {symbol}: {e}")
            return False
            
    def get_latest_features(self, symbol):
        """
        دریافت آخرین فیچرها برای یک نماد
        
        Args:
            symbol: نماد معاملاتی
            
        Returns:
            DataFrame: دیتافریم حاوی فیچرها یا None در صورت خطا
        """
        try:
            conn = self._connect()
            cursor = conn.cursor(dictionary=True)
            
            # دریافت آخرین رکورد برای نماد مورد نظر
            query = """
                SELECT * FROM live_features
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if row:
                # تبدیل JSON به دیکشنری
                features_dict = json.loads(row['features'])
                # تبدیل به دیتافریم
                return pd.DataFrame([features_dict])
            return None
        except Exception as e:
            logger.error(f"Error getting latest features for {symbol}: {e}")
            return None
            
    def get_historical_features(self, symbol, limit=1000):
        """
        دریافت تاریخچه فیچرها برای یک نماد
        
        Args:
            symbol: نماد معاملاتی
            limit: حداکثر تعداد رکوردها
            
        Returns:
            DataFrame: دیتافریم حاوی تاریخچه فیچرها یا None در صورت خطا
        """
        try:
            conn = self._connect()
            cursor = conn.cursor(dictionary=True)
            
            # دریافت رکوردها برای نماد مورد نظر
            query = """
                SELECT * FROM live_features
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            
            cursor.execute(query, (symbol, limit))
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if rows:
                # تبدیل هر رکورد به دیتافریم
                dfs = []
                for row in rows:
                    features_dict = json.loads(row['features'])
                    df = pd.DataFrame([features_dict])
                    df['timestamp'] = row['timestamp']
                    dfs.append(df)
                
                # ترکیب همه دیتافریم‌ها
                return pd.concat(dfs).reset_index(drop=True)
            return None
        except Exception as e:
            logger.error(f"Error getting historical features for {symbol}: {e}")
            return None
            
    def insert_trade_signal(self, symbol, timestamp, decision, confidence, sell_prob, hold_prob, buy_prob):
        """
        ذخیره سیگنال معاملاتی در دیتابیس
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            # درج سیگنال
            cursor.execute("""
                INSERT INTO trade_signals 
                (symbol, timestamp, decision, confidence, sell_probability, hold_probability, buy_probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (symbol, timestamp, decision, confidence, sell_prob, hold_prob, buy_prob))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Trade signal recorded for {symbol}: decision={decision}, confidence={confidence:.2f}")
            return True
        except Exception as e:
            logger.error(f"Error inserting trade signal: {e}")
            return False
            
    def get_latest_signals(self, symbol=None, limit=10):
        """
        دریافت آخرین سیگنال‌های معاملاتی
        
        Args:
            symbol: نماد معاملاتی (اختیاری)
            limit: حداکثر تعداد رکوردها
            
        Returns:
            DataFrame: دیتافریم حاوی سیگنال‌ها یا None در صورت خطا
        """
        try:
            conn = self._connect()
            cursor = conn.cursor(dictionary=True)
            
            if symbol:
                query = """
                    SELECT * FROM trade_signals
                    WHERE symbol = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """
                cursor.execute(query, (symbol, limit))
            else:
                query = """
                    SELECT * FROM trade_signals
                    ORDER BY timestamp DESC
                    LIMIT %s
                """
                cursor.execute(query, (limit,))
                
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if rows:
                return pd.DataFrame(rows)
            return None
        except Exception as e:
            logger.error(f"Error getting latest trade signals: {e}")
            return None
            
    def record_trade(self, symbol, action, price, amount, total_value, profit_loss=None, notes=None):
        """
        ثبت یک معامله در تاریخچه معاملات
        
        Args:
            symbol: نماد معاملاتی
            action: نوع عملیات ('BUY' یا 'SELL')
            price: قیمت معامله
            amount: مقدار معامله شده
            total_value: ارزش کل معامله
            profit_loss: سود یا زیان (اختیاری)
            notes: توضیحات (اختیاری)
            
        Returns:
            bool: آیا عملیات موفقیت‌آمیز بود یا خیر
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO trade_history
                (symbol, timestamp, action, price, amount, total_value, profit_loss, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                symbol,
                datetime.now(),
                action,
                price,
                amount,
                total_value,
                profit_loss,
                notes
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Trade recorded: {symbol} {action} {amount} at {price}")
            return True
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
