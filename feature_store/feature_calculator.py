"""
سرویس محاسبه زنده فیچرها و ذخیره آنها در دیتابیس
"""
import time
import pandas as pd
import mysql.connector
from threading import Thread
from datetime import datetime

from feature_engineering.feature_engineer import build_features
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news

# تنظیمات دیتابیس
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tradebot-pro",
    "port": 3306
}

class FeatureCalculator:
    def __init__(self, symbols=None, update_interval=1):
        """
        سرویس محاسبه زنده فیچرها
        
        Args:
            symbols: لیست نمادهای مورد نظر
            update_interval: فاصله زمانی بروزرسانی فیچرها (ثانیه)
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.conn = None
        self.initialize_db()
        
    def initialize_db(self):
        """ایجاد جدول فیچرها اگر وجود ندارد"""
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            cursor = self.conn.cursor()
            
            # ایجاد جدول features اگر وجود ندارد
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_features (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                feature_name VARCHAR(50),
                feature_value FLOAT,
                timestamp INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # ایجاد ایندکس‌ها
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_feature ON live_features (symbol, feature_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON live_features (timestamp)")
            
            self.conn.commit()
            cursor.close()
            print("Database initialized successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def calculate_and_store_features(self, symbol):
        """محاسبه و ذخیره فیچرها برای یک نماد"""
        try:
            # دریافت آخرین کندل‌ها و اخبار
            candles = get_latest_candles(symbol, 300)
            news = get_latest_news(symbol, 72)  # 72 ساعت اخیر
            
            # محاسبه فیچرها
            features_df = build_features(candles, news, symbol)
            
            # ذخیره فیچرها در دیتابیس
            if features_df is not None and not features_df.empty:
                current_ts = int(time.time())
                
                try:
                    if not self.conn or not self.conn.is_connected():
                        self.conn = mysql.connector.connect(**DB_CONFIG)
                    
                    cursor = self.conn.cursor()
                    
                    # حذف فیچرهای قبلی برای همین نماد
                    cursor.execute(
                        "DELETE FROM live_features WHERE symbol = %s AND timestamp = %s",
                        (symbol, current_ts)
                    )
                    
                    # افزودن فیچرهای جدید
                    for feature_name, value in features_df.iloc[0].items():
                        cursor.execute(
                            "INSERT INTO live_features (symbol, feature_name, feature_value, timestamp) VALUES (%s, %s, %s, %s)",
                            (symbol, feature_name, float(value), current_ts)
                        )
                    
                    self.conn.commit()
                    cursor.close()
                    
                    print(f"[{datetime.now()}] Updated {len(features_df.columns)} features for {symbol}")
                    return True
                except Exception as db_error:
                    print(f"Database error while storing features: {db_error}")
                    # اتصال مجدد در صورت خطا
                    try:
                        self.conn = mysql.connector.connect(**DB_CONFIG)
                    except:
                        pass
                    return False
        except Exception as e:
            print(f"Error calculating features for {symbol}: {e}")
            return False
    
    def update_all_symbols(self):
        """بروزرسانی فیچرها برای تمام نمادها"""
        results = {}
        for symbol in self.symbols:
            success = self.calculate_and_store_features(symbol)
            results[symbol] = success
        return results
    
    def run(self):
        """اجرای سرویس محاسبه فیچر"""
        self.running = True
        
        def _worker():
            while self.running:
                start_time = time.time()
                self.update_all_symbols()
                
                # محاسبه زمان باقی‌مانده تا بروزرسانی بعدی
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self.thread = Thread(target=_worker)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"Feature calculator started. Updating {len(self.symbols)} symbols every {self.update_interval} seconds.")
    
    def stop(self):
        """توقف سرویس"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.conn:
            self.conn.close()
        print("Feature calculator stopped.")

if __name__ == "__main__":
    calculator = FeatureCalculator(update_interval=1)  # هر 1 ثانیه بروز می‌شود
    calculator.run()
    
    try:
        # نگه داشتن برنامه در حال اجرا
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        calculator.stop()
        print("Service stopped by user")
