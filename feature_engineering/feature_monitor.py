import time
import pandas as pd
import mysql.connector
from tabulate import tabulate
import sys
import os
from feature_engineer import build_features
from datetime import datetime

# تنظیمات اتصال به MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # رمز عبور را اینجا وارد کنید
    'database': 'tradebot-pro'  # نام دیتابیس خود را وارد کنید
}

class FeatureMonitor:
    def __init__(self, symbol='BTCUSDT', refresh_rate=10):
        self.symbol = symbol
        self.refresh_rate = refresh_rate
        self.features_history = {}
        self.last_values = {}
        
    def get_latest_data(self):
        """دریافت آخرین داده‌های کندل و خبر از دیتابیس"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # دریافت آخرین کندل‌ها
            candles_query = f"SELECT * FROM candles WHERE symbol='{self.symbol}' ORDER BY timestamp DESC LIMIT 300"
            cursor.execute(candles_query)
            candles = cursor.fetchall()
            
            # تبدیل به DataFrame
            candles_df = pd.DataFrame(candles)
            
            # دریافت آخرین اخبار
            base_symbol = self.symbol.replace('USDT', '')
            news_query = f"SELECT * FROM news WHERE symbol IN ('{base_symbol}', 'BITCOIN', 'BTC', 'ETHEREUM', 'ETH') ORDER BY published_at DESC LIMIT 100"
            cursor.execute(news_query)
            news = cursor.fetchall()
            
            # تبدیل به DataFrame
            news_df = pd.DataFrame(news)
            
            cursor.close()
            conn.close()
            
            return candles_df, news_df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_features(self):
        """محاسبه تمام فیچرها برای آخرین داده‌ها"""
        candles_df, news_df = self.get_latest_data()
        
        if candles_df.empty:
            print("No candle data available")
            return None
            
        print(f"Calculating features based on {len(candles_df)} candles and {len(news_df)} news items")
        
        # محاسبه فیچرها
        features = build_features(candles_df, news_df, self.symbol)
        
        # ذخیره تاریخچه
        now = datetime.now().strftime('%H:%M:%S')
        
        if isinstance(features, pd.DataFrame):
            features_dict = features.iloc[0].to_dict()
        else:
            features_dict = features
            
        return features_dict
    
    def track_feature_changes(self, features):
        """ردیابی تغییرات فیچرها"""
        if features is None:
            return
            
        # بروزرسانی تاریخچه
        now = datetime.now().strftime('%H:%M:%S')
        
        for feature, value in features.items():
            if feature not in self.features_history:
                self.features_history[feature] = []
                
            # نگهداری فقط 5 مقدار آخر
            if len(self.features_history[feature]) >= 5:
                self.features_history[feature].pop(0)
                
            self.features_history[feature].append((now, value))
            
        # ذخیره آخرین مقادیر
        self.last_values = features
    
    def display_features(self):
        """نمایش زیبای فیچرها در کنسول"""
        if not self.last_values:
            print("No feature data available yet")
            return
            
        # پاک کردن صفحه
        os.system('cls' if os.name=='nt' else 'clear')
        
        print(f"=== FEATURE MONITOR: {self.symbol} ===")
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total features: {len(self.last_values)}")
        
        # دسته‌بندی فیچرها
        technical_features = []
        news_features = []
        pattern_features = []
        zero_features = []
        
        for feature, value in self.last_values.items():
            if value == 0:
                zero_features.append((feature, value))
                continue
                
            if feature.startswith('news_'):
                news_features.append((feature, value))
            elif feature in ['doji', 'engulfing', 'hammer', 'morning_star', 'evening_star', 'shooting_star']:
                pattern_features.append((feature, value))
            else:
                technical_features.append((feature, value))
        
        # نمایش فیچرهای تکنیکال غیر صفر
        print("\n=== TECHNICAL INDICATORS ===")
        tech_data = [(f, f"{v:.6f}") for f, v in sorted(technical_features, key=lambda x: abs(x[1]), reverse=True)]
        print(tabulate(tech_data[:20], headers=['Feature', 'Value'], tablefmt='pretty'))
        
        if len(tech_data) > 20:
            print(f"...and {len(tech_data) - 20} more technical features")
        
        # نمایش فیچرهای خبری غیر صفر
        print("\n=== NEWS FEATURES ===")
        if news_features:
            news_data = [(f, f"{v:.6f}") for f, v in sorted(news_features, key=lambda x: abs(x[1]), reverse=True)]
            print(tabulate(news_data, headers=['Feature', 'Value'], tablefmt='pretty'))
        else:
            print("No non-zero news features")
        
        # نمایش فیچرهای الگو
        print("\n=== PATTERN FEATURES ===")
        if pattern_features:
            pattern_data = [(f, f"{v:.6f}") for f, v in sorted(pattern_features, key=lambda x: abs(x[1]), reverse=True)]
            print(tabulate(pattern_data, headers=['Feature', 'Value'], tablefmt='pretty'))
        else:
            print("No non-zero pattern features")
            
        # خلاصه آمار
        print("\n=== FEATURE STATISTICS ===")
        print(f"Non-zero features: {len(technical_features) + len(news_features) + len(pattern_features)}")
        print(f"Zero features: {len(zero_features)}")
        
        # نمایش 10 فیچر صفر اول
        if zero_features:
            print("\n=== SOME ZERO-VALUED FEATURES ===")
            zero_data = [(f, v) for f, v in sorted(zero_features)[:10]]
            print(tabulate(zero_data, headers=['Feature', 'Value'], tablefmt='pretty'))
            if len(zero_features) > 10:
                print(f"...and {len(zero_features) - 10} more zero features")
    
    def run(self):
        """اجرای مانیتورینگ مستمر"""
        try:
            while True:
                features = self.calculate_features()
                self.track_feature_changes(features)
                self.display_features()
                time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring: {e}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    refresh_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Starting feature monitor for {symbol} with refresh rate of {refresh_rate} seconds")
    monitor = FeatureMonitor(symbol, refresh_rate)
    monitor.run()
