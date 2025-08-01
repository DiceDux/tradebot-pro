import time
import pandas as pd
import mysql.connector
from tabulate import tabulate
import sys
import os
from datetime import datetime

# تنظیمات اتصال به MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # رمز عبور را اینجا وارد کنید
    'database': 'database'  # نام دیتابیس خود را وارد کنید
}

class FeatureMonitor:
    def __init__(self, symbol='BTCUSDT', refresh_rate=5):
        self.symbol = symbol
        self.refresh_rate = refresh_rate
        self.last_values = {}
        self.last_update_time = None
        
    def get_latest_data(self, force_reconnect=True):
        """دریافت آخرین داده‌ها با اتصال مجدد اجباری به دیتابیس"""
        try:
            # اتصال جدید برای هر درخواست
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # زمان فعلی برای گزارش
            current_time = datetime.now()
            
            # دریافت آخرین کندل‌ها
            candles_query = f"SELECT * FROM candles WHERE symbol='{self.symbol}' ORDER BY timestamp DESC LIMIT 300"
            cursor.execute(candles_query)
            candles = cursor.fetchall()
            
            # تبدیل به DataFrame
            candles_df = pd.DataFrame(candles) if candles else pd.DataFrame()
            
            # دریافت آخرین اخبار
            base_symbol = self.symbol.replace('USDT', '')
            news_query = f"SELECT * FROM news WHERE symbol IN ('{base_symbol}', 'BITCOIN', 'BTC', 'ETHEREUM', 'ETH') ORDER BY published_at DESC LIMIT 100"
            cursor.execute(news_query)
            news = cursor.fetchall()
            
            # تبدیل به DataFrame
            news_df = pd.DataFrame(news) if news else pd.DataFrame()
            
            cursor.close()
            conn.close()
            
            print(f"Data fetched at {current_time.strftime('%H:%M:%S')}: {len(candles_df)} candles, {len(news_df)} news")
            
            return candles_df, news_df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_features_manually(self, candles_df):
        """محاسبه دستی برخی فیچرهای مهم برای نمایش"""
        if candles_df.empty:
            return {}
            
        features = {}
        
        # تبدیل به سری‌های پانداس برای محاسبات آسان‌تر
        close = pd.Series(candles_df['close'].values)
        high = pd.Series(candles_df['high'].values)
        low = pd.Series(candles_df['low'].values)
        open_ = pd.Series(candles_df['open'].values)
        volume = pd.Series(candles_df['volume'].values)
        
        # قیمت‌های کندل آخر
        features['close'] = close.iloc[-1] if len(close) > 0 else 0
        features['open'] = open_.iloc[-1] if len(open_) > 0 else 0
        features['high'] = high.iloc[-1] if len(high) > 0 else 0
        features['low'] = low.iloc[-1] if len(low) > 0 else 0
        
        # محاسبه میانگین‌های متحرک
        if len(close) >= 20:
            features['sma20'] = close[-20:].mean()
            features['ema20'] = close[-20:].ewm(span=20).mean().iloc[-1]
        
        if len(close) >= 50:
            features['sma50'] = close[-50:].mean() 
            features['ema50'] = close[-50:].ewm(span=50).mean().iloc[-1]
            
        # محاسبه RSI
        if len(close) >= 15:
            delta = close[-15:].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-5)
            features['rsi14'] = 100 - (100 / (1 + rs.iloc[-1])) if not rs.empty else 50
        
        # باندهای بولینگر
        if len(close) >= 20:
            middle = close[-20:].rolling(20).mean().iloc[-1]
            std = close[-20:].rolling(20).std().iloc[-1]
            features['bb_upper'] = middle + 2 * std
            features['bb_lower'] = middle - 2 * std
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        # محاسبه MACD
        if len(close) >= 27:
            ema12 = close[-27:].ewm(span=12).mean()
            ema26 = close[-27:].ewm(span=26).mean()
            macd = ema12 - ema26
            features['macd'] = macd.iloc[-1]
            
        # کندل سبز یا قرمز
        if len(close) > 0 and len(open_) > 0:
            features['green_candle'] = 1 if close.iloc[-1] > open_.iloc[-1] else 0
            features['candle_change'] = (close.iloc[-1] - open_.iloc[-1]) / open_.iloc[-1] * 100  # درصد تغییر
            
        # اضافه کردن زمان بروزرسانی
        self.last_update_time = datetime.now()
        
        return features
    
    def display_features(self, features):
        """نمایش زیبای فیچرها در کنسول"""
        if not features:
            print("No feature data available")
            return
            
        # پاک کردن صفحه
        os.system('cls' if os.name=='nt' else 'clear')
        
        print(f"=== LIVE FEATURE MONITOR: {self.symbol} ===")
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # دسته‌بندی فیچرها
        price_info = []
        technical_indicators = []
        
        for feature, value in features.items():
            if feature in ['open', 'high', 'low', 'close']:
                price_info.append((feature, f"{value:.2f}"))
            else:
                technical_indicators.append((feature, f"{value:.4f}" if isinstance(value, float) else value))
        
        # نمایش اطلاعات قیمت
        print("\n=== PRICE INFORMATION ===")
        print(tabulate(price_info, headers=['Metric', 'Value'], tablefmt='pretty'))
        
        # نمایش اندیکاتورهای تکنیکال
        print("\n=== TECHNICAL INDICATORS ===")
        print(tabulate(technical_indicators, headers=['Indicator', 'Value'], tablefmt='pretty'))
        
        # نمایش سیگنال ساده
        if 'close' in features and 'ema20' in features and 'ema50' in features:
            close = features['close']
            ema20 = features['ema20']
            ema50 = features['ema50']
            
            print("\n=== SIMPLE SIGNALS ===")
            signals = []
            
            # سیگنال مبتنی بر EMA
            if close > ema20 > ema50:
                signals.append(("EMA Cross", "Bullish ↑"))
            elif close < ema20 < ema50:
                signals.append(("EMA Cross", "Bearish ↓"))
            else:
                signals.append(("EMA Cross", "Neutral →"))
                
            # سیگنال مبتنی بر RSI
            if 'rsi14' in features:
                rsi = features['rsi14']
                if rsi > 70:
                    signals.append(("RSI", f"Overbought ({rsi:.1f}) ↓"))
                elif rsi < 30:
                    signals.append(("RSI", f"Oversold ({rsi:.1f}) ↑"))
                else:
                    signals.append(("RSI", f"Neutral ({rsi:.1f}) →"))
            
            print(tabulate(signals, headers=['Signal Type', 'Direction'], tablefmt='pretty'))
    
    def run(self):
        """اجرای مانیتورینگ مستمر با اتصال مجدد به دیتابیس در هر بار"""
        try:
            while True:
                # دریافت داده‌های جدید با اتصال مجدد اجباری
                candles_df, news_df = self.get_latest_data(force_reconnect=True)
                
                # محاسبه فیچرها به صورت دستی
                features = self.calculate_features_manually(candles_df)
                
                # نمایش فیچرها
                self.display_features(features)
                
                # انتظار برای بروزرسانی بعدی
                print(f"\nWaiting {self.refresh_rate} seconds for next update...")
                time.sleep(self.refresh_rate)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring: {e}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    refresh_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Starting feature monitor for {symbol} with refresh rate of {refresh_rate} seconds")
    monitor = FeatureMonitor(symbol, refresh_rate)
    monitor.run()