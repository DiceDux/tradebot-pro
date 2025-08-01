"""
نمایش زنده وضعیت فیچرها و مقادیر محاسبه شده
"""
import time
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
from datetime import datetime
import sys
from threading import Thread

from feature_store.feature_database import FeatureDatabase

class FeatureMonitor:
    def __init__(self, symbol='BTCUSDT', refresh_rate=5):
        """
        نمایش زنده وضعیت فیچرها
        
        Args:
            symbol: نماد مورد نظر
            refresh_rate: فاصله زمانی بروزرسانی (ثانیه)
        """
        self.symbol = symbol
        self.refresh_rate = refresh_rate
        self.db = FeatureDatabase()
        self.running = False
        self.thread = None
        self.last_update_time = None
        self.feature_history = {}  # نگهداری تاریخچه تغییرات
        
    def get_latest_features(self):
        """دریافت آخرین مقادیر فیچرها"""
        return self.db.get_latest_features(self.symbol)
    
    def update_feature_history(self, features_df):
        """بروزرسانی تاریخچه تغییرات فیچرها"""
        if features_df is None or features_df.empty:
            return
            
        now = datetime.now()
        
        for column in features_df.columns:
            value = features_df[column].values[0]
            
            if column not in self.feature_history:
                self.feature_history[column] = []
                
            # نگهداری 5 مقدار آخر
            self.feature_history[column].append((now, value))
            if len(self.feature_history[column]) > 5:
                self.feature_history[column].pop(0)
    
    def calculate_changes(self, feature_name):
        """محاسبه تغییرات یک فیچر"""
        if feature_name not in self.feature_history or len(self.feature_history[feature_name]) < 2:
            return None, None
            
        history = self.feature_history[feature_name]
        latest_value = history[-1][1]
        previous_value = history[-2][1]
        
        if previous_value == 0:
            pct_change = 0
        else:
            pct_change = (latest_value - previous_value) / abs(previous_value) * 100
            
        return latest_value - previous_value, pct_change
    
    def display_features(self, features_df):
        """نمایش زیبای فیچرها در کنسول"""
        if features_df is None or features_df.empty:
            print("No feature data available")
            return
            
        # پاک کردن صفحه
        os.system('cls' if os.name=='nt' else 'clear')
        
        print(f"=== LIVE FEATURE MONITOR: {self.symbol} ===")
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total features: {len(features_df.columns)}")
        
        # دسته‌بندی فیچرها
        technical_features = []
        news_features = []
        pattern_features = []
        price_features = []
        zero_features = []
        
        for feature in features_df.columns:
            value = features_df[feature].values[0]
            
            # اگر تاریخچه داریم، تغییرات را محاسبه می‌کنیم
            abs_change, pct_change = self.calculate_changes(feature)
            change_str = ""
            if abs_change is not None and pct_change is not None:
                direction = "↑" if abs_change > 0 else "↓" if abs_change < 0 else "-"
                change_str = f"{direction} {abs(pct_change):.2f}%"
            
            if value == 0:
                zero_features.append((feature, value, change_str))
                continue
                
            if feature in ['open', 'high', 'low', 'close', 'volume']:
                price_features.append((feature, value, change_str))
            elif feature.startswith('news_'):
                news_features.append((feature, value, change_str))
            elif feature in ['doji', 'engulfing', 'hammer', 'morning_star', 'evening_star', 'shooting_star']:
                pattern_features.append((feature, value, change_str))
            else:
                technical_features.append((feature, value, change_str))
        
        # نمایش فیچرهای قیمت
        if price_features:
            print("\n=== PRICE INFORMATION ===")
            price_data = [(f, f"{v:.2f}", c) for f, v, c in price_features]
            print(tabulate(price_data, headers=['Feature', 'Value', 'Change'], tablefmt='pretty'))
        
        # نمایش فیچرهای تکنیکال غیر صفر
        print("\n=== TECHNICAL INDICATORS ===")
        tech_data = [(f, f"{v:.6f}", c) for f, v, c in sorted(technical_features, key=lambda x: abs(x[1]), reverse=True)]
        print(tabulate(tech_data[:20], headers=['Feature', 'Value', 'Change'], tablefmt='pretty'))
        
        if len(tech_data) > 20:
            print(f"...and {len(tech_data) - 20} more technical features")
        
        # نمایش فیچرهای خبری غیر صفر
        print("\n=== NEWS FEATURES ===")
        if news_features:
            news_data = [(f, f"{v:.6f}", c) for f, v, c in sorted(news_features, key=lambda x: abs(x[1]), reverse=True)]
            print(tabulate(news_data, headers=['Feature', 'Value', 'Change'], tablefmt='pretty'))
        else:
            print("No non-zero news features")
        
        # نمایش فیچرهای الگو
        print("\n=== PATTERN FEATURES ===")
        if pattern_features:
            pattern_data = [(f, f"{v:.6f}", c) for f, v, c in sorted(pattern_features, key=lambda x: abs(x[1]), reverse=True)]
            print(tabulate(pattern_data, headers=['Feature', 'Value', 'Change'], tablefmt='pretty'))
        else:
            print("No non-zero pattern features")
            
        # خلاصه آمار
        print("\n=== FEATURE STATISTICS ===")
        print(f"Non-zero features: {len(technical_features) + len(news_features) + len(pattern_features) + len(price_features)}")
        print(f"Zero features: {len(zero_features)}")
        
        # نمایش 10 فیچر صفر اول
        if zero_features:
            print("\n=== SOME ZERO-VALUED FEATURES ===")
            zero_data = [(f, v, c) for f, v, c in sorted(zero_features)[:10]]
            print(tabulate(zero_data, headers=['Feature', 'Value', 'Change'], tablefmt='pretty'))
            if len(zero_features) > 10:
                print(f"...and {len(zero_features) - 10} more zero features")
        
        # نمایش سیگنال‌های ساده
        self.display_simple_signals(features_df)
    
    def display_simple_signals(self, features_df):
        """نمایش سیگنال‌های ساده براساس فیچرها"""
        if features_df is None or features_df.empty:
            return
            
        signals = []
        
        # بررسی EMA Cross
        if 'ema20' in features_df.columns and 'ema50' in features_df.columns and 'close' in features_df.columns:
            close = features_df['close'].values[0]
            ema20 = features_df['ema20'].values[0]
            ema50 = features_df['ema50'].values[0]
            
            if close > ema20 > ema50:
                signals.append(("EMA Cross", "Bullish ↑", "Strong"))
            elif close < ema20 < ema50:
                signals.append(("EMA Cross", "Bearish ↓", "Strong"))
            elif close > ema20 and ema20 < ema50:
                signals.append(("EMA Cross", "Bullish ↑", "Weak"))
            elif close < ema20 and ema20 > ema50:
                signals.append(("EMA Cross", "Bearish ↓", "Weak"))
            else:
                signals.append(("EMA Cross", "Neutral →", "Neutral"))
        
        # بررسی RSI
        if 'rsi14' in features_df.columns:
            rsi = features_df['rsi14'].values[0]
            if rsi > 70:
                signals.append(("RSI", f"Overbought ({rsi:.1f}) ↓", "Strong"))
            elif rsi < 30:
                signals.append(("RSI", f"Oversold ({rsi:.1f}) ↑", "Strong"))
            elif rsi > 60:
                signals.append(("RSI", f"Bullish ({rsi:.1f}) ↑", "Moderate"))
            elif rsi < 40:
                signals.append(("RSI", f"Bearish ({rsi:.1f}) ↓", "Moderate"))
            else:
                signals.append(("RSI", f"Neutral ({rsi:.1f}) →", "Neutral"))
        
        # بررسی MACD
        if 'macd' in features_df.columns and 'macd_signal' in features_df.columns:
            macd = features_df['macd'].values[0]
            macd_signal = features_df['macd_signal'].values[0]
            
            if macd > macd_signal and macd > 0:
                signals.append(("MACD", f"Bullish ↑", "Strong"))
            elif macd < macd_signal and macd < 0:
                signals.append(("MACD", f"Bearish ↓", "Strong"))
            elif macd > macd_signal:
                signals.append(("MACD", f"Bullish ↑", "Weak"))
            elif macd < macd_signal:
                signals.append(("MACD", f"Bearish ↓", "Weak"))
            else:
                signals.append(("MACD", f"Neutral →", "Neutral"))
        
        # بررسی بولینگر باندز
        if 'close' in features_df.columns and 'bb_upper' in features_df.columns and 'bb_lower' in features_df.columns:
            close = features_df['close'].values[0]
            bb_upper = features_df['bb_upper'].values[0]
            bb_lower = features_df['bb_lower'].values[0]
            
            if close > bb_upper:
                signals.append(("Bollinger", f"Overbought ↓", "Strong"))
            elif close < bb_lower:
                signals.append(("Bollinger", f"Oversold ↑", "Strong"))
            elif close > (bb_upper + bb_lower) / 2:
                signals.append(("Bollinger", f"Bullish ↑", "Moderate"))
            else:
                signals.append(("Bollinger", f"Bearish ↓", "Moderate"))
        
        if signals:
            print("\n=== SIMPLE SIGNALS ===")
            print(tabulate(signals, headers=['Indicator', 'Signal', 'Strength'], tablefmt='pretty'))
    
    def run(self):
        """اجرای حلقه نمایش فیچرها"""
        self.running = True
        
        def _worker():
            while self.running:
                try:
                    # دریافت آخرین فیچرها
                    features_df = self.get_latest_features()
                    
                    # بروزرسانی تاریخچه
                    self.update_feature_history(features_df)
                    
                    # نمایش فیچرها
                    self.display_features(features_df)
                    
                    # انتظار برای بروزرسانی بعدی
                    print(f"\nWaiting {self.refresh_rate} seconds for next update...")
                    time.sleep(self.refresh_rate)
                except Exception as e:
                    print(f"Error in feature monitor: {e}")
                    time.sleep(self.refresh_rate)
        
        self.thread = Thread(target=_worker)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"Feature monitor started for {self.symbol}")
    
    def stop(self):
        """توقف حلقه نمایش"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.db.close()
        print("Feature monitor stopped")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    refresh_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    monitor = FeatureMonitor(symbol, refresh_rate)
    monitor.run()
    
    try:
        # نگه داشتن برنامه در حال اجرا
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitoring stopped by user")