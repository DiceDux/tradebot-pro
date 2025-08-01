"""
تنظیم پویای آستانه‌های تصمیم‌گیری
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os

class AdaptiveThresholds:
    def __init__(self):
        """
        تنظیم‌کننده پویای آستانه‌های تصمیم‌گیری
        """
        self.thresholds = {
            0: 0.4,  # آستانه برای کلاس Sell
            1: 0.5,  # آستانه برای کلاس Hold
            2: 0.4   # آستانه برای کلاس Buy
        }
        self.historical_signals = []
        self.last_update = None
        self.market_state = 'neutral'  # neutral, bullish, bearish
        self.volatility_level = 'normal'  # low, normal, high
        
        # بارگذاری آستانه‌های ذخیره شده
        self.load_thresholds()
    
    def update_thresholds(self, market_data=None):
        """
        بروزرسانی آستانه‌ها براساس شرایط بازار
        
        Args:
            market_data: داده‌های بازار
        """
        # اگر داده‌های بازار فراهم نشده باشد، از مقادیر پیش‌فرض استفاده می‌کنیم
        if market_data is None:
            self.last_update = datetime.now()
            return
        
        # بررسی روند بازار
        if 'ema20' in market_data and 'ema50' in market_data and 'close' in market_data:
            close = market_data['close']
            ema20 = market_data['ema20']
            ema50 = market_data['ema50']
            
            if close > ema20 > ema50:
                self.market_state = 'bullish'
            elif close < ema20 < ema50:
                self.market_state = 'bearish'
            else:
                self.market_state = 'neutral'
        
        # بررسی نوسان بازار
        if 'atr14' in market_data and 'bb_width' in market_data:
            atr = market_data.get('atr14', 0)
            bb_width = market_data.get('bb_width', 0)
            
            # تنظیم سطح نوسان
            if bb_width > 2.5 or atr > 0.02:  # مقادیر را بر اساس داده‌های واقعی خود تنظیم کنید
                self.volatility_level = 'high'
            elif bb_width < 1.0 or atr < 0.01:
                self.volatility_level = 'low'
            else:
                self.volatility_level = 'normal'
        
        # تنظیم آستانه‌ها براساس وضعیت بازار
        if self.market_state == 'bullish':
            self.thresholds[0] = 0.5  # آستانه بالاتر برای Sell در بازار صعودی
            self.thresholds[1] = 0.55  # آستانه بالاتر برای Hold در بازار صعودی
            self.thresholds[2] = 0.35  # آستانه پایین‌تر برای Buy در بازار صعودی
        elif self.market_state == 'bearish':
            self.thresholds[0] = 0.35  # آستانه پایین‌تر برای Sell در بازار نزولی
            self.thresholds[1] = 0.55  # آستانه بالاتر برای Hold در بازار نزولی
            self.thresholds[2] = 0.5  # آستانه بالاتر برای Buy در بازار نزولی
        else:
            self.thresholds[0] = 0.4  # آستانه متوسط برای Sell در بازار خنثی
            self.thresholds[1] = 0.5  # آستانه متوسط برای Hold در بازار خنثی
            self.thresholds[2] = 0.4  # آستانه متوسط برای Buy در بازار خنثی
        
        # تنظیم براساس سطح نوسان
        if self.volatility_level == 'high':
            # در نوسان بالا، آستانه Hold را افزایش می‌دهیم (محافظه‌کارتر می‌شویم)
            self.thresholds[1] += 0.05
        elif self.volatility_level == 'low':
            # در نوسان کم، آستانه Hold را کاهش می‌دهیم (ریسک‌پذیرتر می‌شویم)
            self.thresholds[1] -= 0.05
        
        # ثبت زمان بروزرسانی
        self.last_update = datetime.now()
        
        # ذخیره آستانه‌های جدید
        self.save_thresholds()
        
        print(f"Thresholds updated: Sell={self.thresholds[0]:.2f}, Hold={self.thresholds[1]:.2f}, Buy={self.thresholds[2]:.2f}")
        print(f"Market state: {self.market_state}, Volatility: {self.volatility_level}")
    
    def get_threshold(self, class_index):
        """دریافت آستانه برای یک کلاس خاص"""
        return self.thresholds.get(class_index, 0.5)
    
    def record_signal(self, signal, confidence, market_data=None):
        """
        ثبت سیگنال برای تحلیل آینده
        
        Args:
            signal: سیگنال (0=Sell, 1=Hold, 2=Buy)
            confidence: اطمینان سیگنال
            market_data: داده‌های بازار در زمان سیگنال
        """
        signal_data = {
            'signal': signal,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'market_state': self.market_state,
            'volatility_level': self.volatility_level
        }
        
        # افزودن داده‌های بازار
        if market_data:
            for key in ['close', 'rsi14', 'macd', 'bb_width']:
                if key in market_data:
                    signal_data[key] = market_data[key]
        
        self.historical_signals.append(signal_data)
        
        # نگهداری فقط 1000 سیگنال آخر
        if len(self.historical_signals) > 1000:
            self.historical_signals = self.historical_signals[-1000:]
    
    def analyze_performance(self, recent_results=None):
        """
        تحلیل عملکرد سیگنال‌های اخیر و تنظیم آستانه‌ها
        
        Args:
            recent_results: نتایج معاملات اخیر
        """
        if not self.historical_signals or not recent_results:
            return
            
        # محاسبه نرخ موفقیت سیگنال‌ها
        success_rates = {0: 0.0, 1: 0.0, 2: 0.0}
        signal_counts = {0: 0, 1: 0, 2: 0}
        
        for result in recent_results:
            if 'signal' in result and 'profit' in result:
                signal = result['signal']
                profit = result['profit']
                
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                # سیگنال موفق
                if (signal == 0 and profit < 0) or (signal == 2 and profit > 0) or (signal == 1 and abs(profit) < 0.005):
                    success_rates[signal] = success_rates.get(signal, 0) + 1
        
        # محاسبه نرخ موفقیت
        for signal in success_rates:
            if signal_counts[signal] > 0:
                success_rates[signal] /= signal_counts[signal]
        
        # تنظیم آستانه‌ها براساس نرخ موفقیت
        for signal in success_rates:
            if signal_counts[signal] >= 10:  # حداقل 10 سیگنال برای اعتبارسنجی
                success_rate = success_rates[signal]
                
                if success_rate < 0.4:
                    # نرخ موفقیت پایین، آستانه را افزایش می‌دهیم
                    self.thresholds[signal] = min(0.7, self.thresholds[signal] + 0.05)
                elif success_rate > 0.7:
                    # نرخ موفقیت بالا، آستانه را کاهش می‌دهیم
                    self.thresholds[signal] = max(0.2, self.thresholds[signal] - 0.05)
        
        # ذخیره آستانه‌های جدید
        self.save_thresholds()
        
        print("Thresholds adjusted based on performance:")
        print(f"Sell={self.thresholds[0]:.2f}, Hold={self.thresholds[1]:.2f}, Buy={self.thresholds[2]:.2f}")
    
    def save_thresholds(self, file_path='model/adaptive_thresholds.pkl'):
        """ذخیره آستانه‌ها"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            'thresholds': self.thresholds,
            'market_state': self.market_state,
            'volatility_level': self.volatility_level,
            'last_update': self.last_update,
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_thresholds(self, file_path='model/adaptive_thresholds.pkl'):
        """بارگذاری آستانه‌ها"""
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            self.thresholds = data.get('thresholds', self.thresholds)
            self.market_state = data.get('market_state', 'neutral')
            self.volatility_level = data.get('volatility_level', 'normal')
            self.last_update = data.get('last_update')
            
            if self.last_update:
                elapsed = datetime.now() - self.last_update
                print(f"Loaded thresholds last updated {elapsed.days} days, {elapsed.seconds//3600} hours ago")
                
            print(f"Current thresholds: Sell={self.thresholds[0]:.2f}, Hold={self.thresholds[1]:.2f}, Buy={self.thresholds[2]:.2f}")
            return True
        
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return False