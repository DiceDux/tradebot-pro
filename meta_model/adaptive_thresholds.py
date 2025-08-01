"""
کلاس تنظیم آستانه‌های تطبیقی برای تصمیم‌گیری معاملاتی
این کلاس آستانه‌های تصمیم‌گیری را بر اساس شرایط بازار تنظیم می‌کند
"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import json
import os
import logging

logger = logging.getLogger("adaptive_thresholds")

class AdaptiveThresholds:
    """کلاس تنظیم آستانه‌های تطبیقی برای تصمیم‌گیری معاملاتی"""
    
    def __init__(self):
        self.base_threshold = 0.4  # آستانه پایه برای اطمینان
        self.market_conditions = {}  # شرایط بازار برای هر نماد
        self.history = {}  # تاریخچه تصمیمات و اطمینان برای هر نماد
        self.history_file = "model/threshold_history.json"
        self.max_history_size = 100  # تعداد تصمیمات گذشته برای ذخیره
        
        # بارگیری تاریخچه اگر وجود دارد
        self._load_history()
        
    def apply(self, symbol, prediction, probabilities):
        """
        اعمال آستانه‌های تطبیقی بر اساس شرایط بازار و اطمینان مدل
        
        Args:
            symbol: نماد معاملاتی
            prediction: پیش‌بینی اولیه (0: فروش، 1: نگهداری، 2: خرید)
            probabilities: احتمالات هر کلاس [p_sell, p_hold, p_buy]
            
        Returns:
            (final_decision, confidence): تصمیم نهایی و درصد اطمینان
        """
        # محاسبه اطمینان از پیش‌بینی فعلی
        confidence = probabilities[prediction] * 100
        
        # آستانه اطمینان پویا بر اساس وضعیت بازار
        threshold = self._get_threshold_for_symbol(symbol)
        
        # بررسی پراکندگی احتمالات
        prob_spread = max(probabilities) - min(probabilities)
        
        # اگر احتمالات خیلی به هم نزدیک هستند، تبدیل به نگهداری می‌شود
        if prob_spread < 0.15:  # اختلاف کمتر از 15%
            adjusted_decision = 1  # تبدیل به نگهداری
            logger.info(f"Probabilities too close ({prob_spread:.2f}), adjusted decision to HOLD")
        # اگر اطمینان کمتر از آستانه است، به نگهداری تبدیل می‌شود
        elif confidence < threshold and prediction != 1:  # اگر پیش‌بینی نگهداری نیست
            adjusted_decision = 1  # تبدیل به نگهداری
            logger.info(f"Low confidence ({confidence:.1f}% < {threshold:.1f}%), adjusted decision to HOLD")
        else:
            adjusted_decision = prediction
            
        # ثبت تصمیم در تاریخچه
        self._update_history(symbol, adjusted_decision, confidence, probabilities)
        
        return adjusted_decision, confidence
        
    def _get_threshold_for_symbol(self, symbol):
        """
        دریافت آستانه مناسب بر اساس شرایط نماد و تاریخچه
        """
        # آستانه پایه
        threshold = self.base_threshold * 100  # تبدیل به درصد
        
        # تنظیم بر اساس نوسانات بازار (اگر موجود باشد)
        if symbol in self.market_conditions and 'volatility' in self.market_conditions[symbol]:
            volatility = self.market_conditions[symbol]['volatility']
            # در بازارهای با نوسان بالا، آستانه اطمینان را افزایش می‌دهیم
            threshold += volatility * 10
        
        # تنظیم بر اساس تاریخچه تصمیمات
        if symbol in self.history and len(self.history[symbol]) > 10:
            recent_history = self.history[symbol][-10:]
            # میانگین اطمینان تصمیمات اخیر
            avg_confidence = np.mean([item['confidence'] for item in recent_history])
            
            # اگر میانگین اطمینان بالا بوده، آستانه را کاهش می‌دهیم
            if avg_confidence > 70:
                threshold *= 0.9  # کاهش 10%
            # اگر میانگین اطمینان پایین بوده، آستانه را افزایش می‌دهیم
            elif avg_confidence < 50:
                threshold *= 1.1  # افزایش 10%
        
        return min(80, max(35, threshold))  # محدود کردن بین 35% تا 80%
        
    def update_market_condition(self, symbol, volatility, trend, volume=None):
        """
        به‌روزرسانی شرایط بازار برای یک نماد
        """
        self.market_conditions[symbol] = {
            'volatility': volatility,
            'trend': trend,
            'volume': volume,
            'updated_at': datetime.now().isoformat()
        }
        
    def get_market_condition(self, symbol):
        """دریافت شرایط بازار برای یک نماد"""
        return self.market_conditions.get(symbol, {})
        
    def _update_history(self, symbol, decision, confidence, probabilities):
        """ثبت تصمیم در تاریخچه"""
        if symbol not in self.history:
            self.history[symbol] = []
            
        # اضافه کردن تصمیم جدید
        self.history[symbol].append({
            'timestamp': datetime.now().isoformat(),
            'decision': int(decision),
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities]
        })
        
        # محدود کردن تعداد تصمیمات ذخیره شده
        if len(self.history[symbol]) > self.max_history_size:
            self.history[symbol] = self.history[symbol][-self.max_history_size:]
            
        # ذخیره تاریخچه
        self._save_history()
        
    def _save_history(self):
        """ذخیره تاریخچه تصمیمات در فایل"""
        try:
            # ساخت دایرکتوری اگر وجود ندارد
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Error saving threshold history: {e}")
    
    def _load_history(self):
        """بارگیری تاریخچه تصمیمات از فایل"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading threshold history: {e}")
