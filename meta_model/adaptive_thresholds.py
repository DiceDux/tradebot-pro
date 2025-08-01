"""
کلاس آستانه‌های تطبیقی برای تصمیم‌گیری معاملاتی
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("adaptive_thresholds")

class AdaptiveThresholds:
    """کلاس آستانه‌های تطبیقی برای تصمیم‌گیری معاملاتی"""
    
    def __init__(self):
        # آستانه‌های پایه برای هر کلاس
        self.base_thresholds = {
            0: 0.40,  # آستانه برای فروش
            1: 0.45,  # آستانه برای هولد (بالاتر برای کاهش سیگنال‌های هولد)
            2: 0.40   # آستانه برای خرید
        }
        
        # آستانه‌های فعلی برای هر نماد
        self.current_thresholds = {}
        
        # شرایط بازار برای هر نماد
        self.market_conditions = {}
        
        # تاریخچه سیگنال‌ها
        self.signal_history = []
        
        # ضرایب تنظیم برای شرایط بازار
        self.volatility_factor = 0.2    # تأثیر نوسان بر آستانه‌ها
        self.trend_factor = 0.15        # تأثیر روند بر آستانه‌ها
        
    def update_market_condition(self, symbol, volatility, trend):
        """
        به‌روزرسانی شرایط بازار برای یک نماد
        
        Args:
            symbol: نماد معاملاتی
            volatility: میزان نوسان بازار (0 تا 1)
            trend: قدرت روند (-1 تا 1، منفی=نزولی، مثبت=صعودی)
        """
        if symbol not in self.market_conditions:
            self.market_conditions[symbol] = {
                'volatility': 0.0,
                'trend': 0.0,
                'last_update': datetime.now()
            }
            
        # به‌روزرسانی با میانگین متحرک وزن‌دار
        alpha = 0.7  # وزن داده جدید
        
        prev_volatility = self.market_conditions[symbol]['volatility']
        prev_trend = self.market_conditions[symbol]['trend']
        
        self.market_conditions[symbol]['volatility'] = alpha * volatility + (1 - alpha) * prev_volatility
        self.market_conditions[symbol]['trend'] = alpha * trend + (1 - alpha) * prev_trend
        self.market_conditions[symbol]['last_update'] = datetime.now()
        
        # به‌روزرسانی آستانه‌ها
        self._update_thresholds(symbol)
        
    def _update_thresholds(self, symbol):
        """
        به‌روزرسانی آستانه‌های تصمیم‌گیری برای یک نماد
        
        Args:
            symbol: نماد معاملاتی
        """
        if symbol not in self.current_thresholds:
            self.current_thresholds[symbol] = self.base_thresholds.copy()
            
        volatility = self.market_conditions[symbol]['volatility']
        trend = self.market_conditions[symbol]['trend']
        
        # افزایش آستانه در نوسان بالا (کاهش ریسک)
        volatility_adjustment = volatility * self.volatility_factor
        
        # تنظیم آستانه بر اساس روند (کاهش آستانه خرید در روند صعودی و بالعکس)
        buy_trend_adj = -trend * self.trend_factor
        sell_trend_adj = trend * self.trend_factor
        
        # اعمال تنظیمات به آستانه‌های پایه
        thresholds = self.current_thresholds[symbol]
        
        # بروزرسانی آستانه‌ها با توجه به شرایط بازار
        thresholds[0] = self.base_thresholds[0] + volatility_adjustment + sell_trend_adj  # آستانه فروش
        thresholds[1] = self.base_thresholds[1] + volatility_adjustment                  # آستانه هولد
        thresholds[2] = self.base_thresholds[2] + volatility_adjustment + buy_trend_adj   # آستانه خرید
        
        # محدودسازی آستانه‌ها به بازه منطقی
        for key in thresholds:
            thresholds[key] = max(0.3, min(0.7, thresholds[key]))
            
        logger.debug(f"Updated thresholds for {symbol}: sell={thresholds[0]:.2f}, hold={thresholds[1]:.2f}, buy={thresholds[2]:.2f}")
        
    def apply(self, symbol, prediction, probabilities):
        """
        اعمال آستانه‌های تطبیقی به پیش‌بینی مدل
        
        Args:
            symbol: نماد معاملاتی
            prediction: پیش‌بینی اولیه (0=فروش، 1=هولد، 2=خرید)
            probabilities: احتمالات کلاس‌ها
            
        Returns:
            (adjusted_decision, confidence): تصمیم تعدیل‌شده و میزان اطمینان
        """
        # بررسی وجود آستانه‌های تنظیم‌شده
        if symbol not in self.current_thresholds:
            self.current_thresholds[symbol] = self.base_thresholds.copy()
            
        thresholds = self.current_thresholds[symbol]
        
        # کاهش احتمال سیگنال هولد (افزایش خرید/فروش)
        adjusted_probs = probabilities.copy()
        hold_reduction = 0.1
        hold_diff = adjusted_probs[1] * hold_reduction
        adjusted_probs[1] -= hold_diff
        
        # توزیع مجدد اختلاف بین خرید و فروش
        if trend > 0:  # روند صعودی
            adjusted_probs[2] += hold_diff * 0.7  # 70% به خرید
            adjusted_probs[0] += hold_diff * 0.3  # 30% به فروش
        else:  # روند نزولی یا خنثی
            adjusted_probs[0] += hold_diff * 0.7  # 70% به فروش
            adjusted_probs[2] += hold_diff * 0.3  # 30% به خرید
            
        # نرمال‌سازی مجدد
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        # بررسی آستانه‌ها
        decision = None
        confidence = 0.0
        
        # اولویت با سیگنال‌های خرید و فروش (با آستانه کمتر)
        if adjusted_probs[2] >= thresholds[2] and adjusted_probs[2] > adjusted_probs[0]:
            decision = 2  # خرید
            confidence = adjusted_probs[2] * 100
        elif adjusted_probs[0] >= thresholds[0] and adjusted_probs[0] > adjusted_probs[2]:
            decision = 0  # فروش
            confidence = adjusted_probs[0] * 100
        else:
            # اگر هیچ‌کدام از شرایط بالا برقرار نبود، هولد
            decision = 1  # هولد
            confidence = adjusted_probs[1] * 100
        
        # ثبت در تاریخچه
        self.signal_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'original_prediction': prediction,
            'adjusted_decision': decision,
            'original_probs': probabilities,
            'adjusted_probs': adjusted_probs,
            'confidence': confidence,
            'thresholds': thresholds.copy()
        })
        
        # محدود کردن تاریخچه به 100 آیتم آخر
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
            
        return decision, confidence
        
    def get_current_thresholds(self, symbol):
        """دریافت آستانه‌های فعلی برای یک نماد"""
        if symbol in self.current_thresholds:
            return self.current_thresholds[symbol]
        else:
            return self.base_thresholds
            
    def get_market_condition(self, symbol):
        """دریافت شرایط بازار برای یک نماد"""
        if symbol in self.market_conditions:
            return self.market_conditions[symbol]
        else:
            return {
                'volatility': 0.0,
                'trend': 0.0,
                'last_update': datetime.now()
            }
            
    def get_signal_history(self, limit=10):
        """دریافت تاریخچه سیگنال‌ها"""
        return self.signal_history[-limit:]
