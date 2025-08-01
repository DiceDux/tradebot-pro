"""
کلاس مدیریت ریسک معاملات
"""
import logging
import time
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger("risk_manager")

class RiskManager:
    """کلاس مدیریت ریسک معاملات"""
    
    def __init__(self):
        self.base_risk_percent = 0.01  # ریسک پایه 1% از کل سرمایه
        self.max_risk_percent = 0.03   # حداکثر ریسک 3% از کل سرمایه
        self.max_open_positions = 5    # حداکثر تعداد موقعیت‌های باز همزمان
        self.max_daily_trades = 20     # حداکثر تعداد معاملات روزانه
        self.max_daily_loss_percent = 0.05  # حداکثر زیان روزانه 5%
        self.min_reward_risk_ratio = 1.5  # حداقل نسبت پاداش به ریسک
        self.cooldown_periods = {}     # دوره‌های استراحت برای نمادها
        self.market_state_factors = {} # ضرایب وضعیت بازار برای هر نماد
        self.symbol_performance = {}   # عملکرد تاریخی برای هر نماد
        self.daily_trades = {}         # تعداد معاملات روزانه
        self.daily_pnl = {}            # سود و زیان روزانه
        self.recent_trades = []        # معاملات اخیر
        self.confidence_factors = {    # ضرایب اطمینان برای انواع سیگنال
            0: 0.9,  # Sell
            1: 0.0,  # Hold
            2: 0.9   # Buy
        }
    
    def get_position_risk(self, symbol, decision_type):
        """
        محاسبه درصد ریسک برای یک موقعیت بر اساس نوع تصمیم و شرایط
        
        Args:
            symbol: نماد معاملاتی
            decision_type: نوع تصمیم (0=Sell, 1=Hold, 2=Buy)
            
        Returns:
            float: درصد ریسک (بین 0 تا max_risk_percent)
        """
        # بررسی دوره استراحت
        if self._is_in_cooldown(symbol):
            return 0.0
            
        # بررسی تعداد معاملات روزانه
        today = datetime.now().date()
        today_str = today.isoformat()
        if today_str in self.daily_trades and self.daily_trades[today_str] >= self.max_daily_trades:
            logger.warning(f"Max daily trades ({self.max_daily_trades}) reached for {today_str}")
            return 0.0
            
        # بررسی زیان روزانه
        if today_str in self.daily_pnl and self.daily_pnl[today_str] < -self.max_daily_loss_percent:
            logger.warning(f"Max daily loss ({self.max_daily_loss_percent*100}%) reached for {today_str}")
            return 0.0
            
        # ریسک پایه
        risk = self.base_risk_percent
        
        # تعدیل بر اساس نوع تصمیم
        risk *= self.confidence_factors.get(decision_type, 0.5)
        
        # تعدیل بر اساس وضعیت بازار
        market_factor = self.market_state_factors.get(symbol, 1.0)
        risk *= market_factor
        
        # تعدیل بر اساس عملکرد قبلی در این نماد
        performance_factor = self._calculate_performance_factor(symbol)
        risk *= performance_factor
        
        # محدودسازی به حداکثر ریسک مجاز
        risk = min(risk, self.max_risk_percent)
        
        logger.info(f"Calculated risk for {symbol}: {risk*100:.2f}% (market factor: {market_factor:.2f}, performance factor: {performance_factor:.2f})")
        
        return risk
    
    def update_market_state(self, symbol, volatility, trend):
        """
        به‌روزرسانی وضعیت بازار برای یک نماد
        
        Args:
            symbol: نماد معاملاتی
            volatility: میزان نوسان بازار (0 تا 1)
            trend: قدرت روند (-1 تا 1)
        """
        # محاسبه فاکتور وضعیت بازار
        # - کاهش ریسک در نوسان بالا
        # - افزایش ریسک در روند قوی
        
        # کاهش ریسک در نوسان بالا (فاکتور بین 0.5 تا 1)
        volatility_factor = max(0.5, 1.0 - volatility)
        
        # افزایش ریسک در روند قوی (فاکتور بین 0.8 تا 1.2)
        trend_factor = 1.0 + (abs(trend) * 0.2)
        
        # فاکتور نهایی
        market_factor = volatility_factor * trend_factor
        
        # محدودسازی به بازه منطقی
        market_factor = max(0.5, min(1.5, market_factor))
        
        self.market_state_factors[symbol] = market_factor
        logger.debug(f"Updated market state for {symbol}: volatility={volatility:.4f}, trend={trend:.4f}, factor={market_factor:.4f}")
    
    def register_trade_result(self, symbol, entry_time, exit_time, position_type, profit_loss, risk_amount):
        """
        ثبت نتیجه یک معامله برای تحلیل بعدی
        
        Args:
            symbol: نماد معاملاتی
            entry_time: زمان ورود به موقعیت
            exit_time: زمان خروج از موقعیت
            position_type: نوع موقعیت (buy/sell)
            profit_loss: مقدار سود یا زیان
            risk_amount: مقدار در معرض ریسک
        """
        # ایجاد رکورد معامله
        trade = {
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position_type': position_type,
            'profit_loss': profit_loss,
            'risk_amount': risk_amount,
            'reward_risk_ratio': abs(profit_loss / risk_amount) if risk_amount != 0 else 0.0,
            'is_profit': profit_loss > 0
        }
        
        # افزودن به لیست معاملات اخیر
        self.recent_trades.append(trade)
        if len(self.recent_trades) > 100:  # نگهداری حداکثر 100 معامله اخیر
            self.recent_trades.pop(0)
        
        # به‌روزرسانی عملکرد نماد
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_reward_risk': 0.0
            }
        
        perf = self.symbol_performance[symbol]
        perf['total_trades'] += 1
        perf['profitable_trades'] += 1 if profit_loss > 0 else 0
        perf['total_pnl'] += profit_loss
        perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']
        
        # محاسبه میانگین متحرک نسبت پاداش به ریسک
        rr_ratios = [t['reward_risk_ratio'] for t in self.recent_trades if t['symbol'] == symbol]
        perf['avg_reward_risk'] = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
        
        # به‌روزرسانی آمار روزانه
        today = datetime.now().date().isoformat()
        
        if today not in self.daily_trades:
            self.daily_trades[today] = 0
        self.daily_trades[today] += 1
        
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0.0
        self.daily_pnl[today] += profit_loss
        
        # تنظیم دوره استراحت برای نمادهایی با زیان متوالی
        symbol_trades = [t for t in self.recent_trades if t['symbol'] == symbol][-3:]  # آخرین 3 معامله
        if len(symbol_trades) >= 3 and all(not t['is_profit'] for t in symbol_trades):
            self._set_cooldown(symbol, hours=6)  # 6 ساعت استراحت بعد از 3 زیان متوالی
            logger.warning(f"Setting 6-hour cooldown for {symbol} after 3 consecutive losses")
    
    def is_trade_allowed(self, symbol, position_type, stop_loss, take_profit):
        """
        بررسی آیا معامله مجاز است یا خیر
        
        Args:
            symbol: نماد معاملاتی
            position_type: نوع موقعیت (buy/sell)
            stop_loss: قیمت استاپ لاس
            take_profit: قیمت تیک پرافیت
            
        Returns:
            (bool, str): مجاز بودن و دلیل
        """
        # بررسی دوره استراحت
        if self._is_in_cooldown(symbol):
            return False, f"Symbol {symbol} is in cooldown period"
            
        # بررسی تعداد موقعیت‌های باز
        # توجه: این به داده‌های خارجی نیاز دارد، در اینجا فرض می‌کنیم تعداد موقعیت‌های فعلی به عنوان ورودی داده می‌شود
        open_positions = 0  # این مقدار باید از بیرون تأمین شود
        if open_positions >= self.max_open_positions:
            return False, f"Maximum open positions ({self.max_open_positions}) reached"
        
        # بررسی تعداد معاملات روزانه
        today = datetime.now().date().isoformat()
        if today in self.daily_trades and self.daily_trades[today] >= self.max_daily_trades:
            return False, f"Maximum daily trades ({self.max_daily_trades}) reached"
            
        # بررسی زیان روزانه
        if today in self.daily_pnl and self.daily_pnl[today] < -self.max_daily_loss_percent:
            return False, f"Maximum daily loss ({self.max_daily_loss_percent*100}%) reached"
        
        # محاسبه نسبت پاداش به ریسک
        reward_risk_ratio = self._calculate_reward_risk(position_type, stop_loss, take_profit)
        if reward_risk_ratio < self.min_reward_risk_ratio:
            return False, f"Reward/risk ratio ({reward_risk_ratio:.2f}) below minimum ({self.min_reward_risk_ratio})"
        
        # بررسی عملکرد نماد
        if symbol in self.symbol_performance and self.symbol_performance[symbol]['total_trades'] >= 5:
            win_rate = self.symbol_performance[symbol]['win_rate']
            if win_rate < 0.3:  # اگر نرخ برد کمتر از 30% باشد
                return False, f"Poor performance for {symbol} (win rate: {win_rate*100:.1f}%)"
        
        return True, "Trade allowed"
    
    def _calculate_performance_factor(self, symbol):
        """محاسبه فاکتور عملکرد برای یک نماد"""
        if symbol not in self.symbol_performance or self.symbol_performance[symbol]['total_trades'] < 5:
            return 1.0  # فاکتور خنثی برای نمادهای بدون تاریخچه کافی
        
        perf = self.symbol_performance[symbol]
        
        # فاکتور نرخ برد (0.8 تا 1.2)
        win_factor = 0.8 + perf['win_rate'] * 0.4
        
        # فاکتور نسبت پاداش به ریسک (0.8 تا 1.2)
        rr_factor = min(1.2, max(0.8, perf['avg_reward_risk'] / 2.0))
        
        # فاکتور سودآوری کلی (0.8 تا 1.2)
        profit_factor = 1.0
        if perf['total_pnl'] > 0:
            profit_factor = min(1.2, 1.0 + perf['total_pnl'] / 100.0)
        elif perf['total_pnl'] < 0:
            profit_factor = max(0.8, 1.0 + perf['total_pnl'] / 100.0)
        
        # میانگین وزن‌دار فاکتورها
        return (win_factor * 0.4) + (rr_factor * 0.3) + (profit_factor * 0.3)
    
    def _calculate_reward_risk(self, position_type, stop_loss, take_profit):
        """محاسبه نسبت پاداش به ریسک برای معامله"""
        if position_type == 'buy':
            risk = abs(1 - stop_loss)
            reward = abs(take_profit - 1)
        else:  # 'sell'
            risk = abs(stop_loss - 1)
            reward = abs(1 - take_profit)
            
        if risk == 0:
            return 0.0  # جلوگیری از تقسیم بر صفر
            
        return reward / risk
    
    def _set_cooldown(self, symbol, hours=3):
        """تنظیم دوره استراحت برای یک نماد"""
        expiry = datetime.now() + timedelta(hours=hours)
        self.cooldown_periods[symbol] = expiry
        logger.info(f"Set cooldown for {symbol} until {expiry}")
    
    def _is_in_cooldown(self, symbol):
        """بررسی آیا نماد در دوره استراحت است یا خیر"""
        if symbol not in self.cooldown_periods:
            return False
            
        if datetime.now() > self.cooldown_periods[symbol]:
            # دوره استراحت تمام شده است
            del self.cooldown_periods[symbol]
            return False
            
        return True
    
    def get_stats(self):
        """دریافت آمار کلی مدیریت ریسک"""
        return {
            'total_trades': len(self.recent_trades),
            'profitable_trades': len([t for t in self.recent_trades if t['is_profit']]),
            'symbol_performance': self.symbol_performance,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'cooldown_symbols': {s: expiry.isoformat() for s, expiry in self.cooldown_periods.items() if datetime.now() < expiry}
        }
