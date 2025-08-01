"""
هماهنگ‌کننده معاملات با استفاده از مدل‌های متخصص
این کلاس مسئول دریافت فیچرها، اجرای مدل‌ها و ایجاد سیگنال‌های معاملاتی است
"""
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from feature_store.feature_database import FeatureDatabase
from meta_model.model_combiner import ModelCombiner
from specialist_models.moving_averages_model import MovingAveragesModel
from specialist_models.oscillators_model import OscillatorsModel
from specialist_models.volatility_model import VolatilityModel
from specialist_models.candlestick_model import CandlestickModel
from specialist_models.news_model import NewsModel
from specialist_models.advanced_patterns_model import AdvancedPatternsModel
from meta_model.adaptive_thresholds import AdaptiveThresholds
from trading.demo_account import DemoAccount
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager

# تنظیم لاگر
logger = logging.getLogger("trading_orchestrator")
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/trading.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# تنظیم رنگ‌های ANSI برای خروجی کنسول
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

class TradingOrchestrator:
    """هماهنگ‌کننده معاملات با استفاده از مدل‌های متخصص"""
    
    def __init__(self, symbols=['BTCUSDT'], update_interval=60):
        self.symbols = symbols
        self.update_interval = update_interval
        self.db = FeatureDatabase()
        self.running = False
        self.thread = None
        self.specialist_models = {}
        self.meta_model = None
        self.adaptive_thresholds = AdaptiveThresholds()
        self.last_signal_time = {}  # زمان آخرین سیگنال برای هر نماد
        self.signal_cooldown = 300  # مدت زمان انتظار بین سیگنال‌ها (به ثانیه)
        
        # مدیریت حساب دمو
        self.demo_account = DemoAccount(10000.0)  # 10,000 USDT موجودی اولیه
        
        # مدیریت موقعیت‌های معاملاتی
        self.position_manager = PositionManager()
        
        # مدیریت ریسک
        self.risk_manager = RiskManager()
        
        # تنظیمات TP/SL
        self.tp_levels = [1.01, 1.02, 1.03, 1.05]  # TP1, TP2, TP3, TP4 (درصد)
        self.tp_volumes = [0.25, 0.25, 0.25, 0.25]  # حجم هر TP (درصد)
        self.sl_initial = 0.99  # SL اولیه (درصد)
        self.trailing_activation = 1.01  # فعال‌سازی تریلینگ استاپ (درصد)
        self.trailing_distance = 0.005  # فاصله تریلینگ استاپ (درصد)
        
    def set_demo_balance(self, balance):
        """تنظیم موجودی حساب دمو"""
        self.demo_account = DemoAccount(balance)
        logger.info(f"Demo account balance set to {balance} USDT")
        print(f"{Colors.CYAN}Demo account balance set to {balance} USDT{Colors.RESET}")
        
    def initialize(self):
        """بارگیری مدل‌های ذخیره شده"""
        logger.info(f"Initializing trading orchestrator for {self.symbols}")
        print(f"{Colors.BLUE}Initializing trading system for {', '.join(self.symbols)}{Colors.RESET}")
        
        # بارگیری مدل‌های متخصص
        self.specialist_models = {
            'moving_averages': MovingAveragesModel().load(),
            'oscillators': OscillatorsModel().load(),
            'volatility': VolatilityModel().load(),
            'candlestick': CandlestickModel().load(),
            'news': NewsModel().load(),
            'advanced_patterns': AdvancedPatternsModel().load()
        }
        
        # بارگیری مدل متا
        self.meta_model = ModelCombiner([]).load()
        
        # بررسی بارگیری موفق مدل‌ها
        successfully_loaded = []
        for name, model in self.specialist_models.items():
            if model.model is not None:
                successfully_loaded.append(name)
            else:
                logger.warning(f"Failed to load {name} model")
                
        if self.meta_model.model is None:
            logger.warning("Failed to load meta model")
            print(f"{Colors.YELLOW}Warning: Meta model not loaded. Using specialist models only.{Colors.RESET}")
        
        if successfully_loaded:
            logger.info(f"Successfully loaded models: {', '.join(successfully_loaded)}")
            print(f"{Colors.GREEN}Successfully loaded models: {', '.join(successfully_loaded)}{Colors.RESET}")
        else:
            logger.error("No models could be loaded")
            print(f"{Colors.RED}Error: No models could be loaded. Trading cannot start.{Colors.RESET}")
            return False
            
        # برای هر نماد، تنظیم زمان آخرین سیگنال
        for symbol in self.symbols:
            self.last_signal_time[symbol] = datetime.now().timestamp() - self.signal_cooldown
            
            # بررسی قیمت اولیه
            try:
                candles = self.db.get_latest_features(symbol)
                if candles is not None and 'close' in candles.columns:
                    price = candles['close'].iloc[0]
                    self.position_manager.update_market_price(symbol, price)
                    logger.info(f"Initial price for {symbol}: {price}")
            except Exception as e:
                logger.error(f"Error getting initial price for {symbol}: {e}")
        
        return self
        
    def start(self):
        """شروع فرآیند معاملاتی در یک ترد جداگانه"""
        if self.running:
            logger.warning("Trading orchestrator is already running")
            print(f"{Colors.YELLOW}Trading system is already running{Colors.RESET}")
            return self
            
        self.running = True
        self.thread = threading.Thread(target=self._run_trading_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Trading orchestrator started")
        print(f"{Colors.GREEN}Trading system started. Press Ctrl+C to stop.{Colors.RESET}")
        
        # نمایش وضعیت حساب دمو
        self.demo_account.print_status()
        
        return self
        
    def stop(self):
        """توقف فرآیند معاملاتی"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Trading orchestrator stopped")
        print(f"{Colors.YELLOW}Trading system stopped{Colors.RESET}")
        
        # بستن تمام موقعیت‌های باز
        self.close_all_positions()
        
        # نمایش وضعیت نهایی حساب دمو
        self.demo_account.print_status()
        
        return self
    
    def close_all_positions(self):
        """بستن تمام موقعیت‌های باز"""
        positions = self.position_manager.get_all_positions()
        if not positions:
            print(f"{Colors.YELLOW}No open positions to close{Colors.RESET}")
            return
            
        print(f"{Colors.YELLOW}Closing all open positions...{Colors.RESET}")
        
        for symbol, position in positions.items():
            try:
                # دریافت قیمت فعلی
                candles = self.db.get_latest_features(symbol)
                if candles is not None and 'close' in candles.columns:
                    price = candles['close'].iloc[0]
                    
                    if position['type'] == 'buy':
                        profit_loss = (price - position['entry_price']) * position['size']
                    else:
                        profit_loss = (position['entry_price'] - price) * position['size']
                    
                    # بستن موقعیت
                    self.position_manager.close_position(symbol)
                    
                    # به‌روزرسانی حساب دمو
                    self.demo_account.update_balance(profit_loss)
                    
                    logger.info(f"Closed position for {symbol} at {price}, P&L: {profit_loss:.2f} USDT")
                    print(f"{Colors.CYAN}Closed position for {symbol} at {price}, P&L: {profit_loss:.2f} USDT{Colors.RESET}")
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
        
    def _run_trading_loop(self):
        """حلقه اصلی معاملات"""
        while self.running:
            try:
                # به‌روزرسانی و بررسی موقعیت‌های فعلی
                self._check_active_positions()
                
                # بررسی سیگنال‌های جدید
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                    
                # انتظار تا آپدیت بعدی
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                print(f"{Colors.RED}Error in trading loop: {e}{Colors.RESET}")
                time.sleep(5)  # انتظار کوتاه در صورت خطا
    
    def _check_active_positions(self):
        """بررسی و به‌روزرسانی موقعیت‌های فعال"""
        positions = self.position_manager.get_all_positions()
        
        for symbol, position in positions.items():
            try:
                # دریافت آخرین قیمت
                candles = self.db.get_latest_features(symbol)
                if candles is None or 'close' not in candles.columns:
                    continue
                    
                price = candles['close'].iloc[0]
                self.position_manager.update_market_price(symbol, price)
                
                # محاسبه سود/زیان فعلی
                if position['type'] == 'buy':
                    current_profit = (price / position['entry_price'] - 1) * 100
                else:
                    current_profit = (position['entry_price'] / price - 1) * 100
                    
                # بررسی تریلینگ استاپ
                if position['trailing_stop_active']:
                    # بررسی آیا قیمت از SL تریلینگ عبور کرده است
                    if (position['type'] == 'buy' and price <= position['stop_loss']) or \
                       (position['type'] == 'sell' and price >= position['stop_loss']):
                        
                        # بستن موقعیت با SL
                        self._close_position_with_sl(symbol, position, price)
                        
                elif current_profit >= self.trailing_activation * 100:
                    # فعال‌سازی تریلینگ استاپ
                    if position['type'] == 'buy':
                        new_sl = price * (1 - self.trailing_distance)
                        if new_sl > position['stop_loss']:
                            self.position_manager.update_stop_loss(symbol, new_sl, True)
                            logger.info(f"Trailing stop activated for {symbol} at {new_sl:.2f}")
                    else:
                        new_sl = price * (1 + self.trailing_distance)
                        if new_sl < position['stop_loss']:
                            self.position_manager.update_stop_loss(symbol, new_sl, True)
                            logger.info(f"Trailing stop activated for {symbol} at {new_sl:.2f}")
                
                # بررسی سطوح TP
                tp_levels = position['tp_levels']
                tp_volumes = position['tp_volumes']
                
                for i, (tp, volume) in enumerate(zip(tp_levels, tp_volumes)):
                    if volume <= 0:  # این سطح قبلاً اجرا شده است
                        continue
                        
                    if position['type'] == 'buy' and price >= tp:
                        # اجرای TP جزئی
                        self._execute_partial_tp(symbol, position, price, i)
                    elif position['type'] == 'sell' and price <= tp:
                        # اجرای TP جزئی
                        self._execute_partial_tp(symbol, position, price, i)
                
                # بررسی SL
                if not position['trailing_stop_active']:
                    if (position['type'] == 'buy' and price <= position['stop_loss']) or \
                       (position['type'] == 'sell' and price >= position['stop_loss']):
                        
                        # بستن موقعیت با SL
                        self._close_position_with_sl(symbol, position, price)
                        
            except Exception as e:
                logger.error(f"Error processing active position for {symbol}: {e}")
                
    def _execute_partial_tp(self, symbol, position, price, tp_index):
        """اجرای TP جزئی"""
        tp_size = position['size'] * position['tp_volumes'][tp_index]
        
        if position['type'] == 'buy':
            profit = (price - position['entry_price']) * tp_size
        else:
            profit = (position['entry_price'] - price) * tp_size
            
        # محاسبه کارمزد
        commission = self._calculate_commission(price, tp_size)
        net_profit = profit - commission
        
        # به‌روزرسانی حساب دمو
        self.demo_account.update_balance(net_profit)
        
        # به‌روزرسانی موقعیت
        self.position_manager.execute_partial_tp(symbol, tp_index, tp_size, price)
        
        # حرکت SL به نقطه ورود یا بالاتر پس از اجرای TP1
        if tp_index == 0:  # TP1
            if position['type'] == 'buy':
                new_sl = position['entry_price']  # حداقل در نقطه سر به سر
            else:
                new_sl = position['entry_price']
            self.position_manager.update_stop_loss(symbol, new_sl)
            
        logger.info(f"Executed TP{tp_index+1} for {symbol} at {price:.2f}, size: {tp_size:.6f}, profit: {net_profit:.2f} USDT")
        print(f"{Colors.GREEN}TP{tp_index+1} hit for {symbol} at {price:.2f}, profit: {net_profit:.2f} USDT (fee: {commission:.2f}){Colors.RESET}")
                
    def _close_position_with_sl(self, symbol, position, price):
        """بستن موقعیت با استاپ لاس"""
        if position['type'] == 'buy':
            profit_loss = (price - position['entry_price']) * position['size']
        else:
            profit_loss = (position['entry_price'] - price) * position['size']
            
        # محاسبه کارمزد
        commission = self._calculate_commission(price, position['size'])
        net_profit_loss = profit_loss - commission
        
        # به‌روزرسانی حساب دمو
        self.demo_account.update_balance(net_profit_loss)
        
        # بستن موقعیت
        self.position_manager.close_position(symbol)
        
        logger.info(f"Stop loss triggered for {symbol} at {price:.2f}, P&L: {net_profit_loss:.2f} USDT")
        print(f"{Colors.RED}Stop loss hit for {symbol} at {price:.2f}, P&L: {net_profit_loss:.2f} USDT (fee: {commission:.2f}){Colors.RESET}")
                
    def _process_symbol(self, symbol):
        """پردازش یک نماد"""
        # بررسی زمان آخرین سیگنال (برای جلوگیری از سیگنال‌های مکرر)
        current_time = datetime.now().timestamp()
        if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
            return
            
        # بررسی آیا قبلاً موقعیتی برای این نماد باز شده است
        if self.position_manager.has_position(symbol):
            return
            
        # دریافت آخرین فیچرها
        features = self.db.get_latest_features(symbol)
        
        if features is None or features.empty:
            logger.warning(f"No features available for {symbol}")
            return
            
        # محاسبه ویژگی‌های شرایط بازار
        market_volatility = self._calculate_market_volatility(symbol, features)
        market_trend = self._calculate_market_trend(symbol, features)
        
        # به‌روزرسانی شرایط بازار در آستانه‌های تطبیقی
        self.adaptive_thresholds.update_market_condition(symbol, market_volatility, market_trend)
        
        # پیش‌بینی با مدل‌های متخصص
        specialist_predictions = {}
        specialist_probas = {}
        
        for name, model in self.specialist_models.items():
            if model.model is not None:
                # انتخاب فیچرهای مورد نیاز این مدل متخصص
                required_features = model.get_required_features()
                available_features = [f for f in required_features if f in features.columns]
                
                if available_features:
                    X = features[available_features]
                    try:
                        pred, proba = model.predict(X)
                        specialist_predictions[name] = pred[0]  # فقط نمونه اول
                        specialist_probas[name] = proba[0]  # احتمالات کلاس‌ها
                        
                        # نمایش نتیجه پیش‌بینی این متخصص
                        class_names = {0: "Sell", 1: "Hold", 2: "Buy"}
                        predicted_class = class_names[pred[0]]
                        confidence = proba[0][pred[0]] * 100
                        logger.info(f"[{symbol}] {name} model predicts: {predicted_class} with {confidence:.1f}% confidence")
                    except Exception as e:
                        logger.error(f"Error in {name} prediction: {e}", exc_info=True)
        
        # ساخت فیچرهای ورودی برای مدل متا
        if specialist_probas:
            meta_features = pd.DataFrame()
            
            # تبدیل احتمالات مدل‌های متخصص به فیچرهای مدل متا
            for name, probas in specialist_probas.items():
                for j, prob in enumerate(probas):
                    meta_features.at[0, f"{name}_class{j}"] = prob
                    
            # پیش‌بینی با مدل متا
            try:
                meta_pred, meta_proba = self.meta_model.predict(meta_features)
                
                # اعمال آستانه‌های تطبیقی
                adjusted_decision, confidence = self.adaptive_thresholds.apply(
                    symbol, 
                    meta_pred[0], 
                    meta_proba[0]
                )
                
                # نمایش تصمیم نهایی
                class_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
                decision_text = class_names[adjusted_decision]
                
                # رنگ متناسب با تصمیم
                decision_colors = {0: Colors.RED, 1: Colors.YELLOW, 2: Colors.GREEN}
                color_code = decision_colors[adjusted_decision]
                
                # به‌روزرسانی زمان آخرین سیگنال
                self.last_signal_time[symbol] = current_time
                
                # ثبت در لاگ
                logger.info(f"[{symbol}] Decision: {decision_text} with {confidence:.1f}% confidence. Probabilities: SELL={meta_proba[0][0]:.3f}, HOLD={meta_proba[0][1]:.3f}, BUY={meta_proba[0][2]:.3f}")
                
                # نمایش در کنسول
                print("\n" + "="*50)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRADE SIGNAL FOR {symbol}:")
                print(f"DECISION: {color_code}{decision_text}{Colors.RESET} with {confidence:.1f}% confidence")
                
                # نمایش احتمالات
                print("\nClass probabilities:")
                for i, prob in enumerate(meta_proba[0]):
                    decision_name = class_names[i]
                    decision_color = decision_colors[i]
                    print(f"  {decision_color}{decision_name}{Colors.RESET}: {prob*100:.1f}%")
                
                # نمایش شرایط بازار
                print(f"\nMarket conditions:")
                print(f"  Volatility: {market_volatility:.4f}")
                print(f"  Trend: {market_trend:.4f}")
                
                # نمایش پیش‌بینی‌های متخصصان
                print("\nSpecialist predictions:")
                for name, pred in specialist_predictions.items():
                    probas = specialist_probas[name]
                    decision_name = class_names[pred]
                    decision_color = decision_colors[pred]
                    confidence = probas[pred] * 100
                    print(f"  {name}: {decision_color}{decision_name}{Colors.RESET} ({confidence:.1f}%)")
                
                # قیمت فعلی
                current_price = features['close'].iloc[0]
                print(f"\nCurrent price: {current_price:.2f}")
                
                # مدیریت اجرای سیگنال
                if adjusted_decision != 1 and confidence > 
