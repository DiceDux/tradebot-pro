"""
هماهنگ‌کننده معاملات با استفاده از مدل‌های متخصص
این کلاس مسئول دریافت فیچرها، اجرای مدل‌ها و ایجاد سیگنال‌های معاملاتی است
"""
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from feature_store.feature_database import FeatureDatabase
from meta_model.model_combiner import ModelCombiner
from specialist_models.moving_averages_model import MovingAveragesModel
from specialist_models.oscillators_model import OscillatorsModel
from specialist_models.volatility_model import VolatilityModel
from specialist_models.candlestick_model import CandlestickModel
from specialist_models.news_model import NewsModel
from meta_model.adaptive_thresholds import AdaptiveThresholds

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
            'news': NewsModel().load()
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
        return self
        
    def stop(self):
        """توقف فرآیند معاملاتی"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Trading orchestrator stopped")
        print(f"{Colors.YELLOW}Trading system stopped{Colors.RESET}")
        return self
        
    def _run_trading_loop(self):
        """حلقه اصلی معاملات"""
        while self.running:
            try:
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                    
                # انتظار تا آپدیت بعدی
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                print(f"{Colors.RED}Error in trading loop: {e}{Colors.RESET}")
                time.sleep(5)  # انتظار کوتاه در صورت خطا
                
    def _process_symbol(self, symbol):
        """پردازش یک نماد"""
        # بررسی زمان آخرین سیگنال (برای جلوگیری از سیگنال‌های مکرر)
        current_time = datetime.now().timestamp()
        if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
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
                
                print("="*50 + "\n")
                
                # ثبت در دیتابیس
                self._record_trade_signal(symbol, adjusted_decision, confidence, meta_proba[0])
                
            except Exception as e:
                logger.error(f"Error in meta model prediction: {e}", exc_info=True)
                
    def _calculate_market_volatility(self, symbol, features):
        """محاسبه نوسان بازار"""
        try:
            # استفاده از پهنای باند بولینگر یا ATR به عنوان معیار نوسان
            if 'bb_width' in features.columns:
                return features['bb_width'].iloc[0] / 1000  # نرمال‌سازی
            elif 'atr14' in features.columns:
                return features['atr14'].iloc[0] / 100  # نرمال‌سازی
            return 0.05  # مقدار پیش‌فرض
        except:
            return 0.05  # مقدار پیش‌فرض
    
    def _calculate_market_trend(self, symbol, features):
        """محاسبه روند بازار"""
        try:
            # استفاده از MACD یا میانگین‌های متحرک برای تشخیص روند
            if 'macd' in features.columns:
                return features['macd'].iloc[0]
            elif 'ema20' in features.columns and 'ema50' in features.columns:
                return (features['ema20'].iloc[0] / features['ema50'].iloc[0]) - 1
            return 0  # خنثی
        except:
            return 0  # خنثی
                
    def _record_trade_signal(self, symbol, decision, confidence, probabilities):
        """ثبت سیگنال معاملاتی در دیتابیس"""
        try:
            self.db.insert_trade_signal(
                symbol=symbol,
                timestamp=datetime.now(),
                decision=decision,
                confidence=confidence,
                sell_prob=probabilities[0],
                hold_prob=probabilities[1],
                buy_prob=probabilities[2]
            )
        except Exception as e:
            logger.error(f"Error recording trade signal: {e}", exc_info=True)
