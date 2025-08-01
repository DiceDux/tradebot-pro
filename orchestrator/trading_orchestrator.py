"""
هماهنگ‌کننده سیستم معاملاتی
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import threading

from feature_store.feature_calculator import FeatureCalculator
from feature_store.feature_database import FeatureDatabase
from feature_selection.dynamic_selector import DynamicFeatureSelector
from meta_model.model_combiner import ModelCombiner
from meta_model.adaptive_thresholds import AdaptiveThresholds
from orchestrator.model_pipeline import ModelPipeline

class TradingOrchestrator:
    def __init__(self, symbols=None):
        """
        هماهنگ‌کننده سیستم معاملاتی
        
        Args:
            symbols: لیست نمادهای مورد نظر
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.feature_calculator = None
        self.feature_db = FeatureDatabase()
        self.feature_selector = DynamicFeatureSelector(symbols)
        self.model_pipeline = None
        self.thresholds = AdaptiveThresholds()
        self.running = False
        self.thread = None
        self.last_feature_selection = datetime.now()
    
    def initialize(self):
        """راه‌اندازی اولیه سیستم"""
        # راه‌اندازی محاسبه‌کننده فیچر
        self.feature_calculator = FeatureCalculator(self.symbols, update_interval=1)
        
        # راه‌اندازی پایپلاین مدل
        self.model_pipeline = ModelPipeline(self.symbols)
        
        # بارگذاری مدل‌ها
        self.model_pipeline.load_models()
        
        print("Trading orchestrator initialized successfully")
    
    def start(self):
        """شروع سیستم"""
        if self.running:
            print("System is already running")
            return
            
        # راه‌اندازی محاسبه فیچر
        self.feature_calculator.run()
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("Trading orchestrator started")
    
    def stop(self):
        """توقف سیستم"""
        self.running = False
        if self.feature_calculator:
            self.feature_calculator.stop()
            
        if self.thread:
            self.thread.join(timeout=5)
            
        print("Trading orchestrator stopped")
    
    def _main_loop(self):
        """حلقه اصلی سیستم"""
        try:
            while self.running:
                # بررسی نیاز به انتخاب مجدد فیچرها
                self._check_feature_selection()
                
                # بروزرسانی آستانه‌های تصمیم‌گیری
                self._update_thresholds()
                
                # تحلیل بازار و تولید سیگنال
                for symbol in self.symbols:
                    self._analyze_market(symbol)
                
                # انتظار برای دور بعدی
                time.sleep(10)
                
        except Exception as e:
            print(f"Error in main loop: {e}")
    
    def _check_feature_selection(self, force=False):
        """بررسی نیاز به انتخاب مجدد فیچرها"""
        # بررسی گذشت زمان از آخرین انتخاب فیچر
        elapsed = datetime.now() - self.last_feature_selection
        
        # انتخاب مجدد فیچرها هر 2 ساعت یا در صورت اجبار
        if elapsed > timedelta(hours=2) or force:
            print("Optimizing feature selection for live market conditions...")
            
            # دریافت داده‌های اخیر برای انتخاب فیچر
            print("Fetching recent market data for live feature selection...")
            
            # انتخاب فیچرها
            self.feature_selector.select_features(self.symbols[0], force=True, method='genetic')
            
            # بروزرسانی زمان آخرین انتخاب
            self.last_feature_selection = datetime.now()
            
            # نمایش فیچرهای انتخاب شده
            self.feature_selector.print_selected_features()
    
    def _update_thresholds(self):
        """بروزرسانی آستانه‌های تصمیم‌گیری"""
        # دریافت اطلاعات بازار برای اولین نماد
        market_data = self.feature_db.get_latest_features(self.symbols[0])
        
        if not market_data.empty:
            # تبدیل DataFrame به دیکشنری
            market_dict = market_data.iloc[0].to_dict()
            
            # بروزرسانی آستانه‌ها
            self.thresholds.update_thresholds(market_dict)
    
    def _analyze_market(self, symbol):
        """تحلیل بازار برای یک نماد خاص"""
        try:
            # دریافت آخرین فیچرها
            features = self.feature_db.get_latest_features(symbol)
            
            if features.empty:
                print(f"No features available for {symbol}")
                return
                
            # دریافت لیست فیچرهای فعال
            active_features = self.feature_selector.get_active_features(symbol)
            
            if not active_features:
                print(f"No active features available for {symbol}")
                return
                
            print(f"Analyzing market for {symbol} with {len(active_features)} active features")
            
            # تحلیل بازار و تولید سیگنال
            signal, confidence, details = self.model_pipeline.analyze(symbol, features, active_features)
            
            # اعمال آستانه‌های تطبیقی
            threshold = self.thresholds.get_threshold(signal)
            
            # تبدیل سیگنال به متن
            signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
            signal_text = signal_map.get(signal, "Unknown")
            
            # بررسی عبور از آستانه
            if confidence >= threshold:
                print(f"Signal for {symbol}: {signal_text} with confidence {confidence:.2f} (threshold: {threshold:.2f})")
            else:
                print(f"Signal below threshold for {symbol}: {signal_text} with confidence {confidence:.2f} < {threshold:.2f}")
                signal = 1  # تغییر به Hold
                signal_text = "Hold"
            
            # ثبت سیگنال برای تحلیل آینده
            self.thresholds.record_signal(signal, confidence, features.iloc[0].to_dict())
            
            return signal, confidence, details
            
        except Exception as e:
            print(f"Error analyzing market for {symbol}: {e}")
            return None, None, None