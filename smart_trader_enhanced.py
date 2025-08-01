"""
راه‌اندازی کننده اصلی سیستم معاملاتی پیشرفته
"""
import argparse
import time
import os
import json
from datetime import datetime
import pandas as pd
from orchestrator.trading_orchestrator import TradingOrchestrator
from feature_store.feature_calculator import FeatureCalculator
from feature_store.feature_monitor import FeatureMonitor

def initialize_database():
    """ایجاد جداول مورد نیاز در دیتابیس"""
    from feature_store.feature_database import FeatureDatabase
    db = FeatureDatabase()
    return db

def start_feature_calculator(symbols=['BTCUSDT', 'ETHUSDT'], interval=1):
    """راه‌اندازی محاسبه‌کننده فیچر"""
    calculator = FeatureCalculator(symbols, update_interval=interval)
    calculator.run()
    return calculator

def start_feature_monitor(symbol='BTCUSDT', interval=5):
    """راه‌اندازی نمایشگر فیچر"""
    monitor = FeatureMonitor(symbol, refresh_rate=interval)
    monitor.run()
    return monitor

def start_trading_system(symbols=['BTCUSDT', 'ETHUSDT']):
    """راه‌اندازی سیستم معاملاتی"""
    orchestrator = TradingOrchestrator(symbols)
    orchestrator.initialize()
    orchestrator.start()
    return orchestrator

def main():
    parser = argparse.ArgumentParser(description="Enhanced Smart Trading Bot")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Symbols to trade')
    parser.add_argument('--feature-calc', action='store_true', help='Run feature calculator only')
    parser.add_argument('--monitor', action='store_true', help='Run feature monitor')
    parser.add_argument('--trading', action='store_true', help='Run trading system')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol for feature monitor')
    parser.add_argument('--interval', type=int, default=1, help='Update interval in seconds')
    parser.add_argument('--train', action='store_true', help='Train specialist models')
    
    args = parser.parse_args()
    
    # ایجاد دایرکتوری‌های مورد نیاز
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('model/specialists', exist_ok=True)
    
    print("====== Enhanced Smart Trading Bot ======")
    print("Initializing database...")
    db = initialize_database()
    
    services = []
    
    if args.feature_calc:
        print(f"Starting feature calculator for {args.symbols} with {args.interval}s interval...")
        calculator = start_feature_calculator(args.symbols, args.interval)
        services.append(calculator)
    
    if args.monitor:
        print(f"Starting feature monitor for {args.symbol} with {args.interval}s refresh rate...")
        monitor = start_feature_monitor(args.symbol, args.interval)
        services.append(monitor)
    
    if args.trading:
        print(f"Starting trading system for {args.symbols}...")
        orchestrator = start_trading_system(args.symbols)
        services.append(orchestrator)
    
    if args.train:
        print("Starting model training...")
        from specialist_models.moving_averages_model import MovingAveragesModel
        from specialist_models.oscillators_model import OscillatorsModel
        from specialist_models.volatility_model import VolatilityModel
        from specialist_models.candlestick_model import CandlestickModel
        from specialist_models.news_model import NewsModel
        from meta_model.model_combiner import ModelCombiner
        from feature_selection.feature_groups import FEATURE_GROUPS
        from data.candle_manager import get_historical_candles
        from data.news_manager import get_historical_news
        from feature_engineering.feature_engineer import build_features
        
        print("Loading historical data...")
        # دریافت داده‌های تاریخی برای آموزش
        candles = get_historical_candles(args.symbol, 10000)
        news = get_historical_news(args.symbol, 1000)
        
        print("Building features...")
        # ساخت فیچرها
        features = build_features(candles, news, args.symbol)
        
        print("Creating target variable...")
        # ساخت متغیر هدف
        features['future_price'] = candles['close'].shift(-6)
        features['future_pct_change'] = features['future_price'] / features['close'] - 1
        
        # طبقه‌بندی به سه گروه: فروش، نگهداری، خرید
        bins = [-float('inf'), -0.005, 0.005, float('inf')]
        labels = [0, 1, 2]
        features['target'] = pd.cut(features['future_pct_change'], bins=bins, labels=labels)
        
        # حذف ردیف‌های بدون مقدار هدف
        features = features.dropna(subset=['target']).copy()
        
        print(f"Training data prepared with {len(features)} samples")
        
        # آموزش مدل‌های متخصص
        specialist_models = {
            'moving_averages': MovingAveragesModel(),
            'oscillators': OscillatorsModel(),
            'volatility': VolatilityModel(),
            'candlestick': CandlestickModel(),
            'news': NewsModel()
        }
        
        for name, model in specialist_models.items():
            print(f"Training {name} model...")
            group_features = FEATURE_GROUPS.get(model.feature_group, [])
            available_features = [f for f in group_features if f in features.columns]
            
            if available_features:
                X = features[available_features]
                y = features['target']
                model.train(X, y)
                model.save()
            
        print("Specialist models trained successfully")
    
    if not (args.feature_calc or args.monitor or args.trading or args.train):
        # اجرای همه سرویس‌ها با تنظیمات پیش‌فرض
        print("Starting all services with default settings...")
        calculator = start_feature_calculator(args.symbols, args.interval)
        services.append(calculator)
        
        orchestrator = start_trading_system(args.symbols)
        services.append(orchestrator)
        
        monitor = start_feature_monitor(args.symbols[0], 5)
        services.append(monitor)
    
    # نمایش راهنمای استفاده
    print("\n====== How to use Enhanced Smart Trading Bot ======")
    print("- For feature calculation only: python smart_trader_enhanced.py --feature-calc")
    print("- For feature monitoring: python smart_trader_enhanced.py --monitor --symbol BTCUSDT --interval 5")
    print("- For trading system: python smart_trader_enhanced.py --trading")
    print("- For training models: python smart_trader_enhanced.py --train --symbol BTCUSDT")
    print("- To run all components: python smart_trader_enhanced.py")
    print("======================================================\n")
    
    try:
        print("Press Ctrl+C to stop the bot")
        # نگه داشتن برنامه در حال اجرا
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping all services...")
        for service in services:
            if hasattr(service, 'stop'):
                service.stop()
        print("Bot stopped")

if __name__ == "__main__":
    main()