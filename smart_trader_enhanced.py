"""
راه‌اندازی کننده اصلی سیستم معاملاتی پیشرفته (نسخه بهینه شده)
"""
import argparse
import time
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.utils import resample
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

def prepare_historical_training_data(symbol):
    """
    آماده‌سازی داده‌های تاریخی برای آموزش با استفاده از تکنیک‌های پیشرفته
    """
    from data.candle_manager import get_latest_candles
    from data.news_manager import get_latest_news
    from feature_engineering.feature_engineer import build_features
    
    print(f"Loading historical data for {symbol}...")
    
    # استفاده بهینه از داده‌های موجود بر اساس نوع ارز
    if symbol in ["BTCUSDT", "ETHUSDT"]:
        candles_limit = 15000  # برای بیت‌کوین و اتریوم که ~17000 کندل داریم
        news_limit = 5000  # برای اخبار بیت‌کوین (~22500) و اتریوم (~14800)
    else:
        candles_limit = 10000  # برای سایر ارزها
        news_limit = 3000  # برای سایر ارزها
        
    candles = get_latest_candles(symbol, candles_limit)
    news = get_latest_news(symbol, news_limit)
    
    if candles.empty:
        print(f"Error: No candle data available for {symbol}")
        return None
    
    print(f"Data loaded: {len(candles)} candles and {len(news)} news items")
    print("Building features...")
    
    # ساخت فیچرها
    features = build_features(candles, news, symbol)
    
    print("Creating target variables with multiple prediction horizons...")
    
    # افق‌های پیش‌بینی متنوع - تطبیق داده شده با بازه‌های معاملاتی 4 ساعته
    # برای بازه 4 ساعته، معادل روزانه = 6 کندل، هفتگی = 42 کندل، ماهانه ~180 کندل
    prediction_horizons = [1, 3, 6, 12, 24, 42]  # از خیلی کوتاه‌مدت تا میان‌مدت
    
    # آستانه‌های متفاوت برای طبقه‌بندی
    threshold_pairs = [
        (-0.005, 0.005),  # استاندارد: 0.5% تغییر قیمت
        (-0.008, 0.008),  # تهاجمی‌تر: 0.8% تغییر قیمت
        (-0.003, 0.003),  # محافظه‌کارانه‌تر: 0.3% تغییر قیمت
    ]
    
    all_training_data = []
    
    # ایجاد نمونه‌های آموزشی با افق‌ها و آستانه‌های مختلف
    for horizon in prediction_horizons:
        for neg_threshold, pos_threshold in threshold_pairs:
            # کپی از فیچرها
            features_copy = features.copy()
            
            # محاسبه بازدهی آینده
            features_copy[f'future_price_{horizon}'] = candles['close'].shift(-horizon)
            features_copy[f'future_pct_change_{horizon}'] = features_copy[f'future_price_{horizon}'] / features_copy['close'] - 1
            
            # طبقه‌بندی به سه گروه با آستانه‌های متفاوت
            bins = [-float('inf'), neg_threshold, pos_threshold, float('inf')]
            labels = [0, 1, 2]  # Sell, Hold, Buy
            features_copy['target'] = pd.cut(features_copy[f'future_pct_change_{horizon}'], bins=bins, labels=labels)
            
            # حذف ردیف‌های بدون مقدار هدف
            features_copy = features_copy.dropna(subset=['target']).copy()
            
            if not features_copy.empty:
                # تبدیل به عدد صحیح
                features_copy['target'] = features_copy['target'].astype(int)
                
                # اضافه کردن ستون افق و آستانه برای ردیابی
                features_copy['prediction_horizon'] = horizon
                features_copy['threshold_negative'] = neg_threshold
                features_copy['threshold_positive'] = pos_threshold
                
                # افزودن به داده‌های آموزشی
                all_training_data.append(features_copy)
                
                # گزارش توزیع کلاس‌ها برای این افق و آستانه
                class_counts = features_copy['target'].value_counts()
                print(f"Horizon {horizon}, Thresholds {neg_threshold}/{pos_threshold}: {class_counts.to_dict()}")
    
    # ترکیب تمام داده‌ها
    if all_training_data:
        final_training_data = pd.concat(all_training_data)
        
        # حذف ستون‌های اضافی محاسبه شده
        drop_columns = []
        for horizon in prediction_horizons:
            drop_columns.extend([f'future_price_{horizon}', f'future_pct_change_{horizon}'])
        
        final_training_data = final_training_data.drop(drop_columns, axis=1, errors='ignore')
        
        # بررسی توزیع نهایی
        final_class_counts = final_training_data['target'].value_counts()
        print("\nFinal class distribution before balancing:")
        for cls, count in final_class_counts.items():
            print(f"Class {cls}: {count} samples ({count/len(final_training_data)*100:.1f}%)")
        
        # متعادل‌سازی داده‌ها (نمونه‌گیری مجدد)
        # هدف: داشتن حداقل 300 نمونه از هر کلاس و حداکثر 5000 نمونه از هر کلاس
        balanced_dfs = []
        
        # برای هر کلاس
        target_min_samples = 300
        target_max_samples = 5000
        
        for cls in range(3):  # 0, 1, 2 = Sell, Hold, Buy
            cls_data = final_training_data[final_training_data['target'] == cls]
            cls_count = len(cls_data)
            
            if cls_count < target_min_samples:
                # اگر تعداد نمونه‌ها کمتر از حداقل است، نمونه‌گیری مجدد با جایگذاری
                resampled = resample(
                    cls_data, 
                    replace=True,
                    n_samples=target_min_samples,
                    random_state=42
                )
                balanced_dfs.append(resampled)
                print(f"Class {cls} upsampled: {cls_count} -> {target_min_samples}")
            elif cls_count > target_max_samples:
                # اگر تعداد نمونه‌ها بیشتر از حداکثر است، کاهش تعداد نمونه‌ها
                resampled = resample(
                    cls_data, 
                    replace=False,
                    n_samples=target_max_samples,
                    random_state=42
                )
                balanced_dfs.append(resampled)
                print(f"Class {cls} downsampled: {cls_count} -> {target_max_samples}")
            else:
                # اگر تعداد نمونه‌ها در محدوده مناسب است، بدون تغییر استفاده کنید
                balanced_dfs.append(cls_data)
                print(f"Class {cls} kept as is: {cls_count} samples")
        
        # ترکیب داده‌های متعادل شده
        balanced_training_data = pd.concat(balanced_dfs)
        
        # برچسب‌گذاری کلاس‌ها برای خوانایی بهتر
        class_names = {0: "Sell", 1: "Hold", 2: "Buy"}
        
        # بررسی توزیع بعد از متعادل‌سازی
        balanced_counts = balanced_training_data['target'].value_counts()
        print("\nFinal class distribution after balancing:")
        for cls, count in balanced_counts.items():
            print(f"Class {cls} ({class_names[cls]}): {count} samples ({count/len(balanced_training_data)*100:.1f}%)")
        
        print(f"\nFinal training dataset size: {len(balanced_training_data)} samples")
        
        return balanced_training_data
    else:
        print("Error: Could not prepare any valid training data")
        return None

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
        
        symbol = args.symbol
        
        # استفاده از تابع پیشرفته آماده‌سازی داده‌های تاریخی
        features = prepare_historical_training_data(symbol)
        
        if features is None or features.empty:
            print("Error: Failed to prepare training data. Exiting.")
            sys.exit(1)
        
        # آموزش مدل‌های متخصص
        specialist_models = {
            'moving_averages': MovingAveragesModel(),
            'oscillators': OscillatorsModel(),
            'volatility': VolatilityModel(),
            'candlestick': CandlestickModel(),
            'news': NewsModel()
        }
        
        for name, model in specialist_models.items():
            print(f"\nTraining {name} model...")
            group_features = FEATURE_GROUPS.get(model.feature_group, [])
            available_features = [f for f in group_features if f in features.columns]
            
            if available_features:
                print(f"Using {len(available_features)} features for {name} model")
                print(f"Features: {', '.join(available_features[:10])}{'...' if len(available_features) > 10 else ''}")
                
                X = features[available_features]
                y = features['target']
                
                try:
                    model.train(X, y)
                    model.save()
                    print(f"{name} model trained and saved successfully")
                except Exception as e:
                    print(f"Error training {name} model: {e}")
            else:
                print(f"Warning: No features available for {name} model")
        
        print("\nTraining meta model...")
        # ساخت مدل متا با استفاده از مدل‌های متخصص
        combiner = ModelCombiner(list(specialist_models.values()))
        
        # ایجاد داده‌های ورودی برای مدل متا
        meta_features = pd.DataFrame(index=features.index)
        
        # پیش‌بینی با هر مدل متخصص
        for name, model in specialist_models.items():
            if model.model is None:
                print(f"Warning: {name} model not trained, skipping")
                continue
                
            # انتخاب فیچرهای مربوط به گروه این مدل
            required_features = model.get_required_features()
            available_features = [f for f in required_features if f in features.columns]
            
            if available_features:
                try:
                    # پیش‌بینی احتمالات کلاس‌ها
                    X = features[available_features]
                    _, probas = model.predict(X)
                    
                    # افزودن احتمالات به فیچرهای مدل متا
                    for j in range(probas.shape[1]):
                        meta_features[f"{name}_class{j}"] = probas[:, j]
                    
                    print(f"Added predictions from {name} model to meta features")
                    
                except Exception as e:
                    print(f"Error in {name} prediction for meta-model training: {e}")
        
        if not meta_features.empty:
            # آموزش مدل متا
            combiner.train(meta_features, features['target'])
            combiner.save()
            print("Meta model trained and saved successfully")
        else:
            print("Warning: No meta-features available for meta-model training")
    
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
