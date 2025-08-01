"""
راه‌اندازی کننده اصلی سیستم معاملاتی پیشرفته با پشتیبانی از آموزش چندارزی
"""
import argparse
import time
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import random
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

def prepare_historical_training_data_multi(symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']):
    """
    آماده‌سازی داده‌های تاریخی برای آموزش با استفاده از چندین ارز
    
    Args:
        symbols: لیست ارزهای دیجیتالی که باید آموزش داده شوند
        
    Returns:
        DataFrame: داده‌های آموزشی ترکیب‌شده از چندین ارز
    """
    from data.candle_manager import get_latest_candles
    from data.news_manager import get_latest_news
    from feature_engineering.feature_engineer import build_features
    
    all_training_data = []
    
    for symbol in symbols:
        print(f"\n=== آماده‌سازی داده‌های تاریخی برای {symbol} ===")
        
        # استفاده بهینه از داده‌های موجود بر اساس نوع ارز
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            candles_limit = 15000
            news_limit = 2500
        else:
            candles_limit = 10000
            news_limit = 2000
            
        candles = get_latest_candles(symbol, candles_limit)
        news = get_latest_news(symbol, news_limit)
        
        if candles.empty:
            print(f"Error: No candle data available for {symbol}")
            continue
        
        # بررسی تغییرات قیمت در داده‌های تاریخی
        candles['pct_change'] = candles['close'].pct_change()
        price_changes = candles['pct_change'].dropna()
        
        print(f"Data loaded: {len(candles)} candles and {len(news)} news items")
        print(f"Price change statistics: min={price_changes.min():.4f}, max={price_changes.max():.4f}, mean={price_changes.mean():.4f}, std={price_changes.std():.4f}")
        
        # حذف داده‌های با تغییرات بیش از حد (احتمالاً خطا)
        candles = candles[candles['pct_change'].abs() < 0.2]  # حذف تغییرات بیش از 20%
        
        print("Building features...")
        # ساخت فیچرها
        features = build_features(candles, news, symbol)
        
        # اضافه کردن ستون نماد برای تشخیص منبع داده
        features['symbol'] = symbol
        
        print("Creating target variables with multiple prediction horizons...")
        
        # دسته‌بندی داده‌ها به گروه‌های زمانی برای حفظ تنوع
        time_periods = []
        # تقسیم داده‌ها به 5 دوره زمانی (برای تنوع بیشتر)
        period_size = len(candles) // 5
        for i in range(5):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size
            time_periods.append((start_idx, end_idx))
        
        # افق‌های پیش‌بینی متنوع
        prediction_horizons = [1, 3, 6, 12, 24]
        
        # آستانه‌های متفاوت برای طبقه‌بندی
        threshold_pairs = [
            (-0.002, 0.002),  # بسیار حساس: 0.2% تغییر قیمت
            (-0.005, 0.005),  # استاندارد: 0.5% تغییر قیمت
            (-0.01, 0.01),    # تهاجمی: 1% تغییر قیمت
        ]
        
        symbol_training_data = []
        
        # ایجاد نمونه‌های آموزشی با افق‌ها و آستانه‌های مختلف
        for period_start, period_end in time_periods:
            period_candles = candles.iloc[period_start:period_end].copy()
            period_features = features.iloc[period_start:period_end].copy()
            
            print(f"Processing time period with {len(period_candles)} candles")
            
            for horizon in prediction_horizons:
                for neg_threshold, pos_threshold in threshold_pairs:
                    # محاسبه بازدهی آینده
                    period_features[f'future_price_{horizon}'] = period_candles['close'].shift(-horizon)
                    period_features[f'future_pct_change_{horizon}'] = (
                        period_features[f'future_price_{horizon}'] / period_candles['close'] - 1
                    )
                    
                    # طبقه‌بندی به سه گروه با آستانه‌های متفاوت
                    bins = [-float('inf'), neg_threshold, pos_threshold, float('inf')]
                    labels = [0, 1, 2]  # Sell, Hold, Buy
                    period_features['target'] = pd.cut(period_features[f'future_pct_change_{horizon}'], 
                                                     bins=bins, labels=labels)
                    
                    # حذف ردیف‌های بدون مقدار هدف
                    valid_data = period_features.dropna(subset=['target']).copy()
                    
                    if not valid_data.empty:
                        # تبدیل به عدد صحیح
                        valid_data['target'] = valid_data['target'].astype(int)
                        
                        # اضافه کردن ستون افق و آستانه برای ردیابی
                        valid_data['prediction_horizon'] = horizon
                        valid_data['threshold_negative'] = neg_threshold
                        valid_data['threshold_positive'] = pos_threshold
                        
                        # افزودن به داده‌های آموزشی
                        symbol_training_data.append(valid_data)
                        
                        # گزارش توزیع کلاس‌ها برای این افق و آستانه
                        class_counts = valid_data['target'].value_counts()
                        print(f"Horizon {horizon}, Thresholds {neg_threshold}/{pos_threshold}: {class_counts.to_dict()}")
        
        # ترکیب تمام داده‌های این نماد
        if symbol_training_data:
            final_symbol_data = pd.concat(symbol_training_data)
            
            # حذف ستون‌های اضافی محاسبه شده
            drop_columns = []
            for horizon in prediction_horizons:
                drop_columns.extend([f'future_price_{horizon}', f'future_pct_change_{horizon}'])
            
            final_symbol_data = final_symbol_data.drop(drop_columns, axis=1, errors='ignore')
            
            # بررسی توزیع نهایی
            symbol_class_counts = final_symbol_data['target'].value_counts()
            print(f"\nClass distribution for {symbol} before balancing:")
            for cls, count in symbol_class_counts.items():
                print(f"Class {cls}: {count} samples ({count/len(final_symbol_data)*100:.1f}%)")
            
            # افزودن به داده‌های کلی
            all_training_data.append(final_symbol_data)
    
    # ترکیب داده‌های تمام نمادها
    if all_training_data:
        final_training_data = pd.concat(all_training_data)
        
        # بررسی توزیع نهایی کل داده‌ها
        final_class_counts = final_training_data['target'].value_counts()
        print("\nFinal class distribution before balancing (all symbols):")
        for cls, count in final_class_counts.items():
            print(f"Class {cls}: {count} samples ({count/len(final_training_data)*100:.1f}%)")
        
        # بررسی تعداد کلاس‌های منحصر به فرد
        unique_classes = final_class_counts.index.tolist()
        
        # اگر فقط یک کلاس داریم، باید روشی برای ایجاد مصنوعی نمونه‌های کلاس‌های دیگر پیدا کنیم
        if len(unique_classes) == 1:
            print("\nWARNING: Only one class found in the dataset. Creating synthetic samples for other classes...")
            
            # ایجاد نمونه‌های مصنوعی برای کلاس‌های غایب
            base_samples = final_training_data.copy()
            
            # حذف ستون‌های پیش‌بینی افق و آستانه
            cols_to_drop = ['prediction_horizon', 'threshold_negative', 'threshold_positive', 'symbol']
            drop_cols = [col for col in cols_to_drop if col in base_samples.columns]
            if drop_cols:
                base_samples = base_samples.drop(drop_cols, axis=1)
            
            # یافتن ستون‌های عددی که می‌توانیم تغییر دهیم
            numeric_columns = base_samples.select_dtypes(include=['float64', 'int64']).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'target']
            
            synthetic_samples = []
            
            # برای هر کلاس غایب، نمونه‌های مصنوعی ایجاد می‌کنیم
            for cls in [0, 1, 2]:
                if cls not in unique_classes:
                    print(f"Creating synthetic samples for class {cls}...")
                    
                    # تعداد نمونه‌های مصنوعی که باید ایجاد کنیم
                    n_synthetic = min(300, len(base_samples))
                    
                    # انتخاب تصادفی نمونه‌های پایه
                    base_indices = np.random.choice(base_samples.index, size=n_synthetic, replace=True)
                    synthetic_class = base_samples.loc[base_indices].copy()
                    
                    # تغییر مقادیر عددی برای ایجاد تنوع
                    for col in numeric_columns:
                        if col in synthetic_class.columns:
                            # محاسبه انحراف استاندارد و میانگین
                            mean_val = synthetic_class[col].mean()
                            std_val = synthetic_class[col].std() if synthetic_class[col].std() > 0 else 0.1
                            
                            # اعمال تغییرات تصادفی
                            noise = np.random.normal(0, std_val, size=len(synthetic_class))
                            if cls == 0:  # Sell - تغییرات منفی بیشتر
                                synthetic_class[col] = synthetic_class[col] + noise * 0.5 - std_val * 0.5
                            elif cls == 2:  # Buy - تغییرات مثبت بیشتر
                                synthetic_class[col] = synthetic_class[col] + noise * 0.5 + std_val * 0.5
                            else:  # Hold - تغییرات کمتر
                                synthetic_class[col] = synthetic_class[col] + noise * 0.3
                    
                    # تغییر برچسب کلاس
                    synthetic_class['target'] = cls
                    
                    # افزودن نمونه‌های مصنوعی
                    synthetic_samples.append(synthetic_class)
            
            # افزودن نمونه‌های مصنوعی به داده‌های اصلی
            if synthetic_samples:
                final_training_data = pd.concat([final_training_data] + synthetic_samples)
                
                # بررسی توزیع بعد از افزودن نمونه‌های مصنوعی
                final_class_counts = final_training_data['target'].value_counts()
                print("\nClass distribution after adding synthetic samples:")
                for cls, count in final_class_counts.items():
                    print(f"Class {cls}: {count} samples ({count/len(final_training_data)*100:.1f}%)")
        
        # متعادل‌سازی داده‌ها فقط اگر بیش از یک کلاس داریم
        if len(final_class_counts) > 1:
            # برای هر کلاس
            balanced_dfs = []
            
            # تعداد نمونه‌های هدف برای هر کلاس
            target_samples = 300
            
            for cls in range(3):  # 0, 1, 2 = Sell, Hold, Buy
                if cls in final_class_counts:
                    cls_data = final_training_data[final_training_data['target'] == cls]
                    cls_count = len(cls_data)
                    
                    if cls_count < target_samples:
                        # اگر تعداد نمونه‌ها کمتر از حداقل است، نمونه‌گیری مجدد با جایگذاری
                        if cls_count > 0:
                            resampled = resample(
                                cls_data,
                                replace=True,
                                n_samples=target_samples,
                                random_state=42
                            )
                            balanced_dfs.append(resampled)
                            print(f"Class {cls} upsampled: {cls_count} -> {target_samples}")
                    else:
                        # اگر تعداد نمونه‌ها کافی است، بدون تغییر استفاده کنید
                        balanced_dfs.append(cls_data)
                        print(f"Class {cls} kept as is: {cls_count} samples")
            
            # ترکیب داده‌های متعادل شده
            balanced_training_data = pd.concat(balanced_dfs)
        else:
            # اگر فقط یک کلاس داریم، از داده‌های فعلی استفاده می‌کنیم
            balanced_training_data = final_training_data
        
        # برچسب‌گذاری کلاس‌ها برای خوانایی بهتر
        class_names = {0: "Sell", 1: "Hold", 2: "Buy"}
        
        # بررسی توزیع نهایی
        balanced_counts = balanced_training_data['target'].value_counts()
        print("\nFinal class distribution after balancing:")
        for cls in sorted(balanced_counts.index):
            count = balanced_counts[cls]
            print(f"Class {cls} ({class_names[cls]}): {count} samples ({count/len(balanced_training_data)*100:.1f}%)")
        
        print(f"\nFinal training dataset size: {len(balanced_training_data)} samples with data from {len(symbols)} symbols")
        
        return balanced_training_data
    else:
        print("Error: Could not prepare any valid training data from any symbol")
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
    parser.add_argument('--train-all', action='store_true', help='Train on all supported symbols at once')
    parser.add_argument('--demo-balance', type=float, default=10000, help='Initial balance for demo trading account (USDT)')
    
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
        print(f"Starting trading system for {args.symbols} with demo balance: {args.demo_balance} USDT...")
        orchestrator = start_trading_system(args.symbols)
        orchestrator.set_demo_balance(args.demo_balance)  # تنظیم موجودی حساب دمو
        services.append(orchestrator)
    
    if args.train or args.train_all:
        print("Starting model training...")
        from specialist_models.moving_averages_model import MovingAveragesModel
        from specialist_models.oscillators_model import OscillatorsModel
        from specialist_models.volatility_model import VolatilityModel
        from specialist_models.candlestick_model import CandlestickModel
        from specialist_models.news_model import NewsModel
        from specialist_models.advanced_patterns_model import AdvancedPatternsModel
        from meta_model.model_combiner import ModelCombiner
        from feature_selection.feature_groups import FEATURE_GROUPS
        
        if args.train_all:
            # آموزش روی همه ارزها به‌صورت یکجا
            all_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
            print(f"Training on all symbols: {', '.join(all_symbols)}")
            features = prepare_historical_training_data_multi(all_symbols)
        else:
            # آموزش فقط روی یک ارز
            symbol = args.symbol
            print(f"Training on single symbol: {symbol}")
            
            # از تابع قدیمی استفاده می‌کنیم
            from smart_trader_enhanced import prepare_historical_training_data
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
            'news': NewsModel(),
            'advanced_patterns': AdvancedPatternsModel()
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
    
    if not (args.feature_calc or args.monitor or args.trading or args.train or args.train_all):
        # اجرای همه سرویس‌ها با تنظیمات پیش‌فرض
        print("Starting all services with default settings...")
        calculator = start_feature_calculator(args.symbols, args.interval)
        services.append(calculator)
        
        orchestrator = start_trading_system(args.symbols)
        orchestrator.set_demo_balance(args.demo_balance)  # تنظیم موجودی حساب دمو
        services.append(orchestrator)
        
        monitor = start_feature_monitor(args.symbols[0], 5)
        services.append(monitor)
    
    # نمایش راهنمای استفاده
    print("\n====== How to use Enhanced Smart Trading Bot ======")
    print("- For feature calculation only: python smart_trader_enhanced_multi.py --feature-calc")
    print("- For feature monitoring: python smart_trader_enhanced_multi.py --monitor --symbol BTCUSDT --interval 5")
    print("- For trading system: python smart_trader_enhanced_multi.py --trading --demo-balance 10000")
    print("- For training on a single symbol: python smart_trader_enhanced_multi.py --train --symbol BTCUSDT")
    print("- For training on all symbols at once: python smart_trader_enhanced_multi.py --train-all")
    print("- To run all components: python smart_trader_enhanced_multi.py")
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
