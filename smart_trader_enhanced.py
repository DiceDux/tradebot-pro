"""
ربات معاملات هوشمند ارتقا یافته
با قابلیت‌های پیشرفته تحلیل تکنیکال، سنتیمنت، و یادگیری ماشین
"""
import os
import argparse
import time
import logging
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import mysql.connector
import warnings
from pathlib import Path

# مسیر ریشه پروژه
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

# تنظیمات لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/smart_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smart_trader")

# نادیده گرفتن هشدارها
warnings.filterwarnings('ignore')

# پیکربندی تنظیمات
from utils.config import DB_CONFIG
from utils.config import TRADING_CONFIG

# بارگذاری فایل‌های مدل
from specialist_models.moving_averages_model import MovingAveragesModel
from specialist_models.oscillators_model import OscillatorsModel
from specialist_models.volatility_model import VolatilityModel
from specialist_models.candlestick_model import CandlestickModel
from specialist_models.news_model import NewsModel
from specialist_models.advanced_patterns_model import AdvancedPatternsModel
from meta_model.model_combiner import ModelCombiner
from specialist_models.feature_harmonizer import FeatureHarmonizer
from specialist_models.model_factory import ModelFactory

# واسط‌های دیتا
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from orchestrator.trading_orchestrator import TradingOrchestrator

# تنظیم پردازنده
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # استفاده از CPU به جای GPU
import tensorflow as tf
if tf.test.gpu_device_name():
    print("Device set to use GPU:", tf.test.gpu_device_name())
else:
    print("Device set to use cpu")

def initialize_db():
    """اتصال به دیتابیس و بررسی وضعیت آن"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # بررسی تعداد کندل‌ها
        cursor.execute("SELECT symbol, COUNT(*) as total_candles, MIN(timestamp) as oldest_candle, MAX(timestamp) as newest_candle FROM candles GROUP BY symbol")
        candle_stats = cursor.fetchall()
        
        if candle_stats:
            print("Candle statistics:")
            for stat in candle_stats:
                symbol, count, oldest, newest = stat
                print(f"- {symbol}: {count} candles from {oldest} to {newest}")
        else:
            print("No candle data found in database.")
            
        # بررسی تعداد اخبار
        cursor.execute("SELECT symbol, COUNT(*) as total_news FROM news GROUP BY symbol")
        news_stats = cursor.fetchall()
        
        if news_stats:
            print("News statistics:")
            for stat in news_stats:
                symbol, count = stat
                print(f"- {symbol}: {count} news items")
        else:
            print("No news data found in database.")
            
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"Error connecting to database: {e}")
        return False

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
        
        # استفاده از تمام داده‌های موجود در دیتابیس
        candles = get_latest_candles(symbol, None)
        news = get_latest_news(symbol, None)
        
        if candles.empty:
            print(f"Error: No candle data available for {symbol}")
            continue
        
        print(f"Data loaded: {len(candles)} candles and {len(news)} news items")
        print(f"Date range: {candles['timestamp'].min()} to {candles['timestamp'].max()}")
        
        # بررسی تغییرات قیمت در داده‌های تاریخی
        candles['pct_change'] = candles['close'].pct_change()
        price_changes = candles['pct_change'].dropna()
        
        print(f"Price change statistics: min={price_changes.min():.4f}, max={price_changes.max():.4f}, mean={price_changes.mean():.4f}, std={price_changes.std():.4f}")
        
        # حذف داده‌های با تغییرات بیش از حد (احتمالاً خطا)
        candles = candles[candles['pct_change'].abs() < 0.4]  # حذف تغییرات بیش از 40%
        
        print("Building features...")
        # ساخت فیچرها
        features = build_features(candles, news, symbol)
        
        # اضافه کردن ستون نماد برای تشخیص منبع داده
        features['symbol'] = symbol
        
        print("Creating target variables with multiple prediction horizons...")
        
        # استفاده از افق‌های زمانی کوتاه‌تر برای تمرکز بر خرید و فروش (نه هولد)
        prediction_horizons = [1, 2, 3, 4, 6, 8, 12]
        
        # آستانه‌های متفاوت با تمرکز بر خرید و فروش (کاهش ناحیه هولد)
        threshold_pairs = [
            (-0.001, 0.001),  # آستانه خیلی کم برای کاهش هولد
            (-0.003, 0.003),  # آستانه کم
            (-0.005, 0.005),  # آستانه متوسط
        ]
        
        # افزایش وزن کلاس‌های خرید و فروش
        class_weights = {0: 1.3, 1: 0.7, 2: 1.3}  # وزن بیشتر به خرید و فروش
        
        symbol_training_data = []
        
        # دسته‌بندی داده‌ها به گروه‌های زمانی برای حفظ تنوع
        # تقسیم به دوره‌های زمانی کوچکتر برای تنوع بیشتر
        n_periods = min(4, len(candles) // 3000)  # حداکثر 4 دوره یا براساس تعداد کندل‌ها
        period_size = len(candles) // n_periods
        
        time_periods = []
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(candles)
            time_periods.append((start_idx, end_idx))
        
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
                        
                        # اضافه کردن وزن کلاس برای تمرکز بر خرید و فروش
                        valid_data['class_weight'] = valid_data['target'].map(class_weights)
                        
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
            for cls in sorted(symbol_class_counts.index):
                count = symbol_class_counts[cls]
                print(f"Class {cls}: {count} samples ({count/len(final_symbol_data)*100:.1f}%)")
            
            # افزودن به داده‌های کلی
            all_training_data.append(final_symbol_data)
    
    # ترکیب داده‌های تمام نمادها
    if all_training_data:
        final_training_data = pd.concat(all_training_data)
        
        # بررسی توزیع نهایی کل داده‌ها
        final_class_counts = final_training_data['target'].value_counts()
        print("\nFinal class distribution before balancing (all symbols):")
        for cls in sorted(final_class_counts.index):
            count = final_class_counts[cls]
            print(f"Class {cls}: {count} samples ({count/len(final_training_data)*100:.1f}%)")
        
        # متوازن‌سازی داده‌ها با تمرکز بر خرید و فروش
        from sklearn.utils import resample
        
        # گروه‌های داده بر اساس کلاس
        sell_data = final_training_data[final_training_data['target'] == 0]
        hold_data = final_training_data[final_training_data['target'] == 1]
        buy_data = final_training_data[final_training_data['target'] == 2]
        
        # حفظ داده‌های اصلی (بدون نمونه‌گیری)
        print(f"Hold class kept as is: {len(hold_data)} samples")
        print(f"Sell class kept as is: {len(sell_data)} samples")
        print(f"Buy class kept as is: {len(buy_data)} samples")
        
        # ترکیب داده‌های متوازن شده
        balanced_training_data = pd.concat([sell_data, hold_data, buy_data])
        
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

def train_specialist_models(features_df, target):
    """آموزش مدل‌های متخصص با ویژگی‌های مختلف"""
    print("\nTraining specialist models...")
    specialist_models = []
    
    # آماده‌سازی داده‌ها
    features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # تعریف گروه‌های ویژگی برای مدل‌های متخصص
    feature_groups = {
        'moving_averages': [col for col in features_df.columns if 'ema' in col or 'sma' in col or 'tema' in col] + 
                          ['price_to_ma_3', 'price_to_ma_7', 'price_to_ma_14', 'price_to_ma_30'],
                          
        'oscillators': ['rsi14', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_hist', 
                       'cci', 'willr', 'roc', 'momentum5', 'momentum10'],
                       
        'volatility': ['atr14', 'bb_upper', 'bb_lower', 'bb_width', 'volatility', 'volume', 
                       'volume_mean', 'volume_spike', 'vol_spike', 'atr_spike'] + 
                      [f'std_{p}' for p in [3, 7, 14, 30] if f'std_{p}' in features_df.columns],
                      
        'candlestick': ['doji', 'engulfing', 'hammer', 'morning_star', 'evening_star', 'shooting_star',
                        'candle_range', 'shadow_ratio', 'wick_ratio', 'candle_change', 'green_candle_ratio_20', 
                        'red_candle_ratio_20'],
                        
        'news': [col for col in features_df.columns if 'news_' in col or 'sentiment' in col],
        
        'advanced_patterns': ['fib_382_proximity', 'fib_500_proximity', 'fib_618_proximity',
                             'gartley_pattern', 'butterfly_pattern', 'double_bottom', 'double_top',
                             'direction_change_count', 'signal_to_noise', 'hurst_exponent']
    }
    
    # بارگیری یا آموزش هر مدل متخصص
    for model_name, feature_list in feature_groups.items():
        # اطمینان از وجود تمام ویژگی‌ها در دیتافریم
        available_features = [f for f in feature_list if f in features_df.columns]
        if len(available_features) < 5:  # حداقل 5 ویژگی لازم است
            print(f"Skipping {model_name} model: not enough features available ({len(available_features)})")
            continue
            
        print(f"\nTraining {model_name} model...")
        print(f"Using {len(available_features)} features for {model_name} model")
        print(f"Features: {', '.join(available_features[:10])}...")
        
        # تهیه داده‌های ورودی با ویژگی‌های انتخاب شده
        X = features_df[available_features]
        y = target
        
        try:
            # انتخاب و آموزش مدل متخصص مناسب
            model_instance = None
            if model_name == 'moving_averages':
                model_instance = MovingAveragesModel()
            elif model_name == 'oscillators':
                model_instance = OscillatorsModel()
            elif model_name == 'volatility':
                model_instance = VolatilityModel()
            elif model_name == 'candlestick':
                model_instance = CandlestickModel()
            elif model_name == 'news':
                model_instance = NewsModel()
            elif model_name == 'advanced_patterns':
                model_instance = AdvancedPatternsModel()
                
            if model_instance:
                # استفاده از ModelFactory برای ارزیابی و انتخاب بهترین مدل
                best_model = ModelFactory.evaluate_models(X, y, model_name)
                model_instance.model = best_model
                specialist_models.append(model_instance)
                
                # ذخیره مدل
                if model_instance.save():
                    print(f"{model_name} model trained and saved successfully")
                    
        except Exception as e:
            print(f"Error training {model_name} model: {e}")
            import traceback
            traceback.print_exc()
    
    return specialist_models

def train_meta_model(specialist_models, features_df, target):
    """آموزش مدل متا با استفاده از پیش‌بینی‌های مدل‌های متخصص"""
    print("\nTraining meta model...")
    meta_features = pd.DataFrame(index=features_df.index)
    
    # اضافه کردن پیش‌بینی‌های هر مدل متخصص به فیچرهای متا
    for model in specialist_models:
        try:
            # انتخاب ویژگی‌های مورد نیاز برای این مدل
            required_features = model.get_required_features()
            available_features = [f for f in required_features if f in features_df.columns]
            
            if len(available_features) < 5:
                print(f"Error: Not enough features for {model.model_name} prediction")
                continue
                
            X_model = features_df[available_features]
            
            # هماهنگ‌سازی ویژگی‌ها برای اطمینان از سازگاری
            if hasattr(model.model, 'feature_names_in_'):
                X_model = FeatureHarmonizer.ensure_feature_compatibility(model.model, X_model)
            
            # پیش‌بینی با مدل متخصص
            predictions, probabilities = model.predict(X_model)
            
            # افزودن احتمالات کلاس‌ها به فیچرهای متا
            for i in range(probabilities.shape[1]):
                meta_features[f'{model.model_name}_prob_{i}'] = probabilities[:, i]
                
            print(f"Added predictions from {model.model_name} model to meta features")
            
        except Exception as e:
            print(f"Error in {model.model_name} prediction for meta-model training: {e}")
            import traceback
            traceback.print_exc()
    
    # آموزش مدل متا اگر فیچرهای کافی وجود دارد
    if meta_features.shape[1] > 0:
        combiner = ModelCombiner(specialist_models)
        if combiner.train(meta_features, target):
            combiner.save()
            return combiner
        else:
            print("Error training meta model")
    else:
        print("Error: No meta features available for training")
    
    return None

def train_models():
    """آموزش مدل‌های متخصص و مدل متا"""
    # بررسی آیا پوشه مدل وجود دارد
    os.makedirs('model', exist_ok=True)
    os.makedirs('model/specialists', exist_ok=True)
    
    # بارگیری داده‌های تاریخی و آماده‌سازی ویژگی‌ها
    symbols = TRADING_CONFIG.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'])
    
    # آماده‌سازی داده‌های آموزشی از چندین ارز
    training_data = prepare_historical_training_data_multi(symbols)
    
    if training_data is None or training_data.empty:
        print("Error: No training data available")
        return False
        
    # جداسازی ویژگی‌ها و هدف
    feature_columns = [col for col in training_data.columns if col not in ['target', 'prediction_horizon', 
                                                            'threshold_negative', 'threshold_positive', 'class_weight']]
    features_df = training_data[feature_columns]
    target = training_data['target']
    
    # آموزش مدل‌های متخصص
    specialist_models = train_specialist_models(features_df, target)
    
    if not specialist_models:
        print("Error: Failed to train specialist models")
        return False
        
    # آموزش مدل متا
    meta_model = train_meta_model(specialist_models, features_df, target)
    
    if meta_model:
        print("Meta model trained and saved successfully")
        return True
    else:
        print("Error training meta model")
        return False

def train_models_single_symbol(symbol='BTCUSDT'):
    """آموزش مدل‌ها برای یک نماد خاص"""
    print(f"Training models for {symbol}...")
    
    # بارگیری داده‌های تاریخی
    candles = get_latest_candles(symbol, None)  # بارگیری تمام داده‌های موجود
    news = get_latest_news(symbol, None)
    
    if candles.empty:
        print(f"Error: No candle data available for {symbol}")
        return False
        
    print(f"Data loaded: {len(candles)} candles and {len(news)} news items")
    
    # ساخت ویژگی‌ها
    features = build_features(candles, news, symbol)
    
    # ساخت متغیرهای هدف برای افق‌های زمانی مختلف
    print("Creating target variables...")
    
    # افق‌های زمانی مختلف
    prediction_horizons = [1, 2, 3, 4, 6, 12, 24]
    threshold_pairs = [(-0.001, 0.001), (-0.003, 0.003), (-0.005, 0.005)]
    
    all_targets = []
    
    for horizon in prediction_horizons:
        for neg_threshold, pos_threshold in threshold_pairs:
            # محاسبه بازده آینده
            features[f'future_price_{horizon}'] = candles['close'].shift(-horizon)
            features[f'future_pct_change_{horizon}'] = features[f'future_price_{horizon}'] / candles['close'] - 1
            
            # طبقه‌بندی به سه گروه
            bins = [-float('inf'), neg_threshold, pos_threshold, float('inf')]
            labels = [0, 1, 2]  # Sell, Hold, Buy
            target = pd.cut(features[f'future_pct_change_{horizon}'], bins=bins, labels=labels)
            
            # حذف ردیف‌های بدون مقدار هدف
            valid_rows = ~target.isna()
            
            if valid_rows.sum() > 1000:  # حداقل 1000 نمونه معتبر لازم است
                target_data = pd.DataFrame({
                    'target': target[valid_rows].astype(int),
                    'horizon': horizon,
                    'neg_threshold': neg_threshold,
                    'pos_threshold': pos_threshold
                })
                all_targets.append((target_data, valid_rows))
                
                # گزارش توزیع
                counts = target_data['target'].value_counts()
                print(f"Horizon {horizon}, Thresholds {neg_threshold:.3f}/{pos_threshold:.3f}: "
                      f"Sell={counts.get(0, 0)}, Hold={counts.get(1, 0)}, Buy={counts.get(2, 0)}")
    
    if not all_targets:
        print("Error: Could not create valid target variables")
        return False
        
    # انتخاب بهترین افق و آستانه بر اساس توازن کلاس‌ها
    best_balance = 0
    best_target_data = None
    best_valid_rows = None
    
    for target_data, valid_rows in all_targets:
        counts = target_data['target'].value_counts()
        min_count = counts.min() if len(counts) == 3 else 0
        max_count = counts.max() if len(counts) == 3 else float('inf')
        balance = min_count / max_count if max_count > 0 else 0
        
        if balance > best_balance:
            best_balance = balance
            best_target_data = target_data
            best_valid_rows = valid_rows
    
    if best_target_data is None:
        print("Error: Could not find well-balanced target variable")
        return False
        
    # استفاده از داده‌های انتخاب شده
    valid_features = features.loc[best_valid_rows].copy()
    target = best_target_data['target'].values
    
    # حذف ستون‌های اضافی
    drop_cols = []
    for horizon in prediction_horizons:
        drop_cols.extend([f'future_price_{horizon}', f'future_pct_change_{horizon}'])
    valid_features = valid_features.drop(drop_cols, axis=1, errors='ignore')
    
    # گزارش توزیع نهایی کلاس‌ها
    class_counts = np.bincount(target)
    print("\nFinal class distribution:")
    print(f"Class 0 (Sell): {class_counts[0]} samples ({class_counts[0]/len(target)*100:.1f}%)")
    print(f"Class 1 (Hold): {class_counts[1]} samples ({class_counts[1]/len(target)*100:.1f}%)")
    print(f"Class 2 (Buy): {class_counts[2]} samples ({class_counts[2]/len(target)*100:.1f}%)")
    
    # آموزش مدل‌های متخصص
    specialist_models = train_specialist_models(valid_features, target)
    
    if not specialist_models:
        print("Error: Failed to train specialist models")
        return False
        
    # آموزش مدل متا
    meta_model = train_meta_model(specialist_models, valid_features, target)
    
    if meta_model:
        print("Meta model trained and saved successfully")
        return True
    else:
        print("Error training meta model")
        return False

def start_monitoring(symbol='BTCUSDT', interval=5):
    """شروع نظارت مداوم بر بازار"""
    print(f"Starting monitoring for {symbol} with {interval} minute interval...")
    
    # بارگیری مدل‌ها
    specialist_models = []
    model_files = {
        'moving_averages': (MovingAveragesModel, "model/specialists/moving_averages.pkl"),
        'oscillators': (OscillatorsModel, "model/specialists/oscillators.pkl"),
        'volatility': (VolatilityModel, "model/specialists/volatility.pkl"),
        'candlestick': (CandlestickModel, "model/specialists/candlestick.pkl"),
        'news': (NewsModel, "model/specialists/news.pkl"),
        'advanced_patterns': (AdvancedPatternsModel, "model/specialists/advanced_patterns.pkl")
    }
    
    for name, (model_class, model_path) in model_files.items():
        if os.path.exists(model_path):
            try:
                model = model_class().load()
                specialist_models.append(model)
                print(f"Loaded {name} model")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
    
    if not specialist_models:
        print("Error: No specialist models found. Please train models first.")
        return
    
    # بارگیری مدل متا
    meta_model = ModelCombiner(specialist_models)
    if os.path.exists("model/meta_model.pkl"):
        try:
            meta_model.load()
            print("Loaded meta model")
        except Exception as e:
            print(f"Error loading meta model: {e}")
            return
    else:
        print("Error: Meta model not found. Please train models first.")
        return
    
    print(f"Starting monitoring loop for {symbol}. Press Ctrl+C to stop.")
    
    try:
        while True:
            # دریافت آخرین داده‌ها
            candles = get_latest_candles(symbol, 200)  # آخرین 200 کندل
            news = get_latest_news(symbol, 100)  # آخرین 100 خبر
            
            if candles.empty:
                print("No candle data available")
                time.sleep(60 * interval)
                continue
            
            # ساخت ویژگی‌ها
            features = build_features(candles, news, symbol)
            
            if features.empty:
                print("Error building features")
                time.sleep(60 * interval)
                continue
            
            # پیش‌بینی با هر مدل متخصص
            specialist_predictions = {}
            for model in specialist_models:
                try:
                    # انتخاب ویژگی‌های مورد نیاز
                    required_features = model.get_required_features()
                    available_features = [f for f in required_features if f in features.columns]
                    
                    if len(available_features) < 5:
                        print(f"Warning: Not enough features for {model.model_name}")
                        continue
                    
                    X_model = features[available_features]
                    
                    # هماهنگ‌سازی ویژگی‌ها
                    if hasattr(model.model, 'feature_names_in_'):
                        X_model = FeatureHarmonizer.ensure_feature_compatibility(model.model, X_model)
                    
                    # پیش‌بینی
                    prediction, probabilities = model.predict(X_model)
                    
                    # نمایش نتیجه
                    class_names = {0: "Sell", 1: "Hold", 2: "Buy"}
                    pred_class = prediction[0]
                    confidence = probabilities[0, pred_class] * 100
                    
                    specialist_predictions[model.model_name] = {
                        'prediction': pred_class,
                        'confidence': confidence,
                        'probabilities': probabilities[0]
                    }
                    
                    print(f"{model.model_name} model: {class_names[pred_class]} with {confidence:.1f}% confidence")
                except Exception as e:
                    print(f"Error in {model.model_name} prediction: {e}")
            
            # پیش‌بینی با مدل متا
            try:
                # ساخت ویژگی‌های متا
                meta_features = pd.DataFrame(index=[0])
                
                for model in specialist_models:
                    model_name = model.model_name
                    if model_name in specialist_predictions:
                        probs = specialist_predictions[model_name]['probabilities']
                        for i in range(len(probs)):
                            meta_features[f'{model_name}_prob_{i}'] = probs[i]
                
                # هماهنگ‌سازی ویژگی‌ها
                if hasattr(meta_model.model, 'feature_names_in_'):
                    meta_features = FeatureHarmonizer.ensure_feature_compatibility(meta_model.model, meta_features)
                
                # پیش‌بینی متا
                meta_pred, meta_probs = meta_model.predict(meta_features)
                meta_class = meta_pred[0]
                meta_confidence = meta_probs[0, meta_class] * 100
                
                class_names = {0: "Sell", 1: "Hold", 2: "Buy"}
                print(f"\nMETA MODEL PREDICTION: {class_names[meta_class]} with {meta_confidence:.1f}% confidence")
                
                # نمایش اطلاعات قیمت
                current_price = candles['close'].iloc[-1]
                prev_price = candles['close'].iloc[-2]
                price_change = (current_price - prev_price) / prev_price * 100
                print(f"Current price: {current_price:.2f} ({price_change:+.2f}%)")
                
                # نمایش شاخص‌های مهم
                if 'rsi14' in features.columns:
                    print(f"RSI(14): {features['rsi14'].iloc[0]:.1f}")
                if 'macd' in features.columns:
                    print(f"MACD: {features['macd'].iloc[0]:.6f}")
                if 'bb_width' in features.columns:
                    print(f"BB Width: {features['bb_width'].iloc[0]:.6f}")
                    
                # اطلاعات اخبار
                if 'news_sentiment_mean' in features.columns:
                    print(f"News sentiment: {features['news_sentiment_mean'].iloc[0]:.2f} "
                          f"(last 24h: {features.get('news_sentiment_mean_24h', [0]).iloc[0]:.2f})")
                          
                print("\n" + "-" * 50)
                
            except Exception as e:
                print(f"Error in meta model prediction: {e}")
                import traceback
                traceback.print_exc()
            
            # انتظار تا بررسی بعدی
            print(f"Next update in {interval} minutes...")
            time.sleep(60 * interval)
    except KeyboardInterrupt:
        print("Monitoring stopped by user")

def start_trading(demo_balance=10000):
    """شروع معاملات خودکار"""
    print("Starting trading with orchestrator...")
    
    # بارگیری مدل‌ها
    specialist_models = []
    model_files = {
        'moving_averages': (MovingAveragesModel, "model/specialists/moving_averages.pkl"),
        'oscillators': (OscillatorsModel, "model/specialists/oscillators.pkl"),
        'volatility': (VolatilityModel, "model/specialists/volatility.pkl"),
        'candlestick': (CandlestickModel, "model/specialists/candlestick.pkl"),
        'news': (NewsModel, "model/specialists/news.pkl"),
        'advanced_patterns': (AdvancedPatternsModel, "model/specialists/advanced_patterns.pkl")
    }
    
    for name, (model_class, model_path) in model_files.items():
        if os.path.exists(model_path):
            try:
                model = model_class().load()
                specialist_models.append(model)
                print(f"Loaded {name} model")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
    
    if not specialist_models:
        print("Error: No specialist models found. Please train models first.")
        return
    
    # بارگیری مدل متا
    meta_model = ModelCombiner(specialist_models)
    if os.path.exists("model/meta_model.pkl"):
        try:
            meta_model.load()
            print("Loaded meta model")
        except Exception as e:
            print(f"Error loading meta model: {e}")
            return
    else:
        print("Error: Meta model not found. Please train models first.")
        return
    
    # راه‌اندازی ارکستراتور معاملاتی
    symbols = TRADING_CONFIG.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'])
    check_interval = TRADING_CONFIG.get('check_interval', 5)  # بررسی هر 5 دقیقه
    
    orchestrator = TradingOrchestrator(
        symbols=symbols,
        specialist_models=specialist_models,
        meta_model=meta_model,
        demo_balance=demo_balance,
        check_interval=check_interval
    )
    
    try:
        orchestrator.start_trading_loop()
    except KeyboardInterrupt:
        print("Trading stopped by user")
        orchestrator.stop_trading_loop()
        orchestrator.save_trading_history()

def calculate_features(symbol='BTCUSDT'):
    """محاسبه و نمایش فیچرهای پیشرفته بدون پیش‌بینی"""
    print(f"Calculating features for {symbol}...")
    
    # دریافت داده‌های کندل و خبر
    candles = get_latest_candles(symbol, 200)
    news = get_latest_news(symbol, 100)
    
    if candles.empty:
        print("No candle data available")
        return
        
    # ساخت و نمایش فیچرها
    features = build_features(candles, news, symbol)
    
    if features.empty:
        print("Error building features")
        return
    
    # نمایش اطلاعات فیچرها
    print("\nFeature values:")
    
    # گروه‌بندی فیچرها برای نمایش بهتر
    feature_groups = {
        'Price': ['close', 'open', 'high', 'low'],
        'Moving Averages': [col for col in features.columns if 'ema' in col or 'sma' in col or 'tema' in col],
        'Oscillators': ['rsi14', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_hist', 
                      'cci', 'willr', 'roc', 'momentum5', 'momentum10'],
        'Volatility': ['atr14', 'bb_upper', 'bb_lower', 'bb_width', 'volatility', 'volume', 
                     'volume_mean', 'volume_spike'],
        'Candlestick': ['doji', 'engulfing', 'hammer', 'morning_star', 'evening_star', 'shooting_star',
                       'candle_range', 'shadow_ratio', 'wick_ratio'],
        'News & Sentiment': [col for col in features.columns if 'news_' in col or 'sentiment' in col],
        'Advanced Patterns': ['fib_382_proximity', 'fib_500_proximity', 'fib_618_proximity', 
                            'gartley_pattern', 'butterfly_pattern', 'double_bottom', 'double_top']
    }
    
    # نمایش فیچرها گروه به گروه
    for group, cols in feature_groups.items():
        print(f"\n--- {group} ---")
        valid_cols = [col for col in cols if col in features.columns]
        if valid_cols:
            for col in valid_cols:
                print(f"{col}: {features[col].iloc[0]:.6f}")
        else:
            print("No features in this group")

def print_help():
    """نمایش راهنمای استفاده از برنامه"""
    print("\n====== How to use Enhanced Smart Trading Bot ======")
    print("- For feature calculation only: python smart_trader_enhanced.py --feature-calc")
    print("- For feature monitoring: python smart_trader_enhanced.py --monitor --symbol BTCUSDT --interval 5")
    print("- For trading system: python smart_trader_enhanced.py --trading --demo-balance 10000")
    print("- For training on a single symbol: python smart_trader_enhanced.py --train --symbol BTCUSDT")
    print("- For training on all symbols at once: python smart_trader_enhanced.py --train-all")
    print("- To run all components: python smart_trader_enhanced.py")
    print("======================================================\n")

def main():
    """تابع اصلی برنامه"""
    print("====== Enhanced Smart Trading Bot ======")
    
    parser = argparse.ArgumentParser(description='Smart Trading Bot with Enhanced ML capabilities')
    parser.add_argument('--train', action='store_true', help='Train models for a specific symbol')
    parser.add_argument('--train-all', action='store_true', help='Train models with data from all symbols')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to use (default: BTCUSDT)')
    parser.add_argument('--monitor', action='store_true', help='Monitor market in real-time')
    parser.add_argument('--trading', action='store_true', help='Start automated trading')
    parser.add_argument('--demo-balance', type=float, default=10000, help='Demo balance for trading')
    parser.add_argument('--interval', type=int, default=5, help='Interval in minutes for monitoring/trading')
    parser.add_argument('--feature-calc', action='store_true', help='Only calculate and display features')
    args = parser.parse_args()
    
    # بررسی اتصال به دیتابیس
    print("Initializing database...")
    if not initialize_db():
        print("Error initializing database. Please check connection settings.")
        return
    
    # انتخاب عملیات بر اساس آرگومان‌ها
    if args.train:
        print(f"Starting model training for {args.symbol}...")
        train_models_single_symbol(args.symbol)
    elif args.train_all:
        print("Starting model training...")
        symbols = TRADING_CONFIG.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'])
        print(f"Training on all symbols: {', '.join(symbols)}")
        train_models()
    elif args.monitor:
        start_monitoring(args.symbol, args.interval)
    elif args.trading:
        start_trading(args.demo_balance)
    elif args.feature_calc:
        calculate_features(args.symbol)
    else:
        print_help()
        print("Press Ctrl+C to stop the bot")
        
        try:
            # اجرای همه بخش‌ها به صورت پیش‌فرض
            start_trading(args.demo_balance)
        except KeyboardInterrupt:
            print("\nBot stopped by user")

if __name__ == "__main__":
    main()
