"""
محاسبه فیچرهای پیشرفته برای تحلیل بازار
با استفاده از ترکیب فیچرهای تکنیکال، آماری، خبری و فاندامنتال
"""
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import logging
import os
from .feature_config import FEATURE_CONFIG

# تنظیم لاگر
logger = logging.getLogger("feature_engineer")
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/features.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# بارگیری کتابخانه‌ها با مدیریت خطا
try:
    import ta
except ImportError:
    ta = None
    logger.warning("ta library not available, some features will be disabled")

try:
    import talib
except ImportError:
    talib = None
    logger.warning("talib library not available, some pattern features will be disabled")

try:
    # برای استفاده از FinBERT برای تحلیل احساسات خبرها
    from feature_engineering.sentiment_finbert import analyze_sentiment_finbert
    USE_FINBERT = True
    logger.info("FinBERT model loaded for sentiment analysis")
except ImportError:
    USE_FINBERT = False
    logger.warning("FinBERT not available, falling back to basic sentiment analysis")

def safe_ema(close, span):
    if len(close) >= span:
        return close[-span:].ewm(span=span).mean().values[-1]
    else:
        return 0.0

def safe_sma(close, window):
    if len(close) >= window:
        return close[-window:].rolling(window=window).mean().values[-1]
    else:
        return 0.0

def safe_tema(close, window):
    if len(close) >= window:
        ema1 = close[-window:].ewm(span=window).mean()
        ema2 = ema1.ewm(span=window).mean()
        ema3 = ema2.ewm(span=window).mean()
        return 3 * (ema1.values[-1] - ema2.values[-1]) + ema3.values[-1]
    else:
        return 0.0

def safe_rsi(close, window):
    if len(close) >= window + 1:
        delta = close[-(window+1):].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / (loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs.values[-1])) if not np.isnan(rs.values[-1]) else 50
    else:
        return 50

def safe_atr(high, low, close, window):
    if len(close) >= window + 1:
        tr = pd.concat([
            (high[-(window+1):] - low[-(window+1):]),
            (high[-(window+1):] - close[-(window+1):].shift()).abs(),
            (low[-(window+1):] - close[-(window+1):].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean().values[-1]
    else:
        return 0.0

def safe_macd(close, fast=12, slow=26, signal=9):
    if len(close) >= slow + 1:
        ema_fast = close[-(slow+1):].ewm(span=fast).mean()
        ema_slow = close[-(slow+1):].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd.values[-1], macd_signal.values[-1], macd_hist.values[-1]
    else:
        return 0.0, 0.0, 0.0

def safe_bb(close, window=20):
    if len(close) >= window:
        bb_mid = close[-window:].rolling(window=window).mean()
        bb_std = close[-window:].rolling(window=window).std()
        bb_upper = (bb_mid + 2 * bb_std).values[-1]
        bb_lower = (bb_mid - 2 * bb_std).values[-1]
        bb_width = bb_upper - bb_lower
        return bb_upper, bb_lower, bb_width
    else:
        return 0.0, 0.0, 0.0

def calculate_obv(df):
    if len(df) < 2:
        return 0.0
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv[-1]

def calculate_vwap(df):
    if len(df) < 1:
        return 0.0
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap.values[-1]

def safe_pattern_value(pattern_func):
    """دریافت امن مقدار آخرین الگوی شمعی"""
    try:
        result = pattern_func
        if len(result) > 0:
            return result[-1]
        else:
            return 0
    except Exception:
        return 0

def build_features(candles_df, news_df, symbol):
    """
    ساخت فیچرها با بهبود پشتیبانی از MySQL و اطمینان از محاسبه صحیح فیچرها
    """
    features = {}
    debug_info = {}  # برای ذخیره اطلاعات دیباگ
    
    # === بررسی و اصلاح داده های ورودی ===
    if candles_df is None or candles_df.empty:
        logger.warning(f"Empty candle data provided for {symbol}")
        return pd.DataFrame([{}])
    
    # اطمینان از وجود ستون‌های اصلی
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    for col in required_columns:
        if col not in candles_df.columns:
            if col == 'timestamp' and 'time' in candles_df.columns:
                # برخی دیتابیس‌ها از 'time' به جای 'timestamp' استفاده می‌کنند
                candles_df['timestamp'] = candles_df['time']
                logger.info(f"Using 'time' column as 'timestamp' for {symbol}")
            else:
                logger.warning(f"Missing required column '{col}' in candle data for {symbol}")
                candles_df[col] = 0.0
    
    # === رفع مشکل اطلاعات اخبار ===
    if news_df is not None and not news_df.empty:
        # اطلاعات اخبار
        debug_info['news_count'] = len(news_df)
        debug_info['news_columns'] = list(news_df.columns)
        
        # ساخت یک کپی از دیتافریم خبری
        news_df = news_df.copy()
        
        # بررسی و تبدیل ستون‌های زمان برای اخبار
        if 'ts' not in news_df.columns:
            if 'published_at' in news_df.columns:
                logger.info(f"Converting 'published_at' to 'ts' for {symbol} news ({len(news_df)} items)")
                print(f"Converting 'published_at' to 'ts' for {symbol} news ({len(news_df)} items)")
                
                # بررسی نوع داده published_at
                sample = news_df['published_at'].iloc[0] if len(news_df) > 0 else None
                logger.info(f"Sample published_at value: {sample} (type: {type(sample).__name__})")
                print(f"Sample published_at value: {sample} (type: {type(sample).__name__})")
                
                try:
                    # اگر نوع داده timestamp یا datetime64 است
                    if pd.api.types.is_datetime64_dtype(news_df['published_at']):
                        # تبدیل صحیح به int64 و سپس به timestamp
                        news_df['ts'] = news_df['published_at'].astype('int64') // 10**9
                        logger.info("Converted datetime64 values to Unix timestamps")
                        print("Converted datetime64 values to Unix timestamps")
                    
                    # اگر نوع داده رشته است
                    elif pd.api.types.is_string_dtype(news_df['published_at']):
                        # تبدیل رشته به datetime64 و سپس به timestamp
                        news_df['ts'] = pd.to_datetime(news_df['published_at']).astype('int64') // 10**9
                        logger.info("Converted string date values to Unix timestamps")
                        print("Converted string date values to Unix timestamps")
                    
                    # اگر عدد صحیح است
                    elif pd.api.types.is_integer_dtype(news_df['published_at']):
                        # اگر قبلاً timestamp است، مستقیماً استفاده می‌کنیم
                        news_df['ts'] = news_df['published_at']
                        logger.info("Using existing integer timestamps")
                        print("Using existing integer timestamps")
                    
                    else:
                        # تلاش برای تبدیل با to_datetime
                        news_df['ts'] = pd.to_datetime(news_df['published_at']).astype('int64') // 10**9
                        logger.info("Successfully converted to timestamps with generic method")
                        print("Successfully converted to timestamps with generic method")
                    
                    # گزارش نتایج تبدیل
                    if len(news_df) > 0:
                        now_ts = int(time.time())
                        oldest_ts = news_df['ts'].min()
                        newest_ts = news_df['ts'].max()
                        
                        # محاسبه فاصله زمانی بر حسب ساعت
                        oldest_hours = (now_ts - oldest_ts) / 3600
                        newest_hours = (now_ts - newest_ts) / 3600
                        
                        logger.info(f"News timespan: {oldest_hours:.1f} hours ago to {newest_hours:.1f} hours ago")
                        print(f"News timespan: {oldest_hours:.1f} hours ago to {newest_hours:.1f} hours ago")
                        
                        # آمار توزیع اخبار در بازه‌های زمانی
                        h1 = len(news_df[news_df['ts'] >= now_ts - 3600])
                        h6 = len(news_df[news_df['ts'] >= now_ts - 21600])
                        h24 = len(news_df[news_df['ts'] >= now_ts - 86400])
                        h72 = len(news_df[news_df['ts'] >= now_ts - 259200])
                        
                        logger.info(f"News distribution: 1h: {h1}, 6h: {h6}, 24h: {h24}, 72h: {h72}, Total: {len(news_df)}")
                        print(f"News distribution: 1h: {h1}, 6h: {h6}, 24h: {h24}, 72h: {h72}, Total: {len(news_df)}")
                    
                except Exception as e:
                    logger.error(f"Error in timestamp conversion: {e}")
                    print(f"Error in timestamp conversion: {e}")
                    # روش جایگزین با استفاده از strptime
                    logger.info("Using direct string parsing with strptime")
                    print("Using direct string parsing with strptime")
                    ts_values = []
                    
                    for val in news_df['published_at']:
                        try:
                            # تبدیل با فرمت MySQL timestamp
                            if isinstance(val, str):
                                dt = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                                ts_values.append(int(dt.timestamp()))
                            elif isinstance(val, pd.Timestamp):
                                # اگر Timestamp پانداس است
                                ts_values.append(int(val.timestamp()))
                            elif isinstance(val, (int, float)):
                                # اگر عدد است
                                ts_values.append(int(val))
                            else:
                                # در صورت نوع نامشخص
                                ts_values.append(int(time.time()))
                        except Exception:
                            # در صورت خطا، از زمان فعلی استفاده می‌کنیم
                            ts_values.append(int(time.time()))
                    
                    news_df['ts'] = ts_values
            else:
                logger.warning(f"No 'published_at' column found in news data for {symbol}")
                print(f"Warning: No 'published_at' column found in news data for {symbol}")
                # ایجاد ستون ts با مقادیر پیش‌فرض
                news_df['ts'] = int(time.time())
        
        # اطمینان از وجود sentiment_score
        if 'sentiment_score' not in news_df.columns:
            logger.warning(f"No sentiment_score column in news data for {symbol}")
            print(f"Warning: No sentiment_score column in news data for {symbol}")
            
            # استفاده از FinBERT برای محاسبه امتیاز احساسات
            if USE_FINBERT and 'content' in news_df.columns:
                logger.info(f"Using FinBERT to calculate sentiment scores for {len(news_df)} news items")
                print(f"Using FinBERT to calculate sentiment scores for {len(news_df)} news items")
                sentiment_scores = []
                
                for content in news_df['content']:
                    try:
                        score = analyze_sentiment_finbert(content)
                        sentiment_scores.append(score)
                    except Exception as e:
                        logger.error(f"Error calculating FinBERT sentiment: {e}")
                        sentiment_scores.append(0.0)
                
                news_df['sentiment_score'] = sentiment_scores
            else:
                news_df['sentiment_score'] = 0.0
        
        # تبدیل sentiment_score به عدد
        news_df['sentiment_score'] = pd.to_numeric(news_df['sentiment_score'], errors='coerce').fillna(0.0)

    # =========== فیچرهای تکنیکال و آماری ===========
    if candles_df is not None and not candles_df.empty:
        # تبدیل داده‌های ستون‌ها به سری‌های pandas
        try:
            close = pd.Series(candles_df['close'].values)
            high = pd.Series(candles_df['high'].values)
            low = pd.Series(candles_df['low'].values)
            open_ = pd.Series(candles_df['open'].values)
            volume = pd.Series(candles_df['volume'].values)
            
            # اطلاعات دیباگ
            debug_info['candle_count'] = len(close)
            debug_info['price_latest'] = close.values[-1] if len(close) > 0 else 0
        except Exception as e:
            logger.error(f"Error preparing candle data: {e}")
            print(f"Error preparing candle data: {e}")
            # ایجاد سری‌های خالی
            close = pd.Series([])
            high = pd.Series([])
            low = pd.Series([])
            open_ = pd.Series([])
            volume = pd.Series([])

        # EMA ها
        for ema_span in [5, 9, 10, 20, 21, 50, 100, 200]:
            k = f'ema{ema_span}'
            if FEATURE_CONFIG.get(k, True):  # پیش‌فرض فعال
                features[k] = safe_ema(close, ema_span)

        # SMA ها
        for sma_win in [20, 50]:
            k = f'sma{sma_win}'
            if FEATURE_CONFIG.get(k, True):
                features[k] = safe_sma(close, sma_win)

        # TEMA
        if FEATURE_CONFIG.get('tema20', True):
            features['tema20'] = safe_tema(close, 20)

        # RSI
        if FEATURE_CONFIG.get('rsi14', True):
            features['rsi14'] = safe_rsi(close, 14)

        # ATR
        if FEATURE_CONFIG.get('atr14', True):
            features['atr14'] = safe_atr(high, low, close, 14)

        # MACD
        macd, macd_signal, macd_hist = safe_macd(close, 12, 26, 9)
        if FEATURE_CONFIG.get('macd', True): features['macd'] = macd
        if FEATURE_CONFIG.get('macd_signal', True): features['macd_signal'] = macd_signal
        if FEATURE_CONFIG.get('macd_hist', True): features['macd_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_lower, bb_width = safe_bb(close, 20)
        if FEATURE_CONFIG.get('bb_upper', True): features['bb_upper'] = bb_upper
        if FEATURE_CONFIG.get('bb_lower', True): features['bb_lower'] = bb_lower
        if FEATURE_CONFIG.get('bb_width', True): features['bb_width'] = bb_width

        # OBV
        if FEATURE_CONFIG.get("obv", True):
            features['obv'] = calculate_obv(candles_df)

        # VWAP
        if FEATURE_CONFIG.get("vwap", True):
            features['vwap'] = calculate_vwap(candles_df)

        # Stochastic
        if FEATURE_CONFIG.get('stoch_k', True) or FEATURE_CONFIG.get('stoch_d', True):
            if len(close) >= 14:
                low14 = low[-14:]
                high14 = high[-14:]
                stoch_k = 100 * (close.values[-1] - low14.min()) / (high14.max() - low14.min() + 1e-8)
                if FEATURE_CONFIG.get('stoch_k', True):
                    features['stoch_k'] = stoch_k
                if FEATURE_CONFIG.get('stoch_d', True):
                    features['stoch_d'] = pd.Series([stoch_k]).rolling(3).mean().values[-1]
            else:
                if FEATURE_CONFIG.get('stoch_k', True):
                    features['stoch_k'] = 0.0
                if FEATURE_CONFIG.get('stoch_d', True):
                    features['stoch_d'] = 0.0

        # CCI
        if FEATURE_CONFIG.get('cci', True):
            if len(close) >= 20:
                tp = (high[-20:] + low[-20:] + close[-20:]) / 3
                cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
                features['cci'] = cci.values[-1]
            else:
                features['cci'] = 0.0

        # willr
        if FEATURE_CONFIG.get('willr', True):
            if len(close) >= 14:
                low14 = low[-14:]
                high14 = high[-14:]
                willr = (high14.max() - close.values[-1]) / (high14.max() - low14.min() + 1e-8) * -100
                features['willr'] = willr
            else:
                features['willr'] = 0.0

        # ROC
        if FEATURE_CONFIG.get('roc', True):
            if len(close) >= 11:
                features['roc'] = close.pct_change(periods=10).values[-1]
            else:
                features['roc'] = 0.0

        # PSAR (نمونه: فقط آخرین قیمت)
        if FEATURE_CONFIG.get('psar', True):
            features['psar'] = close.values[-1]

        # candle_change
        if FEATURE_CONFIG.get('candle_change', True):
            if len(close) >= 2:
                features['candle_change'] = close.pct_change().values[-1]
            else:
                features['candle_change'] = 0.0

        # candle_range
        if FEATURE_CONFIG.get('candle_range', True):
            features['candle_range'] = (high.values[-1] - low.values[-1])

        # volume_mean
        if FEATURE_CONFIG.get('volume_mean', True):
            if len(volume) >= 20:
                features['volume_mean'] = volume[-20:].mean()
            else:
                features['volume_mean'] = volume.mean()

        # volume_spike
        if FEATURE_CONFIG.get('volume_spike', True):
            if len(volume) >= 20:
                features['volume_spike'] = float(volume.values[-1] > np.mean(volume[-20:]) * 1.5)
            else:
                features['volume_spike'] = 0.0

        # قیمت‌های کندل آخر
        for k in ['close','open','high','low','volume']:
            if FEATURE_CONFIG.get(k, True):
                features[k] = candles_df[k].values[-1]

        # ====== اندیکاتورهای مدرن و پرایس اکشن (با ta) ======
        if ta is not None:
            adx_window = 14
            adx_min = adx_window + 1
            if FEATURE_CONFIG.get('adx14', True):
                if len(close) >= adx_min and len(high) >= adx_min and len(low) >= adx_min:
                    try:
                        adx_vals = ta.trend.ADXIndicator(
                            high[-adx_min:], low[-adx_min:], close[-adx_min:], window=adx_window
                        ).adx().values
                        features['adx14'] = adx_vals[-1] if len(adx_vals) > 0 else 0.0
                    except Exception:
                        features['adx14'] = 0.0
                else:
                    features['adx14'] = 0.0

            if FEATURE_CONFIG.get('supertrend', True):
                try:
                    features['supertrend'] = ta.trend.stc(close).values[-1]
                except Exception:
                    features['supertrend'] = 0.0

            if FEATURE_CONFIG.get('donchian_high', True):
                if len(high) >= 20:
                    features['donchian_high'] = high[-20:].max()
                else:
                    features['donchian_high'] = 0.0

            if FEATURE_CONFIG.get('donchian_low', True):
                if len(low) >= 20:
                    features['donchian_low'] = low[-20:].min()
                else:
                    features['donchian_low'] = 0.0

            if FEATURE_CONFIG.get('momentum5', True):
                if len(close) >= 6:
                    features['momentum5'] = close.pct_change(5).values[-1]
                else:
                    features['momentum5'] = 0.0

            if FEATURE_CONFIG.get('momentum10', True):
                if len(close) >= 11:
                    features['momentum10'] = close.pct_change(10).values[-1]
                else:
                    features['momentum10'] = 0.0

            if FEATURE_CONFIG.get('mean_reversion_zscore', True):
                if len(close) >= 20:
                    mean = close[-20:].mean()
                    std = close[-20:].std()
                    features['mean_reversion_zscore'] = (close.values[-1] - mean) / (std + 1e-8)
                else:
                    features['mean_reversion_zscore'] = 0.0

            if FEATURE_CONFIG.get('volatility', True):
                if len(close) >= 20:
                    features['volatility'] = close[-20:].std()
                else:
                    features['volatility'] = 0.0

            if FEATURE_CONFIG.get('price_gap', True):
                if len(close) >= 2:
                    features['price_gap'] = close.values[-1] - close.values[-2]
                else:
                    features['price_gap'] = 0.0

            if FEATURE_CONFIG.get('shadow_ratio', True):
                if len(close) >= 1:
                    features['shadow_ratio'] = (high.values[-1] - low.values[-1]) / (abs(close.values[-1] - open_.values[-1]) + 1e-8)
                else:
                    features['shadow_ratio'] = 0.0

            if FEATURE_CONFIG.get('green_candles_10', True):
                if len(close) >= 10:
                    features['green_candles_10'] = int((close[-10:] > open_[-10:]).sum())
                else:
                    features['green_candles_10'] = 0

            if FEATURE_CONFIG.get('red_candles_10', True):
                if len(close) >= 10:
                    features['red_candles_10'] = int((close[-10:] < open_[-10:]).sum())
                else:
                    features['red_candles_10'] = 0

            if FEATURE_CONFIG.get('williams_vix_fix', True):
                if len(high) >= 22 and len(close) >= 1:
                    features['williams_vix_fix'] = (high[-22:].max() - close.values[-1]) / (high[-22:].max() + 1e-8)
                else:
                    features['williams_vix_fix'] = 0.0

        # ===== فیچرهای خاص برای نقاط ورود/خروج =====

        # کراس EMA9/EMA21
        if FEATURE_CONFIG.get('ema_cross_9_21', True):
            if len(close) >= 22:
                ema9 = close.ewm(span=9).mean().values
                ema21 = close.ewm(span=21).mean().values
                cross = (ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]) or (ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1])
                features['ema_cross_9_21'] = float(cross)
            else:
                features['ema_cross_9_21'] = 0.0

        # breakout/breakdown در 30 کندل اخیر
        if FEATURE_CONFIG.get('breakout_30', True):
            features['breakout_30'] = float(close.values[-1] > high[-30:].max() * 1.001) if len(close) >= 30 else 0.0
        if FEATURE_CONFIG.get('breakdown_30', True):
            features['breakdown_30'] = float(close.values[-1] < low[-30:].min() * 0.999) if len(close) >= 30 else 0.0

        # درصد کندل‌های سبز/قرمز در 20 کندل آخر
        if FEATURE_CONFIG.get('green_candle_ratio_20', True):
            greens = (close[-20:] > open_[-20:]).sum() if len(close)>=20 else 0
            features['green_candle_ratio_20'] = greens / 20 if len(close)>=20 else 0
        if FEATURE_CONFIG.get('red_candle_ratio_20', True):
            reds = (close[-20:] < open_[-20:]).sum() if len(close)>=20 else 0
            features['red_candle_ratio_20'] = reds / 20 if len(close)>=20 else 0

        # اختلاف قیمت فعلی با EMA50 و EMA200
        if FEATURE_CONFIG.get('price_ema50_diff', True):
            features['price_ema50_diff'] = close.values[-1] - safe_ema(close, 50)
        if FEATURE_CONFIG.get('price_ema200_diff', True):
            features['price_ema200_diff'] = close.values[-1] - safe_ema(close, 200)

        # حجم spike حرفه‌ای
        if FEATURE_CONFIG.get('vol_spike', True):
            if len(volume) >= 20:
                features['vol_spike'] = float(volume.values[-1] > volume[-20:].mean() * 1.5)
            else:
                features['vol_spike'] = 0.0

        # ATR spike
        if FEATURE_CONFIG.get('atr_spike', True):
            if len(high) >= 15 and len(low) >= 15 and len(close) >= 15:
                tr = pd.concat([
                    high[-15:] - low[-15:],
                    (high[-15:] - close[-15:].shift()).abs(),
                    (low[-15:] - close[-15:].shift()).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().values[-1]
                prev_tr = pd.concat([
                    high[-16:-1] - low[-16:-1],
                    (high[-16:-1] - close[-16:-1].shift()).abs(),
                    (low[-16:-1] - close[-16:-1].shift()).abs()
                ], axis=1).max(axis=1)
                prev_atr = prev_tr.rolling(14).mean().values[-1]
                features['atr_spike'] = float(atr > prev_atr * 1.5) if prev_atr > 0 else 0.0
            else:
                features['atr_spike'] = 0.0

        # wick ratio
        if FEATURE_CONFIG.get('wick_ratio', True):
            if len(close) >= 1:
                body = abs(close.values[-1] - open_.values[-1])
                high_wick = high.values[-1] - max(close.values[-1], open_.values[-1])
                low_wick = min(close.values[-1], open_.values[-1]) - low.values[-1]
                features['wick_ratio'] = (high_wick + low_wick) / (body + 1e-8)
            else:
                features['wick_ratio'] = 0.0

        # ==== کندل پترن‌ها (talib) - با استفاده از safe_pattern_value ====
        if talib is not None:
            if FEATURE_CONFIG.get('engulfing', True):
                features['engulfing'] = safe_pattern_value(talib.CDLENGULFING(open_, high, low, close))
                
            if FEATURE_CONFIG.get('hammer', True):
                features['hammer'] = safe_pattern_value(talib.CDLHAMMER(open_, high, low, close))
                
            if FEATURE_CONFIG.get('doji', True):
                features['doji'] = safe_pattern_value(talib.CDLDOJI(open_, high, low, close))
                
            if FEATURE_CONFIG.get('morning_star', True):
                features['morning_star'] = safe_pattern_value(talib.CDLMORNINGSTAR(open_, high, low, close))
                
            if FEATURE_CONFIG.get('evening_star', True):
                features['evening_star'] = safe_pattern_value(talib.CDLEVENINGSTAR(open_, high, low, close))
                
            if FEATURE_CONFIG.get('shooting_star', True):
                features['shooting_star'] = safe_pattern_value(talib.CDLSHOOTINGSTAR(open_, high, low, close))
        else:
            # اگر talib نیست، پیاده‌سازی ساده خودمان را انجام می‌دهیم
            
            # Doji: بدنه شمع کمتر از 10% کل طول شمع است
            if FEATURE_CONFIG.get('doji', True):
                if len(close) >= 1 and len(open_) >= 1 and len(high) >= 1 and len(low) >= 1:
                    body = abs(close.values[-1] - open_.values[-1])
                    range_size = high.values[-1] - low.values[-1]
                    features['doji'] = 1 if (range_size > 0 and body / range_size < 0.1) else 0
                else:
                    features['doji'] = 0
                    
            # Engulfing: شمع فعلی بدنه شمع قبلی را کاملاً در بر می‌گیرد
            if FEATURE_CONFIG.get('engulfing', True):
                if len(close) >= 2 and len(open_) >= 2:
                    bullish_engulfing = ((open_.values[-1] < open_.values[-2]) and 
                                       (close.values[-1] > close.values[-2]) and
                                       (close.values[-1] > open_.values[-2]) and
                                       (open_.values[-1] < close.values[-2]))
                    
                    bearish_engulfing = ((open_.values[-1] > open_.values[-2]) and
                                       (close.values[-1] < close.values[-2]) and
                                       (close.values[-1] < open_.values[-2]) and
                                       (open_.values[-1] > close.values[-2]))
                                       
                    features['engulfing'] = 100 if bullish_engulfing else (-100 if bearish_engulfing else 0)
                else:
                    features['engulfing'] = 0
                    
            # Hammer: بدنه کوچک در بالا، سایه بلند در پایین
            if FEATURE_CONFIG.get('hammer', True):
                if len(close) >= 1 and len(open_) >= 1 and len(high) >= 1 and len(low) >= 1:
                    body = abs(close.values[-1] - open_.values[-1])
                    upper_shadow = high.values[-1] - max(close.values[-1], open_.values[-1])
                    lower_shadow = min(close.values[-1], open_.values[-1]) - low.values[-1]
                    
                    features['hammer'] = 1 if (lower_shadow > 2 * body and upper_shadow < 0.2 * body) else 0
                else:
                    features['hammer'] = 0

        # === فیچرهای پیشرفته اضافی (بیش از 50 فیچر جدید) ===
        if candles_df is not None and len(close) > 30:
            # 1. فیچرهای روند و مومنتوم
            for period in [3, 7, 14, 30]:
                if len(close) > period:
                    # شتاب قیمت
                    features[f'price_momentum_{period}'] = close.pct_change(period).values[-1]
                    
                    # نسبت قیمت به میانگین متحرک دوره
                    ma = close.rolling(period).mean()
                    features[f'price_to_ma_{period}'] = close.values[-1] / ma.values[-1] if ma.values[-1] > 0 else 1.0
                    
                    # انحراف معیار دوره
                    features[f'std_{period}'] = close.rolling(period).std().values[-1]
                    
                    # حداکثر و حداقل قیمت دوره
                    features[f'max_price_{period}'] = close[-period:].max() / close.values[-1] - 1
                    features[f'min_price_{period}'] = close[-period:].min() / close.values[-1] - 1
                    
                    # نسبت‌های حجم
                    features[f'vol_change_{period}'] = volume.pct_change(period).values[-1]
                    
            # 2. فیچرهای نوسان پیشرفته
            if len(close) > 20:
                # شاخص چوب پرچم (Flag Pole Index)
                recent_trend = close[-20:].pct_change(5).values
                features['flag_pole_index'] = np.std(recent_trend)
                
                # Hurst Exponent (تقریب ساده)
                series = np.array(close[-20:])
                lags = range(2, 10)
                tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
                hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
                features['hurst_exponent'] = hurst * 2  # نرمال کردن بین 0 و 2
                
                # شاخص قدرت مومنتوم نسبی (RSI و MACD ترکیبی)
                rsi = safe_rsi(close, 14)
                macd_val, _, _ = safe_macd(close)
                features['rsi_macd_strength'] = (rsi/100) * np.sign(macd_val)
                
            # 3. فیچرهای الگوهای قیمت
            if len(close) > 30 and len(open_) > 30:
                # الگوی W (دابل باتم)
                lows = low[-30:].values
                if len(lows) >= 30:
                    lower_band = np.percentile(lows, 25)  # باند پایینی 25٪
                    bottoms = []
                    for i in range(1, len(lows)-1):
                        if lows[i] < lower_band and lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                            bottoms.append(i)
                    
                    # بررسی آیا دو کف با فاصله مناسب وجود دارد
                    double_bottom = 0
                    if len(bottoms) >= 2:
                        for i in range(len(bottoms)-1):
                            distance = bottoms[i+1] - bottoms[i]
                            if 5 <= distance <= 15:  # فاصله معقول بین دو کف
                                double_bottom = 1
                                break
                    
                    features['double_bottom'] = double_bottom
                    
                    # الگوی M (دابل تاپ)
                    highs = high[-30:].values
                    upper_band = np.percentile(highs, 75)  # باند بالایی 75٪
                    tops = []
                    for i in range(1, len(highs)-1):
                        if highs[i] > upper_band and highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                            tops.append(i)
                    
                    double_top = 0
                    if len(tops) >= 2:
                        for i in range(len(tops)-1):
                            distance = tops[i+1] - tops[i]
                            if 5 <= distance <= 15:  # فاصله معقول بین دو قله
                                double_top = 1
                                break
                    
                    features['double_top'] = double_top
            
            # 4. فیچرهای تشخیص فشار خرید/فروش
            if len(close) > 20 and len(volume) > 20:
                # فشار خرید (بالا رفتن قیمت با افزایش حجم)
                green_candles = close[-20:] > open_[-20:]
                green_volumes = volume[-20:][green_candles]
                red_volumes = volume[-20:][~green_candles]
                
                features['buying_pressure'] = green_volumes.mean() / volume[-20:].mean() if len(green_volumes) > 0 else 0
                features['selling_pressure'] = red_volumes.mean() / volume[-20:].mean() if len(red_volumes) > 0 else 0
                
                # شاخص تجمعی قدرت (خرید - فروش)
                features['strength_index'] = features['buying_pressure'] - features['selling_pressure']
            
            # 5. فیچرهای محاسبات آماری پیشرفته
            if len(close) > 50:
                # کورتوسیس (تیزی توزیع)
                returns = close.pct_change().dropna()
                if len(returns) > 10:
                    features['kurtosis'] = pd.Series(returns[-50:]).kurtosis()
                    
                    # چولگی (Skewness) - تقارن توزیع
                    features['skewness'] = pd.Series(returns[-50:]).skew()
                    
                    # Maximum Drawdown (حداکثر افت)
                    price_series = close[-50:].values
                    peak = price_series[0]
                    max_drawdown = 0
                    
                    for price in price_series:
                        if price > peak:
                            peak = price
                        drawdown = (peak - price) / peak
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    features['max_drawdown'] = max_drawdown
            
            # 6. فیچرهای ترکیبی متا
            # ترکیب اندیکاتورها برای تشخیص روند
            if 'rsi14' in features and 'macd' in features and 'ema20' in features and 'ema50' in features:
                trend_signals = []
                
                # RSI سیگنال
                if features['rsi14'] > 60:
                    trend_signals.append(1)  # صعودی
                elif features['rsi14'] < 40:
                    trend_signals.append(-1)  # نزولی
                else:
                    trend_signals.append(0)  # خنثی
                    
                # MACD سیگنال
                if features['macd'] > 0:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
                    
                # EMA سیگنال
                if features['ema20'] > features['ema50']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
                    
                # محاسبه میانگین سیگنال‌ها
                features['trend_meta_signal'] = sum(trend_signals) / len(trend_signals)

            # 7. فیچرهای مبتنی بر ارتباطات سمبل‌ها
            # این فیچرها برای زمانی که چندین سمبل داریم، مفید هستند
            # فعلاً فقط به عنوان مثال اضافه می‌شوند
            features['market_correlation'] = 0.5  # مثال
            features['sector_rotation'] = 0.0  # مثال
            
            # 8. فیچرهای الگوی هارمونیک (اعداد فیبوناچی)
            if len(high) > 50 and len(low) > 50:
                # محاسبه درصد بازگشت فیبوناچی
                high_point = high[-50:].max()
                low_point = low[-50:].min()
                range_size = high_point - low_point
                
                if range_size > 0:
                    fib_382 = high_point - range_size * 0.382
                    fib_500 = high_point - range_size * 0.5
                    fib_618 = high_point - range_size * 0.618
                    
                    # بررسی نزدیکی قیمت به سطوح فیبوناچی
                    current_price = close.values[-1]
                    features['fib_382_proximity'] = abs(current_price - fib_382) / range_size
                    features['fib_500_proximity'] = abs(current_price - fib_500) / range_size
                    features['fib_618_proximity'] = abs(current_price - fib_618) / range_size
                    
                    # آیا قیمت در محدوده الگوی گارتلی است؟
                    features['gartley_pattern'] = int(0.76 < features['fib_618_proximity'] < 0.854)
                    
                    # آیا قیمت در محدوده الگوی پروانه است؟
                    features['butterfly_pattern'] = int(1.27 < features['fib_618_proximity'] < 1.414)
                
            # 9. فیچرهای حافظه بلندمدت
            for lookback in [30, 60, 90, 120]:
                if len(close) > lookback:
                    # موقعیت قیمت نسبت به دامنه N روز اخیر
                    high_n = high[-lookback:].max()
                    low_n = low[-lookback:].min()
                    price_range = high_n - low_n
                    
                    if price_range > 0:
                        position_in_range = (close.values[-1] - low_n) / price_range
                        features[f'position_in_range_{lookback}'] = position_in_range
                        
                        # فاصله تا حداکثر قیمت N روز اخیر
                        features[f'distance_to_high_{lookback}'] = (high_n - close.values[-1]) / close.values[-1]
                        
                        # فاصله تا حداقل قیمت N روز اخیر
                        features[f'distance_to_low_{lookback}'] = (close.values[-1] - low_n) / close.values[-1]
            
            # 10. فیچرهای آشوب و پیچیدگی
            if len(close) > 50:
                # شاخص پیچیدگی (میزان نوسانات قیمت)
                returns = close.pct_change().dropna()
                if len(returns) > 10:
                    # تعداد تغییر جهت‌ها
                    direction_changes = np.sum(np.diff(np.signbit(returns[-30:])))
                    features['direction_change_count'] = direction_changes
                    
                    # نسبت سیگنال به نویز
                    mean_return = np.mean(returns[-30:])
                    std_return = np.std(returns[-30:])
                    features['signal_to_noise'] = abs(mean_return) / std_return if std_return > 0 else 0

    # =========== فیچرهای خبری و فاندامنتال ===========
    try:
        now_ts = candles_df['timestamp'].values[-1] if candles_df is not None and not candles_df.empty and 'timestamp' in candles_df else int(time.time())
    except:
        now_ts = int(time.time())
        
    # تعریف بازه‌های زمانی برای فیچرهای خبری
    ranges = {
        '1h': 1*60*60, '6h': 6*60*60, '12h': 12*60*60, '24h': 24*60*60,
        '36h': 36*60*60, '48h': 48*60*60, '50h': 50*60*60, '62h': 62*60*60,
    }
    weights = {'1h':1.0,'6h':0.8,'12h':0.7,'24h':0.6,'36h':0.5,'48h':0.4,'50h':0.3,'62h':0.2}
    weighted_score = 0.0
    total_weight = 0.0
    result = {}

    # اطمینان از اینکه news_df شامل اطلاعات و ستون‌های مورد نیاز است
    if news_df is not None and not news_df.empty:
        # تبدیل sentiment_score به float
        if 'sentiment_score' in news_df.columns:
            news_df['sentiment_score'] = pd.to_numeric(news_df['sentiment_score'], errors='coerce').fillna(0.0)
            
            # فیچرهای پایه خبری
            features['news_count'] = len(news_df)
            features['news_sentiment_mean'] = news_df['sentiment_score'].mean()
            features['news_sentiment_std'] = news_df['sentiment_score'].std()
            features['news_pos_count'] = news_df[news_df['sentiment_score'] > 0.1].shape[0]
            features['news_neg_count'] = news_df[news_df['sentiment_score'] < -0.1].shape[0]
            features['news_latest_sentiment'] = news_df['sentiment_score'].values[0] if len(news_df) > 0 else 0.0
            
            if 'content' in news_df.columns:
                features['news_content_len'] = news_df['content'].str.len().mean()
            else:
                features['news_content_len'] = 0.0

            # اضافه کردن فیچرهای مبتنی بر بازه زمانی
            for rng, seconds in ranges.items():
                # انتخاب ستون زمان مناسب
                ts_column = 'ts' if 'ts' in news_df.columns else 'published_at'
                
                if ts_column in news_df.columns:
                    # فیلتر کردن اخبار اخیر برای این بازه زمانی
                    recent = news_df[news_df[ts_column] >= now_ts - seconds]
                    result[f'news_count_{rng}'] = len(recent)
                    
                    if len(recent) > 0 and 'sentiment_score' in recent.columns:
                        # محاسبه فیچرهای خبری برای این بازه زمانی
                        s = recent['sentiment_score']
                        result[f'news_sentiment_mean_{rng}'] = s.mean()
                        result[f'news_sentiment_max_{rng}'] = s.max()
                        result[f'news_sentiment_min_{rng}'] = s.min()
                        result[f'news_pos_ratio_{rng}'] = (s > 0.1).mean()
                        result[f'news_neg_ratio_{rng}'] = (s < -0.1).mean()
                        
                        # امتیاز وزن‌دار
                        weighted_score += s.mean() * weights[rng]
                        total_weight += weights[rng]
                    else:
                        # مقادیر پیش‌فرض
                        result[f'news_sentiment_mean_{rng}'] = 0.0
                        result[f'news_sentiment_max_{rng}'] = 0.0
                        result[f'news_sentiment_min_{rng}'] = 0.0
                        result[f'news_pos_ratio_{rng}'] = 0.0
                        result[f'news_neg_ratio_{rng}'] = 0.0
                else:
                    # اگر ستون زمان نداریم، مقادیر پیش‌فرض را قرار می‌دهیم
                    result[f'news_count_{rng}'] = 0
                    result[f'news_sentiment_mean_{rng}'] = 0.0
                    result[f'news_sentiment_max_{rng}'] = 0.0
                    result[f'news_sentiment_min_{rng}'] = 0.0
                    result[f'news_pos_ratio_{rng}'] = 0.0
                    result[f'news_neg_ratio_{rng}'] = 0.0
                    
            # محاسبه امتیاز وزن‌دار
            if total_weight > 0:
                result['news_weighted_score'] = weighted_score / total_weight
            else:
                result['news_weighted_score'] = 0.0
                
            features.update(result)
            
            # اضافه کردن فیچر‌های پیشرفته خبری با FinBERT
            if USE_FINBERT and 'content' in news_df.columns:
                # تعداد اخبار با احساسات قوی مثبت
                features['strong_positive_news'] = len(news_df[news_df['sentiment_score'] > 0.8])
                
                # تعداد اخبار با احساسات قوی منفی
                features['strong_negative_news'] = len(news_df[news_df['sentiment_score'] < -0.8])
                
                # نسبت اخبار مثبت به منفی
                pos_count = features['news_pos_count']
                neg_count = features['news_neg_count']
                if neg_count > 0:
                    features['pos_to_neg_ratio'] = pos_count / neg_count
                else:
                    features['pos_to_neg_ratio'] = pos_count if pos_count > 0 else 1.0
                    
                # روند تغییر احساسات
                if len(news_df) >= 10:
                    first_half = news_df.iloc[:len(news_df)//2]['sentiment_score'].mean()
                    second_half = news_df.iloc[len(news_df)//2:]['sentiment_score'].mean()
                    features['sentiment_trend'] = second_half - first_half
                else:
                    features['sentiment_trend'] = 0.0

                # تنوع احساسات
                if len(news_df) >= 5:
                    features['sentiment_diversity'] = news_df['sentiment_score'].nunique() / len(news_df)
                else:
                    features['sentiment_diversity'] = 0.0
                
        else:
            # اگر sentiment_score نداریم، فیچرهای خبری را با مقادیر پیش‌فرض پر می‌کنیم
            logger.warning(f"No sentiment_score in news data for {symbol}")
            features['news_count'] = len(news_df)
            features['news_sentiment_mean'] = 0.0
            features['news_sentiment_std'] = 0.0
            features['news_pos_count'] = 0
            features['news_neg_count'] = 0
            features['news_latest_sentiment'] = 0.0
            features['news_content_len'] = 0.0
            features['news_weighted_score'] = 0.0
            
            # فیچرهای زمانی خبری را نیز با مقادیر پیش‌فرض پر می‌کنیم
            for rng in ranges:
                features[f'news_count_{rng}'] = 0
                for v in ['sentiment_mean','sentiment_max','sentiment_min','pos_ratio','neg_ratio']:
                    features[f'news_{v}_{rng}'] = 0.0
    else:
        # اگر اخباری موجود نیست، همه فیچرهای خبری را با مقادیر پیش‌فرض پر می‌کنیم
        features['news_count'] = 0
        features['news_sentiment_mean'] = 0.0
        features['news_sentiment_std'] = 0.0
        features['news_pos_count'] = 0
        features['news_neg_count'] = 0
        features['news_latest_sentiment'] = 0.0
        features['news_content_len'] = 0.0
        features['news_weighted_score'] = 0.0
        
        # فیچرهای زمانی خبری را نیز با مقادیر پیش‌فرض پر می‌کنیم
        for rng in ranges:
            features[f'news_count_{rng}'] = 0
            for v in ['sentiment_mean','sentiment_max','sentiment_min','pos_ratio','neg_ratio']:
                features[f'news_{v}_{rng}'] = 0.0

    # چاپ اطلاعات دیباگ
    if debug_info:
        print(f"=== Feature calculation debug for {symbol} ===")
        print(f"Candles: {debug_info.get('candle_count', 0)}, Price: {debug_info.get('price_latest', 0)}")
        if 'news_count' in debug_info:
            print(f"News: {debug_info.get('news_count', 0)} items, {debug_info.get('oldest_news', 'unknown')} to {debug_info.get('newest_news', 'unknown')}")

    # اطمینان از جایگزینی مقادیر inf/NaN با صفر
    features_df = pd.DataFrame([features])
    features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # گزارش تعداد فیچرهای غیر صفر برای اطلاعات
    non_zero_count = (features_df != 0).sum().sum()
    total_features = len(features_df.columns)
    logger.info(f"Generated {non_zero_count}/{total_features} non-zero features for {symbol}")
    print(f"Generated {non_zero_count}/{total_features} non-zero features for {symbol}")
    
    return features_df

def calculate_news_features(news_df, candles_df):
    """
    محاسبه فیچرهای مبتنی بر اخبار و سنتیمنت
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame()
    
    # تبدیل به دیتافریم
    result = pd.DataFrame()
    
    try:
        now_ts = candles_df['timestamp'].values[-1] if candles_df is not None and not candles_df.empty and 'timestamp' in candles_df else int(time.time())
    
        # بررسی وجود ستون sentiment_score
        if 'sentiment_score' in news_df.columns:
            # فیچرهای پایه
            result['news_count'] = [len(news_df)]
            result['news_sentiment_mean'] = [news_df['sentiment_score'].mean()]
            result['news_sentiment_std'] = [news_df['sentiment_score'].std()]
            
            # تعداد اخبار مثبت و منفی
            result['news_pos_count'] = [len(news_df[news_df['sentiment_score'] > 0.1])]
            result['news_neg_count'] = [len(news_df[news_df['sentiment_score'] < -0.1])]
            
            # تازه‌ترین خبر
            result['news_latest_sentiment'] = [news_df['sentiment_score'].iloc[0] if len(news_df) > 0 else 0.0]
            
            # طول محتوا
            if 'content' in news_df.columns:
                result['news_content_len'] = [news_df['content'].str.len().mean()]
            else:
                result['news_content_len'] = [0.0]
                
            # تعریف بازه‌های زمانی
            ranges = {
                '1h': 1*60*60, '6h': 6*60*60, '12h': 12*60*60, '24h': 24*60*60,
                '36h': 36*60*60, '48h': 48*60*60, '50h': 50*60*60, '62h': 62*60*60,
            }
            weights = {'1h': 1.0, '6h': 0.8, '12h': 0.7, '24h': 0.6, '36h': 0.5, '48h': 0.4, '50h': 0.3, '62h': 0.2}
            weighted_score = 0.0
            total_weight = 0.0
            
            # انتخاب ستون زمان مناسب
            ts_column = 'ts' if 'ts' in news_df.columns else 'published_at'
            
            if ts_column in news_df.columns:
                # محاسبه فیچرها برای هر بازه زمانی
                for rng, seconds in ranges.items():
                    recent = news_df[news_df[ts_column] >= now_ts - seconds]
                    result[f'news_count_{rng}'] = [len(recent)]
                    
                    if len(recent) > 0:
                        s = recent['sentiment_score']
                        result[f'news_sentiment_mean_{rng}'] = [s.mean()]
                        result[f'news_sentiment_max_{rng}'] = [s.max()]
                        result[f'news_sentiment_min_{rng}'] = [s.min()]
                        result[f'news_pos_ratio_{rng}'] = [(s > 0.1).mean()]
                        result[f'news_neg_ratio_{rng}'] = [(s < -0.1).mean()]
                        
                        # امتیاز وزن‌دار
                        weighted_score += s.mean() * weights[rng]
                        total_weight += weights[rng]
                    else:
                        result[f'news_sentiment_mean_{rng}'] = [0.0]
                        result[f'news_sentiment_max_{rng}'] = [0.0]
                        result[f'news_sentiment_min_{rng}'] = [0.0]
                        result[f'news_pos_ratio_{rng}'] = [0.0]
                        result[f'news_neg_ratio_{rng}'] = [0.0]
                
                # محاسبه امتیاز وزن‌دار نهایی
                if total_weight > 0:
                    result['news_weighted_score'] = [weighted_score / total_weight]
                else:
                    result['news_weighted_score'] = [0.0]
                    
                # فیچرهای پیشرفته خبری اضافی
                if USE_FINBERT and 'content' in news_df.columns:
                    # تعداد اخبار با احساسات قوی
                    result['strong_positive_news'] = [len(news_df[news_df['sentiment_score'] > 0.8])]
                    result['strong_negative_news'] = [len(news_df[news_df['sentiment_score'] < -0.8])]
                    
                    # نسبت اخبار مثبت به منفی
                    pos_count = result['news_pos_count'].iloc[0]
                    neg_count = result['news_neg_count'].iloc[0]
                    if neg_count > 0:
                        result['pos_to_neg_ratio'] = [pos_count / neg_count]
                    else:
                        result['pos_to_neg_ratio'] = [pos_count if pos_count > 0 else 1.0]
                    
                    # روند تغییر احساسات
                    if len(news_df) >= 10:
                        first_half = news_df.iloc[:len(news_df)//2]['sentiment_score'].mean()
                        second_half = news_df.iloc[len(news_df)//2:]['sentiment_score'].mean()
                        result['sentiment_trend'] = [second_half - first_half]
                    else:
                        result['sentiment_trend'] = [0.0]
    
    except Exception as e:
        logger.error(f"Error calculating news features: {e}")
        return pd.DataFrame()
    
    return result
