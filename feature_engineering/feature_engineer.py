import pandas as pd
import numpy as np
from .feature_config import FEATURE_CONFIG

try:
    import ta
except ImportError:
    ta = None

try:
    import talib
except ImportError:
    talib = None

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

def build_features(candles_df, news_df, symbol):
    features = {}
    # --- رفع مشکل ts برای اخبار ---
    if news_df is not None and not news_df.empty:
        if 'ts' not in news_df.columns:
            if 'published_at' in news_df.columns:
                news_df = news_df.copy()
                news_df['ts'] = pd.to_datetime(news_df['published_at']).values.astype('int64') // 10**9

    # =========== فیچرهای تکنیکال و آماری ===========
    if candles_df is not None and not candles_df.empty:
        close = candles_df['close']
        high = candles_df['high']
        low = candles_df['low']
        open_ = candles_df['open']
        volume = candles_df['volume']

        # EMA ها
        for ema_span in [5, 9, 10, 20, 21, 50, 100, 200]:
            k = f'ema{ema_span}'
            if FEATURE_CONFIG.get(k):
                features[k] = safe_ema(close, ema_span)

        # SMA ها
        for sma_win in [20, 50]:
            k = f'sma{sma_win}'
            if FEATURE_CONFIG.get(k):
                features[k] = safe_sma(close, sma_win)

        # TEMA
        if FEATURE_CONFIG.get('tema20'):
            features['tema20'] = safe_tema(close, 20)

        # RSI
        if FEATURE_CONFIG.get('rsi14'):
            features['rsi14'] = safe_rsi(close, 14)

        # ATR
        if FEATURE_CONFIG.get('atr14'):
            features['atr14'] = safe_atr(high, low, close, 14)

        # MACD
        macd, macd_signal, macd_hist = safe_macd(close, 12, 26, 9)
        if FEATURE_CONFIG.get('macd'): features['macd'] = macd
        if FEATURE_CONFIG.get('macd_signal'): features['macd_signal'] = macd_signal
        if FEATURE_CONFIG.get('macd_hist'): features['macd_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_lower, bb_width = safe_bb(close, 20)
        if FEATURE_CONFIG.get('bb_upper'): features['bb_upper'] = bb_upper
        if FEATURE_CONFIG.get('bb_lower'): features['bb_lower'] = bb_lower
        if FEATURE_CONFIG.get('bb_width'): features['bb_width'] = bb_width

        # OBV
        if FEATURE_CONFIG.get("obv"):
            features['obv'] = calculate_obv(candles_df)

        # VWAP
        if FEATURE_CONFIG.get("vwap"):
            features['vwap'] = calculate_vwap(candles_df)

        # Stochastic
        if FEATURE_CONFIG.get('stoch_k') or FEATURE_CONFIG.get('stoch_d'):
            if len(close) >= 14:
                low14 = low[-14:]
                high14 = high[-14:]
                stoch_k = 100 * (close.values[-1] - low14.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8)
                if FEATURE_CONFIG.get('stoch_k'):
                    features['stoch_k'] = stoch_k
                if FEATURE_CONFIG.get('stoch_d'):
                    features['stoch_d'] = pd.Series([stoch_k]).rolling(3).mean().values[-1]
            else:
                if FEATURE_CONFIG.get('stoch_k'):
                    features['stoch_k'] = 0.0
                if FEATURE_CONFIG.get('stoch_d'):
                    features['stoch_d'] = 0.0

        # CCI
        if FEATURE_CONFIG.get('cci'):
            if len(close) >= 20:
                tp = (high[-20:] + low[-20:] + close[-20:]) / 3
                cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
                features['cci'] = cci.values[-1]
            else:
                features['cci'] = 0.0

        # willr
        if FEATURE_CONFIG.get('willr'):
            if len(close) >= 14:
                low14 = low[-14:]
                high14 = high[-14:]
                willr = (high14.values[-1] - close.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8) * -100
                features['willr'] = willr
            else:
                features['willr'] = 0.0

        # ROC
        if FEATURE_CONFIG.get('roc'):
            if len(close) >= 11:
                features['roc'] = close.pct_change(periods=10).values[-1]
            else:
                features['roc'] = 0.0

        # PSAR (نمونه: فقط آخرین قیمت)
        if FEATURE_CONFIG.get('psar'):
            features['psar'] = close.values[-1]

        # candle_change
        if FEATURE_CONFIG.get('candle_change'):
            if len(close) >= 2:
                features['candle_change'] = close.pct_change().values[-1]
            else:
                features['candle_change'] = 0.0

        # candle_range
        if FEATURE_CONFIG.get('candle_range'):
            features['candle_range'] = (high.values[-1] - low.values[-1])

        # volume_mean
        if FEATURE_CONFIG.get('volume_mean'):
            if len(volume) >= 20:
                features['volume_mean'] = volume[-20:].mean()
            else:
                features['volume_mean'] = volume.mean()

        # volume_spike
        if FEATURE_CONFIG.get('volume_spike'):
            if len(volume) >= 20:
                features['volume_spike'] = float(volume.values[-1] > np.mean(volume[-20:]) * 1.5)
            else:
                features['volume_spike'] = 0.0

        # قیمت‌های کندل آخر
        for k in ['close','open','high','low','volume']:
            if FEATURE_CONFIG.get(k):
                features[k] = candles_df[k].values[-1]

        # ====== اندیکاتورهای مدرن و پرایس اکشن (با ta) ======
        if ta is not None:
            adx_window = 14
            adx_min = adx_window + 1
            if FEATURE_CONFIG.get('adx14'):
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

            if FEATURE_CONFIG.get('supertrend'):
                try:
                    features['supertrend'] = ta.trend.stc(close).values[-1]
                except Exception:
                    features['supertrend'] = 0.0

            if FEATURE_CONFIG.get('donchian_high'):
                if len(high) >= 20:
                    features['donchian_high'] = high[-20:].max()
                else:
                    features['donchian_high'] = 0.0

            if FEATURE_CONFIG.get('donchian_low'):
                if len(low) >= 20:
                    features['donchian_low'] = low[-20:].min()
                else:
                    features['donchian_low'] = 0.0

            if FEATURE_CONFIG.get('momentum5'):
                if len(close) >= 6:
                    features['momentum5'] = close.pct_change(5).values[-1]
                else:
                    features['momentum5'] = 0.0

            if FEATURE_CONFIG.get('momentum10'):
                if len(close) >= 11:
                    features['momentum10'] = close.pct_change(10).values[-1]
                else:
                    features['momentum10'] = 0.0

            if FEATURE_CONFIG.get('mean_reversion_zscore'):
                if len(close) >= 20:
                    mean = close[-20:].mean()
                    std = close[-20:].std()
                    features['mean_reversion_zscore'] = (close.values[-1] - mean) / (std + 1e-8)
                else:
                    features['mean_reversion_zscore'] = 0.0

            if FEATURE_CONFIG.get('volatility'):
                if len(close) >= 20:
                    features['volatility'] = close[-20:].std()
                else:
                    features['volatility'] = 0.0

            if FEATURE_CONFIG.get('price_gap'):
                if len(close) >= 2:
                    features['price_gap'] = close.values[-1] - close.values[-2]
                else:
                    features['price_gap'] = 0.0

            if FEATURE_CONFIG.get('shadow_ratio'):
                if len(close) >= 1:
                    features['shadow_ratio'] = (high.values[-1] - low.values[-1]) / (abs(close.values[-1] - open_.values[-1]) + 1e-8)
                else:
                    features['shadow_ratio'] = 0.0

            if FEATURE_CONFIG.get('green_candles_10'):
                if len(close) >= 10:
                    features['green_candles_10'] = int((close[-10:] > open_[-10:]).sum())
                else:
                    features['green_candles_10'] = 0

            if FEATURE_CONFIG.get('red_candles_10'):
                if len(close) >= 10:
                    features['red_candles_10'] = int((close[-10:] < open_[-10:]).sum())
                else:
                    features['red_candles_10'] = 0

            if FEATURE_CONFIG.get('williams_vix_fix'):
                if len(high) >= 22 and len(close) >= 1:
                    features['williams_vix_fix'] = (high[-22:].max() - close.values[-1]) / (high[-22:].max() + 1e-8)
                else:
                    features['williams_vix_fix'] = 0.0

        # ===== فیچرهای خاص برای نقاط ورود/خروج =====

        # کراس EMA9/EMA21
        if FEATURE_CONFIG.get('ema_cross_9_21'):
            if len(close) >= 22:
                ema9 = close.ewm(span=9).mean().values
                ema21 = close.ewm(span=21).mean().values
                cross = (ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]) or (ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1])
                features['ema_cross_9_21'] = float(cross)
            else:
                features['ema_cross_9_21'] = 0.0

        # breakout/breakdown در 30 کندل اخیر
        if FEATURE_CONFIG.get('breakout_30'):
            features['breakout_30'] = float(close.values[-1] > high[-30:].max() * 1.001) if len(close) >= 30 else 0.0
        if FEATURE_CONFIG.get('breakdown_30'):
            features['breakdown_30'] = float(close.values[-1] < low[-30:].min() * 0.999) if len(close) >= 30 else 0.0

        # درصد کندل‌های سبز/قرمز در 20 کندل آخر
        if FEATURE_CONFIG.get('green_candle_ratio_20'):
            greens = (close[-20:] > open_[-20:]).sum() if len(close)>=20 else 0
            features['green_candle_ratio_20'] = greens / 20 if len(close)>=20 else 0
        if FEATURE_CONFIG.get('red_candle_ratio_20'):
            reds = (close[-20:] < open_[-20:]).sum() if len(close)>=20 else 0
            features['red_candle_ratio_20'] = reds / 20 if len(close)>=20 else 0

        # اختلاف قیمت فعلی با EMA50 و EMA200
        if FEATURE_CONFIG.get('price_ema50_diff'):
            features['price_ema50_diff'] = close.values[-1] - safe_ema(close, 50)
        if FEATURE_CONFIG.get('price_ema200_diff'):
            features['price_ema200_diff'] = close.values[-1] - safe_ema(close, 200)

        # حجم spike حرفه‌ای
        if FEATURE_CONFIG.get('vol_spike'):
            if len(volume) >= 20:
                features['vol_spike'] = float(volume.values[-1] > volume[-20:].mean() * 1.5)
            else:
                features['vol_spike'] = 0.0

        # ATR spike
        if FEATURE_CONFIG.get('atr_spike'):
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
        if FEATURE_CONFIG.get('wick_ratio'):
            if len(close) >= 1:
                body = abs(close.values[-1] - open_.values[-1])
                high_wick = high.values[-1] - max(close.values[-1], open_.values[-1])
                low_wick = min(close.values[-1], open_.values[-1]) - low.values[-1]
                features['wick_ratio'] = (high_wick + low_wick) / (body + 1e-8)
            else:
                features['wick_ratio'] = 0.0

        # ==== کندل پترن‌ها (talib) ====
        if talib is not None:
            if FEATURE_CONFIG.get('engulfing'):
                if len(close) >= 2:
                    features['engulfing'] = talib.CDLENGULFING(open_, high, low, close)[-1]
                else:
                    features['engulfing'] = 0.0

            if FEATURE_CONFIG.get('hammer'):
                if len(close) >= 1:
                    features['hammer'] = talib.CDLHAMMER(open_, high, low, close)[-1]
                else:
                    features['hammer'] = 0.0

            if FEATURE_CONFIG.get('doji'):
                if len(close) >= 1:
                    features['doji'] = talib.CDLDOJI(open_, high, low, close)[-1]
                else:
                    features['doji'] = 0.0

            if FEATURE_CONFIG.get('morning_star'):
                if len(close) >= 3:
                    features['morning_star'] = talib.CDLMORNINGSTAR(open_, high, low, close)[-1]
                else:
                    features['morning_star'] = 0.0

            if FEATURE_CONFIG.get('shooting_star'):
                if len(close) >= 1:
                    features['shooting_star'] = talib.CDLSHOOTINGSTAR(open_, high, low, close)[-1]
                else:
                    features['shooting_star'] = 0.0

    # =========== فیچرهای خبری و فاندامنتال ===========
    now_ts = candles_df['timestamp'].values[-1] if candles_df is not None and not candles_df.empty and 'timestamp' in candles_df else int(pd.Timestamp.now().timestamp())
    ranges = {
        '1h': 1*60*60, '6h': 6*60*60, '12h': 12*60*60, '24h': 24*60*60,
        '36h': 36*60*60, '48h': 48*60*60, '50h': 50*60*60, '62h': 62*60*60,
    }
    weights = {'1h':1.0,'6h':0.8,'12h':0.7,'24h':0.6,'36h':0.5,'48h':0.4,'50h':0.3,'62h':0.2}
    weighted_score = 0.0
    total_weight = 0.0
    result = {}

    if news_df is not None and not news_df.empty:
        features['news_count'] = len(news_df)
        features['news_sentiment_mean'] = news_df['sentiment_score'].astype(float).mean() if 'sentiment_score' in news_df else 0.0
        features['news_sentiment_std'] = news_df['sentiment_score'].astype(float).std() if 'sentiment_score' in news_df else 0.0
        features['news_pos_count'] = news_df[news_df['sentiment_score'].astype(float) > 0.1].shape[0] if 'sentiment_score' in news_df else 0
        features['news_neg_count'] = news_df[news_df['sentiment_score'].astype(float) < -0.1].shape[0] if 'sentiment_score' in news_df else 0
        features['news_latest_sentiment'] = news_df['sentiment_score'].astype(float).values[0] if 'sentiment_score' in news_df else 0.0
        features['news_content_len'] = news_df['content'].str.len().mean() if 'content' in news_df else 0.0

        for rng, seconds in ranges.items():
            recent = news_df[news_df['ts'] >= now_ts-seconds]
            result[f'news_count_{rng}'] = len(recent)
            if len(recent) > 0 and 'sentiment_score' in recent:
                s = recent['sentiment_score'].astype(float)
                result[f'news_sentiment_mean_{rng}'] = s.mean()
                result[f'news_sentiment_max_{rng}'] = s.max()
                result[f'news_sentiment_min_{rng}'] = s.min()
                result[f'news_pos_ratio_{rng}'] = (s > 0.1).mean()
                result[f'news_neg_ratio_{rng}'] = (s < -0.1).mean()
                weighted_score += s.mean() * weights[rng]
                total_weight += weights[rng]
            else:
                for v in ['sentiment_mean','sentiment_max','sentiment_min','pos_ratio','neg_ratio']:
                    result[f'news_{v}_{rng}'] = 0.0
        if total_weight > 0:
            result['news_weighted_score'] = weighted_score / total_weight
        else:
            result['news_weighted_score'] = 0.0
        features.update(result)
    else:
        for name in ['news_count','news_sentiment_mean','news_sentiment_std','news_pos_count',
                     'news_neg_count','news_latest_sentiment','news_content_len']:
            features[name] = 0.0
        for rng in ranges:
            features[f'news_count_{rng}'] = 0.0
            for v in ['sentiment_mean','sentiment_max','sentiment_min','pos_ratio','neg_ratio']:
                features[f'news_{v}_{rng}'] = 0.0
        features['news_weighted_score'] = 0.0

    features_df = pd.DataFrame([features])
    features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features_df
