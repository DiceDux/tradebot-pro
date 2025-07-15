import pandas as pd
import numpy as np
from feature_config import FEATURE_CONFIG

try:
    import ta
except ImportError:
    ta = None

try:
    import talib
except ImportError:
    talib = None

def calculate_obv(df):
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

        # فیچرهای کلاسیک و میانگین‌ها
        if FEATURE_CONFIG.get('ema5'): features['ema5'] = close.ewm(span=5).mean().values[-1]
        if FEATURE_CONFIG.get('ema10'): features['ema10'] = close.ewm(span=10).mean().values[-1]
        if FEATURE_CONFIG.get('ema20'): features['ema20'] = close.ewm(span=20).mean().values[-1]
        if FEATURE_CONFIG.get('ema50'): features['ema50'] = close.ewm(span=50).mean().values[-1]
        if FEATURE_CONFIG.get('ema100'): features['ema100'] = close.ewm(span=100).mean().values[-1]
        if FEATURE_CONFIG.get('ema200'): features['ema200'] = close.ewm(span=200).mean().values[-1]
        if FEATURE_CONFIG.get('sma20'): features['sma20'] = close.rolling(window=20).mean().values[-1]
        if FEATURE_CONFIG.get('sma50'): features['sma50'] = close.rolling(window=50).mean().values[-1]
        if FEATURE_CONFIG.get('tema20'):
            ema1 = close.ewm(span=20).mean()
            ema2 = ema1.ewm(span=20).mean()
            ema3 = ema2.ewm(span=20).mean()
            features['tema20'] = 3 * (ema1.values[-1] - ema2.values[-1]) + ema3.values[-1]
        if FEATURE_CONFIG.get('rsi14'):
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss.replace(0, np.nan))
            features['rsi14'] = 100 - (100 / (1 + rs.values[-1])) if not np.isnan(rs.values[-1]) else 50
        if FEATURE_CONFIG.get('atr14'):
            tr = pd.concat([
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            features['atr14'] = tr.rolling(14).mean().values[-1]
        if FEATURE_CONFIG.get('macd') or FEATURE_CONFIG.get('macd_signal') or FEATURE_CONFIG.get('macd_hist'):
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            if FEATURE_CONFIG.get('macd'): features['macd'] = macd.values[-1]
            if FEATURE_CONFIG.get('macd_signal'): features['macd_signal'] = signal.values[-1]
            if FEATURE_CONFIG.get('macd_hist'): features['macd_hist'] = (macd - signal).values[-1]
        if FEATURE_CONFIG.get('bb_upper') or FEATURE_CONFIG.get('bb_lower') or FEATURE_CONFIG.get('bb_width'):
            bb_mid = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            if FEATURE_CONFIG.get('bb_upper'): features['bb_upper'] = (bb_mid + 2 * bb_std).values[-1]
            if FEATURE_CONFIG.get('bb_lower'): features['bb_lower'] = (bb_mid - 2 * bb_std).values[-1]
            if FEATURE_CONFIG.get('bb_width'): features['bb_width'] = (features.get('bb_upper', 0) - features.get('bb_lower', 0))
        if FEATURE_CONFIG.get("obv"): features['obv'] = calculate_obv(candles_df)
        if FEATURE_CONFIG.get("vwap"): features['vwap'] = calculate_vwap(candles_df)
        if FEATURE_CONFIG.get('stoch_k') or FEATURE_CONFIG.get('stoch_d'):
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            stoch_k = 100 * (close.values[-1] - low14.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8)
            if FEATURE_CONFIG.get('stoch_k'): features['stoch_k'] = stoch_k
            if FEATURE_CONFIG.get('stoch_d'): features['stoch_d'] = pd.Series([stoch_k]).rolling(3).mean().values[-1]
        if FEATURE_CONFIG.get('cci'):
            tp = (high + low + close) / 3
            cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            features['cci'] = cci.values[-1]
        if FEATURE_CONFIG.get('willr'):
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            willr = (high14.values[-1] - close.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8) * -100
            features['willr'] = willr
        if FEATURE_CONFIG.get('roc'): features['roc'] = close.pct_change(periods=10).values[-1]
        if FEATURE_CONFIG.get('psar'): features['psar'] = close.values[-1]
        if FEATURE_CONFIG.get('candle_change'): features['candle_change'] = close.pct_change().values[-1]
        if FEATURE_CONFIG.get('candle_range'): features['candle_range'] = (high - low).values[-1]
        if FEATURE_CONFIG.get('volume_mean'): features['volume_mean'] = volume.rolling(20).mean().values[-1]
        if FEATURE_CONFIG.get('volume_spike'): features['volume_spike'] = float(volume.values[-1] > features.get('volume_mean', 0) * 1.5)
        for k in ['close','open','high','low','volume']:
            if FEATURE_CONFIG.get(k): features[k] = candles_df[k].values[-1]

        # فیچرهای مدرن و آماری و پرایس اکشن
        if ta is not None:
            if FEATURE_CONFIG.get('adx14'):
                features['adx14'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().values[-1]
            if FEATURE_CONFIG.get('supertrend'):
                try:
                    features['supertrend'] = ta.trend.stc(close).values[-1]
                except Exception: features['supertrend'] = np.nan
            if FEATURE_CONFIG.get('donchian_high'): features['donchian_high'] = high.rolling(20).max().values[-1]
            if FEATURE_CONFIG.get('donchian_low'): features['donchian_low'] = low.rolling(20).min().values[-1]
            if FEATURE_CONFIG.get('momentum5'): features['momentum5'] = close.pct_change(5).values[-1]
            if FEATURE_CONFIG.get('momentum10'): features['momentum10'] = close.pct_change(10).values[-1]
            if FEATURE_CONFIG.get('mean_reversion_zscore'):
                mean = close.rolling(20).mean().values[-1]
                std = close.rolling(20).std().values[-1]
                features['mean_reversion_zscore'] = (close.values[-1] - mean) / (std + 1e-8)
            if FEATURE_CONFIG.get('volatility'): features['volatility'] = close.rolling(20).std().values[-1]
            if FEATURE_CONFIG.get('price_gap'): features['price_gap'] = close.values[-1] - close.values[-2]
            if FEATURE_CONFIG.get('shadow_ratio'):
                features['shadow_ratio'] = (high.values[-1] - low.values[-1]) / (abs(close.values[-1] - open_.values[-1]) + 1e-8)
            if FEATURE_CONFIG.get('green_candles_10'): features['green_candles_10'] = int((close[-10:] > open_[-10:]).sum())
            if FEATURE_CONFIG.get('red_candles_10'): features['red_candles_10'] = int((close[-10:] < open_[-10:]).sum())
            if FEATURE_CONFIG.get('williams_vix_fix'):
                features['williams_vix_fix'] = (high.rolling(22).max().values[-1] - close.values[-1]) / (high.rolling(22).max().values[-1] + 1e-8)

        # کندل پترن‌ها (talib)
        if talib is not None:
            if FEATURE_CONFIG.get('engulfing'): features['engulfing'] = talib.CDLENGULFING(open_, high, low, close)[-1]
            if FEATURE_CONFIG.get('hammer'): features['hammer'] = talib.CDLHAMMER(open_, high, low, close)[-1]
            if FEATURE_CONFIG.get('doji'): features['doji'] = talib.CDLDOJI(open_, high, low, close)[-1]
            if FEATURE_CONFIG.get('morning_star'): features['morning_star'] = talib.CDLMORNINGSTAR(open_, high, low, close)[-1]
            if FEATURE_CONFIG.get('shooting_star'): features['shooting_star'] = talib.CDLSHOOTINGSTAR(open_, high, low, close)[-1]
            # بقیه پترن‌های talib را هم می‌توانی اضافه کنی

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
