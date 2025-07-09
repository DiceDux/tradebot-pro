import pandas as pd
import numpy as np

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

    # --- رفع مشکل ts ---
    if news_df is not None and not news_df.empty:
        if 'ts' not in news_df.columns:
            if 'published_at' in news_df.columns:
                news_df = news_df.copy()
                news_df['ts'] = pd.to_datetime(news_df['published_at']).astype(int) // 10**9

    # =========== تکنیکال ===========
    if candles_df is not None and not candles_df.empty:
        close = candles_df['close']
        high = candles_df['high']
        low = candles_df['low']
        volume = candles_df['volume']
        features['ema5'] = close.ewm(span=5).mean().values[-1]
        features['ema10'] = close.ewm(span=10).mean().values[-1]
        features['ema20'] = close.ewm(span=20).mean().values[-1]
        features['ema50'] = close.ewm(span=50).mean().values[-1]
        features['ema100'] = close.ewm(span=100).mean().values[-1]
        features['ema200'] = close.ewm(span=200).mean().values[-1]
        features['sma20'] = close.rolling(window=20).mean().values[-1]
        features['sma50'] = close.rolling(window=50).mean().values[-1]
        features['tema20'] = 3*close.ewm(span=20).mean().values[-1] - 3*close.ewm(span=20).mean().ewm(span=20).mean().values[-1] + close.ewm(span=20).mean().ewm(span=20).mean().ewm(span=20).mean().values[-1]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        features['rsi14'] = 100 - (100 / (1 + rs.values[-1])) if rs.values[-1] is not np.nan else 50
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr14'] = tr.rolling(14).mean().values[-1]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd.values[-1]
        features['macd_signal'] = signal.values[-1]
        features['macd_hist'] = (macd - signal).values[-1]
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        features['bb_upper'] = (bb_mid + 2 * bb_std).values[-1]
        features['bb_lower'] = (bb_mid - 2 * bb_std).values[-1]
        features['bb_width'] = (features['bb_upper'] - features['bb_lower'])
        features['obv'] = calculate_obv(candles_df)
        features['vwap'] = calculate_vwap(candles_df)
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close.values[-1] - low14.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8)
        features['stoch_d'] = pd.Series([features['stoch_k']]).rolling(3).mean().values[-1]
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        features['cci'] = cci.values[-1]
        willr = (high14.values[-1] - close.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8) * -100
        features['willr'] = willr
        features['roc'] = close.pct_change(periods=10).values[-1]
        features['psar'] = close.values[-1]
        features['candle_change'] = close.pct_change().values[-1]
        features['candle_range'] = (high - low).values[-1]
        features['volume_mean'] = volume.rolling(20).mean().values[-1]
        features['volume_spike'] = float(volume.values[-1] > features['volume_mean'] * 1.5)
        features['close'] = close.values[-1]
        features['open'] = candles_df['open'].values[-1]
        features['high'] = high.values[-1]
        features['low'] = low.values[-1]
        features['volume'] = volume.values[-1]
    else:
        for name in [
            'ema5','ema10','ema20','ema50','ema100','ema200','sma20','sma50','tema20',
            'rsi14','atr14','macd','macd_signal','macd_hist','bb_upper','bb_lower','bb_width',
            'obv','vwap','stoch_k','stoch_d','cci','willr','roc','psar','candle_change','candle_range',
            'volume_mean','volume_spike','close','open','high','low','volume'
        ]:
            features[name] = 0.0

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
