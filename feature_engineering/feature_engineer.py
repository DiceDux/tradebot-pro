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

    # =========================== تکنیکال ===========================
    if candles_df is not None and not candles_df.empty:
        close = candles_df['close']
        high = candles_df['high']
        low = candles_df['low']
        volume = candles_df['volume']

        # Moving Averages
        features['ema5'] = close.ewm(span=5).mean().values[-1]
        features['ema10'] = close.ewm(span=10).mean().values[-1]
        features['ema20'] = close.ewm(span=20).mean().values[-1]
        features['ema50'] = close.ewm(span=50).mean().values[-1]
        features['ema100'] = close.ewm(span=100).mean().values[-1]
        features['ema200'] = close.ewm(span=200).mean().values[-1]
        features['sma20'] = close.rolling(window=20).mean().values[-1]
        features['sma50'] = close.rolling(window=50).mean().values[-1]
        features['tema20'] = 3*close.ewm(span=20).mean().values[-1] - 3*close.ewm(span=20).mean().ewm(span=20).mean().values[-1] + close.ewm(span=20).mean().ewm(span=20).mean().ewm(span=20).mean().values[-1]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        features['rsi14'] = 100 - (100 / (1 + rs.values[-1])) if rs.values[-1] is not np.nan else 50

        # ATR
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr14'] = tr.rolling(14).mean().values[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd.values[-1]
        features['macd_signal'] = signal.values[-1]
        features['macd_hist'] = (macd - signal).values[-1]

        # Bollinger Bands
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        features['bb_upper'] = (bb_mid + 2 * bb_std).values[-1]
        features['bb_lower'] = (bb_mid - 2 * bb_std).values[-1]
        features['bb_width'] = (features['bb_upper'] - features['bb_lower'])

        # OBV
        features['obv'] = calculate_obv(candles_df)

        # VWAP
        features['vwap'] = calculate_vwap(candles_df)

        # Stochastic Oscillator
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close.values[-1] - low14.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8)
        features['stoch_d'] = pd.Series([features['stoch_k']]).rolling(3).mean().values[-1]

        # CCI
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        features['cci'] = cci.values[-1]

        # Williams %R
        willr = (high14.values[-1] - close.values[-1]) / (high14.values[-1] - low14.values[-1] + 1e-8) * -100
        features['willr'] = willr

        # Momentum
        features['roc'] = close.pct_change(periods=10).values[-1]

        # PSAR (ساده)
        features['psar'] = close.values[-1]  # جایگزین کن با پیاده‌سازی دقیق‌تر اگر خواستی

        # سایر ویژگی‌های کندل
        features['candle_change'] = close.pct_change().values[-1]
        features['candle_range'] = (high - low).values[-1]
        features['volume_mean'] = volume.rolling(20).mean().values[-1]
        features['volume_spike'] = (volume.values[-1] > features['volume_mean'] * 1.5)

        # قیمت و حجم فعلی
        features['close'] = close.values[-1]
        features['open'] = candles_df['open'].values[-1]
        features['high'] = high.values[-1]
        features['low'] = low.values[-1]
        features['volume'] = volume.values[-1]

    else:
        # مقادیر پیش‌فرض
        for name in ['ema5','ema10','ema20','ema50','ema100','ema200','sma20','sma50','tema20',
                     'rsi14','atr14','macd','macd_signal','macd_hist','bb_upper','bb_lower','bb_width',
                     'obv','vwap','stoch_k','stoch_d','cci','willr','roc','psar','candle_change','candle_range',
                     'volume_mean','volume_spike','close','open','high','low','volume']:
            features[name] = 0.0

    # ========================== فاندامنتال/اخبار ==========================
    if news_df is not None and not news_df.empty:
        # آمار اخبار
        features['news_count'] = len(news_df)
        features['news_sentiment_mean'] = news_df['sentiment_score'].astype(float).mean() if 'sentiment_score' in news_df else 0.0
        features['news_sentiment_std'] = news_df['sentiment_score'].astype(float).std() if 'sentiment_score' in news_df else 0.0
        features['news_pos_count'] = news_df[news_df['sentiment_score'].astype(float) > 0.1].shape[0] if 'sentiment_score' in news_df else 0
        features['news_neg_count'] = news_df[news_df['sentiment_score'].astype(float) < -0.1].shape[0] if 'sentiment_score' in news_df else 0
        features['news_latest_sentiment'] = news_df['sentiment_score'].astype(float).values[0] if 'sentiment_score' in news_df else 0.0
        # میانگین طول محتوا
        features['news_content_len'] = news_df['content'].str.len().mean() if 'content' in news_df else 0.0
    else:
        for name in ['news_count','news_sentiment_mean','news_sentiment_std','news_pos_count',
                     'news_neg_count','news_latest_sentiment','news_content_len']:
            features[name] = 0.0

    # ======= ویژگی‌های ترکیبی/آماری قابل گسترش =======
    # برای توسعه بعدی: فیچرهای ترکیبی، واگرایی، اخبار کل بازار و غیره

    # ========================== خروجی نهایی ==========================
    features_df = pd.DataFrame([features])
    features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features_df

