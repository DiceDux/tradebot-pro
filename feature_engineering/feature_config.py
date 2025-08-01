FEATURE_CONFIG = {
    # === فیچرهای قیمت پایه ===
    "close": True,
    "open": True,
    "high": True,
    "low": True,
    "volume": True,

    # === میانگین‌های متحرک ===
    "ema5": True,
    "ema9": True,
    "ema10": True,
    "ema20": True,
    "ema21": True,
    "ema50": True,
    "ema100": True,
    "ema200": True,
    "sma20": True,
    "sma50": True,
    "tema20": True,

    # === اسیلاتورها ===
    "rsi14": True,
    "stoch_k": True,
    "stoch_d": True,
    "macd": True,
    "macd_signal": True,
    "macd_hist": True,
    "cci": True,
    "willr": True,
    "roc": True,

    # === نوسان و باندها ===
    "atr14": True,
    "bb_upper": True,
    "bb_lower": True,
    "bb_width": True,
    "donchian_high": True,
    "donchian_low": True,
    "volatility": True,
    "psar": True,

    # === فیچرهای حجم ===
    "volume_mean": True,
    "volume_spike": True,
    "vol_spike": True,
    "obv": True,
    "vwap": True,

    # === الگوهای کندل ===
    "doji": True,
    "engulfing": True,
    "hammer": True,
    "morning_star": True,
    "shooting_star": True,
    "evening_star": True,
    "candle_range": True,
    "shadow_ratio": True,
    "wick_ratio": True,
    "candle_change": True,

    # === روند و مومنتوم ===
    "adx14": True,
    "supertrend": True,
    "momentum5": True,
    "momentum10": True,
    "mean_reversion_zscore": True,
    "ema_cross_9_21": True,
    "price_gap": True,
    "price_ema50_diff": True,
    "price_ema200_diff": True,

    # === شکست‌ها ===
    "breakout_30": True,
    "breakdown_30": True,

    # === کندل‌های متوالی ===
    "green_candles_10": True,
    "red_candles_10": True,
    "green_candle_ratio_20": True,
    "red_candle_ratio_20": True,

    # === اخبار پایه ===
    "news_count": True,
    "news_sentiment_mean": True,
    "news_sentiment_std": True,
    "news_pos_count": True,
    "news_neg_count": True,
    "news_latest_sentiment": True,
    "news_content_len": True,
    "news_weighted_score": True,

    # === اخبار زمان‌بندی شده ===
    "news_count_1h": True, 
    "news_sentiment_mean_1h": True,
    "news_sentiment_max_1h": True, 
    "news_sentiment_min_1h": True,
    "news_pos_ratio_1h": True,
    "news_neg_ratio_1h": True,
    
    "news_count_6h": True,
    "news_sentiment_mean_6h": True,
    "news_sentiment_max_6h": True,
    "news_sentiment_min_6h": True,
    "news_pos_ratio_6h": True,
    "news_neg_ratio_6h": True,
    
    "news_count_12h": True,
    "news_sentiment_mean_12h": True,
    "news_sentiment_max_12h": True,
    "news_sentiment_min_12h": True,
    "news_pos_ratio_12h": True,
    "news_neg_ratio_12h": True,
    
    "news_count_24h": True,
    "news_sentiment_mean_24h": True,
    "news_sentiment_max_24h": True,
    "news_sentiment_min_24h": True,
    "news_pos_ratio_24h": True,
    "news_neg_ratio_24h": True,
    
    "news_count_36h": True,
    "news_sentiment_mean_36h": True,
    "news_sentiment_max_36h": True,
    "news_sentiment_min_36h": True,
    "news_pos_ratio_36h": True,
    "news_neg_ratio_36h": True,
    
    "news_count_48h": True,
    "news_sentiment_mean_48h": True,
    "news_sentiment_max_48h": True,
    "news_sentiment_min_48h": True,
    "news_pos_ratio_48h": True,
    "news_neg_ratio_48h": True,
    
    "news_count_50h": True,
    "news_sentiment_mean_50h": True,
    "news_sentiment_max_50h": True,
    "news_sentiment_min_50h": True,
    "news_pos_ratio_50h": True,
    "news_neg_ratio_50h": True,
    
    "news_count_62h": True,
    "news_sentiment_mean_62h": True,
    "news_sentiment_max_62h": True,
    "news_sentiment_min_62h": True,
    "news_pos_ratio_62h": True,
    "news_neg_ratio_62h": True,
    
    # === اخبار پیشرفته (FinBERT) ===
    "strong_positive_news": True,
    "strong_negative_news": True,
    "pos_to_neg_ratio": True,
    "sentiment_trend": True,
    "sentiment_diversity": True,
    
    # === روند و مومنتوم پیشرفته ===
    "price_momentum_3": True,
    "price_momentum_7": True,
    "price_momentum_14": True,
    "price_momentum_30": True,
    
    "price_to_ma_3": True,
    "price_to_ma_7": True,
    "price_to_ma_14": True,
    "price_to_ma_30": True,
    
    "std_3": True,
    "std_7": True,
    "std_14": True,
    "std_30": True,
    
    "max_price_3": True,
    "max_price_7": True,
    "max_price_14": True,
    "max_price_30": True,
    
    "min_price_3": True,
    "min_price_7": True,
    "min_price_14": True,
    "min_price_30": True,
    
    "vol_change_3": True,
    "vol_change_7": True,
    "vol_change_14": True,
    "vol_change_30": True,
    
    # === نوسان پیشرفته ===
    "flag_pole_index": True,
    "hurst_exponent": True,
    "rsi_macd_strength": True,
    
    # === الگوهای قیمت ===
    "double_bottom": True,
    "double_top": True,
    
    # === فشار بازار ===
    "buying_pressure": True,
    "selling_pressure": True,
    "strength_index": True,
    
    # === آماری پیشرفته ===
    "kurtosis": True,
    "skewness": True,
    "max_drawdown": True,
    
    # === ترکیبی متا ===
    "trend_meta_signal": True,
    
    # === الگوهای هارمونیک ===
    "fib_382_proximity": True,
    "fib_500_proximity": True,
    "fib_618_proximity": True,
    "gartley_pattern": True,
    "butterfly_pattern": True,
    
    # === حافظه بلندمدت ===
    "position_in_range_30": True,
    "position_in_range_60": True,
    "position_in_range_90": True,
    "position_in_range_120": True,
    
    "distance_to_high_30": True,
    "distance_to_high_60": True,
    "distance_to_high_90": True,
    "distance_to_high_120": True,
    
    "distance_to_low_30": True,
    "distance_to_low_60": True,
    "distance_to_low_90": True,
    "distance_to_low_120": True,
    
    # === آشوب و پیچیدگی ===
    "direction_change_count": True,
    "signal_to_noise": True,
}
