"""
تعریف گروه‌های فیچر برای مدل‌های متخصص
"""

# تعریف گروه‌های فیچر برای مدل‌های متخصص
FEATURE_GROUPS = {
    # گروه میانگین‌های متحرک و روندها
    "moving_averages": [
        "ema5", "ema9", "ema10", "ema20", "ema21", "ema50", "ema100", "ema200", 
        "sma20", "sma50", "tema20",
        "price_ema50_diff", "price_ema200_diff", "ema_cross_9_21",
        "donchian_high", "donchian_low"
    ],
    
    # گروه اسیلاتورها
    "oscillators": [
        "rsi14", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist",
        "cci", "willr", "roc", "momentum5", "momentum10", "adx14", "supertrend"
    ],
    
    # گروه نوسان و حجم
    "volatility": [
        "atr14", "bb_upper", "bb_lower", "bb_width", "volatility",
        "volume", "volume_mean", "volume_spike", "vol_spike", "atr_spike",
        "obv", "vwap", "williams_vix_fix"
    ],
    
    # گروه الگوهای شمعی و پرایس اکشن
    "candlesticks": [
        "doji", "engulfing", "hammer", "morning_star", "evening_star", "shooting_star",
        "candle_range", "shadow_ratio", "wick_ratio", "candle_change",
        "green_candles_10", "red_candles_10", "green_candle_ratio_20", "red_candle_ratio_20",
        "price_gap", "breakout_30", "breakdown_30"
    ],
    
    # گروه فاندامنتال و اخبار
    "news": [
        "news_count", "news_sentiment_mean", "news_sentiment_std", 
        "news_pos_count", "news_neg_count", "news_latest_sentiment", 
        "news_content_len", "news_weighted_score",
        "news_count_1h", "news_sentiment_mean_1h", "news_sentiment_max_1h", 
        "news_sentiment_min_1h", "news_pos_ratio_1h", "news_neg_ratio_1h",
        "news_count_6h", "news_sentiment_mean_6h", "news_sentiment_max_6h", 
        "news_sentiment_min_6h", "news_pos_ratio_6h", "news_neg_ratio_6h",
        "news_count_12h", "news_sentiment_mean_12h", "news_sentiment_max_12h", 
        "news_sentiment_min_12h", "news_pos_ratio_12h", "news_neg_ratio_12h",
        "news_count_24h", "news_sentiment_mean_24h", "news_sentiment_max_24h", 
        "news_sentiment_min_24h", "news_pos_ratio_24h", "news_neg_ratio_24h"
    ],
    
    # گروه قیمت
    "price": [
        "open", "high", "low", "close", "mean_reversion_zscore"
    ]
}

# لیست تمام فیچرهای شناخته شده
ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group)

# فیچرهای ضروری که همیشه باید شامل شوند
ESSENTIAL_FEATURES = ["close", "open", "high", "low", "volume", "ema20", "ema50", "rsi14", "macd"]

def get_group_for_feature(feature_name):
    """تعیین گروه یک فیچر"""
    for group_name, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group_name
    return "unknown"

def get_all_features_in_groups(group_names):
    """دریافت تمام فیچرهای موجود در یک سری گروه"""
    all_features = []
    for group in group_names:
        if group in FEATURE_GROUPS:
            all_features.extend(FEATURE_GROUPS[group])
    return all_features