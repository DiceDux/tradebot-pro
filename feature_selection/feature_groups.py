"""
تعریف گروه‌های فیچر برای مدل‌های متخصص
"""

# تعریف گروه‌های فیچر برای مدل‌های متخصص
FEATURE_GROUPS = {
    # گروه میانگین‌های متحرک و روندها
    "moving_averages": [
        # میانگین‌های متحرک پایه
        "ema5", "ema9", "ema10", "ema20", "ema21", "ema50", "ema100", "ema200", 
        "sma20", "sma50", "tema20",
        # میانگین‌های متحرک نسبی و کراس‌ها
        "price_ema50_diff", "price_ema200_diff", "ema_cross_9_21",
        "donchian_high", "donchian_low",
        # نسبت‌های میانگین متحرک
        "price_to_ma_3", "price_to_ma_7", "price_to_ma_14", "price_to_ma_30",
        # سیگنال‌های روند
        "trend_meta_signal", "ma_ratio_short", "ma_ratio_long"
    ],
    
    # گروه اسیلاتورها
    "oscillators": [
        # اسیلاتورهای پایه
        "rsi14", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist",
        "cci", "willr", "roc", "momentum5", "momentum10", "adx14", "supertrend",
        # شاخص‌های مومنتوم
        "price_momentum_3", "price_momentum_7", "price_momentum_14", "price_momentum_30",
        # شاخص‌های واگرایی و قدرت
        "rsi_macd_strength", "price_rsi_divergence", "indicator_consensus"
    ],
    
    # گروه نوسان و حجم
    "volatility": [
        # نوسان پایه
        "atr14", "bb_upper", "bb_lower", "bb_width", "volatility",
        # حجم پایه
        "volume", "volume_mean", "volume_spike", "vol_spike", "atr_spike",
        "obv", "vwap", "williams_vix_fix",
        # تغییرات حجم
        "vol_change_3", "vol_change_7", "vol_change_14", "vol_change_30",
        # نوسان پیشرفته
        "flag_pole_index", "hurst_exponent", "std_3", "std_7", "std_14", "std_30",
        # فشار خرید و فروش
        "buying_pressure", "selling_pressure", "strength_index"
    ],
    
    # گروه الگوهای شمعی و پرایس اکشن
    "candlesticks": [
        # الگوهای شمعی
        "doji", "engulfing", "hammer", "morning_star", "evening_star", "shooting_star",
        # ویژگی‌های شمعی
        "candle_range", "shadow_ratio", "wick_ratio", "candle_change",
        "green_candles_10", "red_candles_10", "green_candle_ratio_20", "red_candle_ratio_20",
        # شکست‌ها و شکاف‌ها
        "price_gap", "breakout_30", "breakdown_30",
        # الگوهای پیشرفته
        "double_bottom", "double_top",
        # شاخص‌های آماری قیمت
        "kurtosis", "skewness", "max_drawdown", "signal_to_noise"
    ],
    
    # گروه فاندامنتال و اخبار
    "news": [
        # اخبار پایه
        "news_count", "news_sentiment_mean", "news_sentiment_std", 
        "news_pos_count", "news_neg_count", "news_latest_sentiment", 
        "news_content_len", "news_weighted_score",
        
        # اخبار بازه 1 ساعته
        "news_count_1h", "news_sentiment_mean_1h", "news_sentiment_max_1h", 
        "news_sentiment_min_1h", "news_pos_ratio_1h", "news_neg_ratio_1h",
        
        # اخبار بازه 6 ساعته
        "news_count_6h", "news_sentiment_mean_6h", "news_sentiment_max_6h", 
        "news_sentiment_min_6h", "news_pos_ratio_6h", "news_neg_ratio_6h",
        
        # اخبار بازه 12 ساعته
        "news_count_12h", "news_sentiment_mean_12h", "news_sentiment_max_12h", 
        "news_sentiment_min_12h", "news_pos_ratio_12h", "news_neg_ratio_12h",
        
        # اخبار بازه 24 ساعته
        "news_count_24h", "news_sentiment_mean_24h", "news_sentiment_max_24h", 
        "news_sentiment_min_24h", "news_pos_ratio_24h", "news_neg_ratio_24h",
        
        # اخبار بازه 36 ساعته
        "news_count_36h", "news_sentiment_mean_36h", "news_sentiment_max_36h", 
        "news_sentiment_min_36h", "news_pos_ratio_36h", "news_neg_ratio_36h",
        
        # اخبار بازه 48 ساعته
        "news_count_48h", "news_sentiment_mean_48h", "news_sentiment_max_48h", 
        "news_sentiment_min_48h", "news_pos_ratio_48h", "news_neg_ratio_48h",
        
        # اخبار بازه 50 ساعته
        "news_count_50h", "news_sentiment_mean_50h", "news_sentiment_max_50h", 
        "news_sentiment_min_50h", "news_pos_ratio_50h", "news_neg_ratio_50h",
        
        # اخبار بازه 62 ساعته
        "news_count_62h", "news_sentiment_mean_62h", "news_sentiment_max_62h", 
        "news_sentiment_min_62h", "news_pos_ratio_62h", "news_neg_ratio_62h",
        
        # اخبار پیشرفته (FinBERT)
        "strong_positive_news", "strong_negative_news", "pos_to_neg_ratio",
        "sentiment_trend", "sentiment_diversity"
    ],
    
    # گروه قیمت و دامنه
    "price": [
        # قیمت‌های پایه
        "open", "high", "low", "close", "mean_reversion_zscore",
        
        # حد سود و ضرر فیبوناچی
        "fib_382_proximity", "fib_500_proximity", "fib_618_proximity",
        "gartley_pattern", "butterfly_pattern",
        
        # فیچرهای محدوده قیمت
        "position_in_range_30", "position_in_range_60", "position_in_range_90", "position_in_range_120",
        "distance_to_high_30", "distance_to_high_60", "distance_to_high_90", "distance_to_high_120",
        "distance_to_low_30", "distance_to_low_60", "distance_to_low_90", "distance_to_low_120",
        
        # قیمت‌های بازه‌های زمانی
        "max_price_3", "max_price_7", "max_price_14", "max_price_30",
        "min_price_3", "min_price_7", "min_price_14", "min_price_30"
    ],
    
    # گروه جدید: الگوهای پیشرفته بازار
    "advanced_patterns": [
        # الگوهای هارمونیک
        "fib_382_proximity", "fib_500_proximity", "fib_618_proximity",
        "gartley_pattern", "butterfly_pattern",
        
        # الگوهای کلاسیک
        "double_bottom", "double_top",
        
        # شاخص‌های پیچیدگی
        "direction_change_count", "signal_to_noise",
        "hurst_exponent", "flag_pole_index",
        
        # شاخص‌های فشار
        "buying_pressure", "selling_pressure", "strength_index"
    ]
}

# لیست تمام فیچرهای شناخته شده
ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group)
ALL_FEATURES = list(set(ALL_FEATURES))  # حذف موارد تکراری

# فیچرهای ضروری که همیشه باید شامل شوند
ESSENTIAL_FEATURES = [
    "close", "open", "high", "low", "volume", 
    "ema20", "ema50", "ema200", "rsi14", "macd", 
    "bb_width", "news_sentiment_mean", "adx14"
]

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
    return list(set(all_features))  # حذف موارد تکراری
