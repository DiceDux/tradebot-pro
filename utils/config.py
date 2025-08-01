SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"]  # قابل تغییر و توسعه
PREDICTION_INTERVAL = 300  # هر 5 دقیقه (به ثانیه)
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tradebot-pro",
    "port": 3306
}
CANDLE_LIMIT = 100  # تعداد کندل برای هر تحلیل
NEWS_HOURS = 168   # تعداد ساعت عقب‌گرد برای دریافت اخبار (۱ هفته)

# پیکربندی سیستم معاملاتی
TRADING_CONFIG = {
    "symbols": SYMBOLS,  # نمادهای قابل معامله
    "check_interval": 5,  # بررسی هر 5 دقیقه
    "candle_limit": CANDLE_LIMIT,
    "news_hours": NEWS_HOURS,
    "confidence_threshold": 70.0,  # آستانه اطمینان برای انجام معامله (درصد)
    "max_trades_per_day": 5,  # حداکثر تعداد معاملات روزانه
    "take_profit": 1.5,  # درصد سود مطلوب
    "stop_loss": 1.0,  # درصد ضرر قابل قبول
    "risk_per_trade": 2.0,  # درصد سرمایه در معرض ریسک در هر معامله
    "allow_buy": True,  # اجازه خرید
    "allow_sell": True,  # اجازه فروش
    "trade_sizing": "dynamic",  # سایز معاملات (fixed یا dynamic)
    "market_hours": "24/7"  # ساعات معامله (برای ارزهای دیجیتال 24/7 است)
}
