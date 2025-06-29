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