import time
from utils.config import SYMBOLS, PREDICTION_INTERVAL
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals
from data.db_manager import save_analysis

def main():
    print("TradeBot Pro is running...")
    model = load_or_train_model()  # مدل را آموزش بده یا لود کن
    while True:
        for symbol in SYMBOLS:
            # 1. دریافت کندل و خبر
            candles = get_latest_candles(symbol)
            news = get_latest_news(symbol)
            # 2. ساخت فیچرهای کامل (تکنیکال + فاندامنتال)
            features = build_features(candles, news, symbol)
            # 3. پیش‌بینی سیگنال و تحلیل
            signal, analysis = predict_signals(model, features)
            # 4. ذخیره تحلیل (برای پنل UI)
            save_analysis(symbol, analysis)
            print(f"{symbol}: {signal} | تحلیل ذخیره شد")
        # اجرای تحلیل بعد از هر ۵ دقیقه
        time.sleep(PREDICTION_INTERVAL)

if __name__ == "__main__":
    main()