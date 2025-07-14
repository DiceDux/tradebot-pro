import time
import pandas as pd
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals
from data.candle_manager import get_latest_candles
from utils.price_fetcher import get_realtime_price

LIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BALANCE = 100
TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.7

def live_test():
    model, feature_names = load_or_train_model()
    balance = BALANCE
    positions = {}
    while True:
        for symbol in LIVE_SYMBOLS:
            candles = get_latest_candles(symbol, interval="4h", limit=200)
            price_now = get_realtime_price(symbol)
            # ساخت کندل جدید ۴ساعته با قیمت فعلی
            new_candle = candles.iloc[-1].copy()
            new_candle["close"] = price_now
            new_candle["timestamp"] = int(time.time())
            candles = pd.concat([candles, pd.DataFrame([new_candle])], ignore_index=True)
            features = build_features(candles, None, symbol)
            features_df = pd.DataFrame([features])
            signal, analysis = predict_signals(model, feature_names, features_df)
            # (منطق مدیریت پوزیشن و سود پله‌ای مشابه بک‌تست را همینجا اضافه کن)
            print(f"{symbol} | signal={signal} | price={price_now} | balance={balance}")
        time.sleep(60)  # هر دقیقه یا هر چند ثانیه که خواستی

if __name__ == "__main__":
    live_test()
