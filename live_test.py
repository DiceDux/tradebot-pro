import time
import pandas as pd
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals
from data.candle_manager import get_latest_candles
from utils.price_fetcher import get_realtime_price
from feature_engineering.feature_monitor import FeatureMonitor  # اضافه کن

LIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BALANCE = 100
TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.7

def live_test():
    model, all_feature_names = load_or_train_model()
    # مرحله بازارسنجی و انتخاب فیچرها
    for symbol in LIVE_SYMBOLS:
        candles = get_latest_candles(symbol, interval="4h", limit=200)
        features_list = []
        for i in range(len(candles)-100, len(candles)):
            features_list.append(build_features(candles.iloc[:i+1], None, symbol))
        X = pd.DataFrame(features_list)
        monitor = FeatureMonitor(model, all_feature_names)
        monitor.evaluate_features(X, y=None)
    # استخراج فقط فیچرهای فعال
    feature_names = monitor.get_active_feature_names()
    balance = BALANCE
    positions = {}
    while True:
        for symbol in LIVE_SYMBOLS:
            candles = get_latest_candles(symbol, interval="4h", limit=200)
            price_now = get_realtime_price(symbol)
            new_candle = candles.iloc[-1].copy()
            new_candle["close"] = price_now
            new_candle["timestamp"] = int(time.time())
            candles = pd.concat([candles, pd.DataFrame([new_candle])], ignore_index=True)
            features = build_features(candles, None, symbol)
            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(columns=feature_names, fill_value=0)
            signal, analysis = predict_signals(model, feature_names, features_df)
            confidence = analysis.get("confidence", 0.0)
            if confidence < THRESHOLD:
                signal = "Hold"
            # (منطق مدیریت پوزیشن و سود پله‌ای مشابه بک‌تست را همینجا اضافه کن)
            print(f"{symbol} | signal={signal} | price={price_now} | balance={balance} | conf={confidence:.2f}")
        time.sleep(60)

if __name__ == "__main__":
    live_test()
