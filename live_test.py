import time
import pandas as pd
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from utils.price_fetcher import get_realtime_price
from feature_engineering.feature_monitor import FeatureMonitor

LIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BALANCE = 100
TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.7
NEWS_HOURS = 48  # یا بیشتر، بستگی به فیچرها

def run_feature_monitor(model, all_feature_names, symbol):
    # آماده‌سازی دیتافریم فیچر برای بازارسنجی
    candles = get_latest_candles(symbol, interval="4h", limit=120)
    news = get_latest_news(symbol, hours=NEWS_HOURS)
    features_list = []
    for i in range(len(candles) - 100, len(candles)):
        candle_slice = candles.iloc[:i + 1]
        if not news.empty:
            candle_time = pd.to_datetime(candle_slice.iloc[-1]['timestamp'], unit='s')
            news_slice = news[news['published_at'] <= candle_time]
        else:
            news_slice = pd.DataFrame()
        features = build_features(candle_slice, news_slice, symbol)
        features_list.append(features)
    X = pd.DataFrame(features_list)
    monitor = FeatureMonitor(model, all_feature_names)
    monitor.evaluate_features(X, y=None)
    return monitor.get_active_feature_names()

def live_test():
    model, all_feature_names = load_or_train_model()
    # برای هر نماد بازارسنجی کن و فیچرهای فعال را جدا نگه دار
    symbol_features = {}
    for symbol in LIVE_SYMBOLS:
        print(f"Running feature monitor for {symbol} ...")
        feature_names = run_feature_monitor(model, all_feature_names, symbol)
        symbol_features[symbol] = feature_names

    balance = {symbol: BALANCE for symbol in LIVE_SYMBOLS}
    positions = {symbol: None for symbol in LIVE_SYMBOLS}
    entry_price = {symbol: 0 for symbol in LIVE_SYMBOLS}
    sl_price = {symbol: 0 for symbol in LIVE_SYMBOLS}
    tp_prices = {symbol: [] for symbol in LIVE_SYMBOLS}
    qty_left = {symbol: 1.0 for symbol in LIVE_SYMBOLS}
    tp_idx = {symbol: 0 for symbol in LIVE_SYMBOLS}
    trade_balance = {symbol: BALANCE for symbol in LIVE_SYMBOLS}

    print("===== Starting LIVE trading/test =====")
    while True:
        for symbol in LIVE_SYMBOLS:
            try:
                # گرفتن کندل و قیمت زنده
                candles = get_latest_candles(symbol, interval="4h", limit=120)
                price_now = get_realtime_price(symbol)
                news = get_latest_news(symbol, hours=NEWS_HOURS)
                # ساخت کندل جدید با قیمت زنده
                new_candle = candles.iloc[-1].copy()
                new_candle["close"] = price_now
                new_candle["timestamp"] = int(time.time())
                candles = pd.concat([candles, pd.DataFrame([new_candle])], ignore_index=True)
                # ساخت فیچر با اخبار و کندل
                candle_slice = candles.iloc[-100:]  # آخرین 100 کندل
                if not news.empty:
                    candle_time = pd.to_datetime(new_candle['timestamp'], unit='s')
                    news_slice = news[news['published_at'] <= candle_time]
                else:
                    news_slice = pd.DataFrame()
                features = build_features(candle_slice, news_slice, symbol)
                features_df = pd.DataFrame([features]).reindex(columns=symbol_features[symbol], fill_value=0)
                # پیش‌بینی مدل
                signal, analysis = predict_signals(model, symbol_features[symbol], features_df)
                confidence = analysis.get("confidence", 0.0)
                if confidence < THRESHOLD:
                    signal = "Hold"

                # مدیریت پوزیشن و TP/SL (شبیه بک‌تست)
                pos = positions[symbol]
                if pos is None:
                    if signal == "Buy":
                        positions[symbol] = "long"
                        entry_price[symbol] = price_now
                        sl_price[symbol] = price_now * (1 - SL_PCT)
                        tp_prices[symbol] = [price_now * (1 + tp) for tp in TP_STEPS]
                        qty_left[symbol] = 1.0
                        tp_idx[symbol] = 0
                        trade_balance[symbol] = balance[symbol] * 1.0  # 100% بالانس هر نماد وارد معامله
                        print(f"[{symbol}] LONG ENTRY | price={price_now:.2f} | conf={confidence:.2f} | used_balance={trade_balance[symbol]:.2f}")
                else:
                    trailing_sl = entry_price[symbol] * (1 - SL_PCT)
                    if tp_idx[symbol] > 0:
                        trailing_sl = tp_prices[symbol][tp_idx[symbol] - 1]
                    # حد ضرر
                    if price_now <= trailing_sl:
                        loss = (price_now - entry_price[symbol]) * qty_left[symbol] * trade_balance[symbol] / entry_price[symbol]
                        balance[symbol] += loss
                        print(f"[{symbol}] STOP LOSS | price={price_now:.2f} | P/L={loss:.2f} | bal={balance[symbol]:.2f}")
                        positions[symbol] = None
                        qty_left[symbol] = 1.0
                    # حد سود پله‌ای
                    elif tp_idx[symbol] < len(tp_prices[symbol]) and price_now >= tp_prices[symbol][tp_idx[symbol]]:
                        sell_qty = TP_QTYS[tp_idx[symbol]]
                        profit = (tp_prices[symbol][tp_idx[symbol]] - entry_price[symbol]) * sell_qty * trade_balance[symbol] / entry_price[symbol]
                        balance[symbol] += profit
                        qty_left[symbol] -= sell_qty
                        print(f"[{symbol}] TAKE PROFIT {tp_idx[symbol]+1} | price={tp_prices[symbol][tp_idx[symbol]]:.2f} | qty={sell_qty:.2f} | P/L={profit:.2f} | bal={balance[symbol]:.2f}")
                        tp_idx[symbol] += 1
                        if qty_left[symbol] <= 0.001 or tp_idx[symbol] == len(tp_prices[symbol]):
                            positions[symbol] = None
                            qty_left[symbol] = 1.0
                print(f"[{symbol}] signal={signal} | conf={confidence:.2f} | price={price_now:.2f} | bal={balance[symbol]:.2f}")
            except Exception as e:
                print(f"[{symbol}] ERROR: {e}")
        time.sleep(60)

if __name__ == "__main__":
    live_test()
