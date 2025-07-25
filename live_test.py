import time
import pandas as pd
import threading
import tkinter as tk
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, retrain_active_model, load_dynamic_model, predict_signals
from data.candle_manager import get_latest_candles, keep_last_200_candles
from data.news_manager import get_latest_news, get_news_for_range
from data.fetch_online import fetch_candles_binance, save_candles_to_db, fetch_news_newsapi, save_news_to_db
from feature_engineering.sentiment_finbert import analyze_sentiment_finbert
from utils.price_fetcher import get_realtime_price
from feature_engineering.feature_monitor import FeatureMonitor
from feature_engineering.feature_config import FEATURE_CONFIG
import os
import importlib
import numpy as np

LIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BALANCE = 100
TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.7
CANDLE_LIMIT = 200

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "ede1e0b0db7140fdbbd20f6f1b440cb9")

trades_log = []
status_texts = {symbol: "" for symbol in LIVE_SYMBOLS}
latest_prices = {symbol: 0.0 for symbol in LIVE_SYMBOLS}
positions = {symbol: None for symbol in LIVE_SYMBOLS}
entry_price = {symbol: 0 for symbol in LIVE_SYMBOLS}
sl_price = {symbol: 0 for symbol in LIVE_SYMBOLS}
tp_prices = {symbol: [] for symbol in LIVE_SYMBOLS}
qty_left = {symbol: 1.0 for symbol in LIVE_SYMBOLS}
tp_idx = {symbol: 0 for symbol in LIVE_SYMBOLS}
balance = {symbol: BALANCE for symbol in LIVE_SYMBOLS}
active_feature_names_per_symbol = {}

def activate_all_features():
    for k in FEATURE_CONFIG.keys():
        FEATURE_CONFIG[k] = True

def auto_feature_selection(symbol, model, all_feature_names):
    candles = get_latest_candles(symbol, CANDLE_LIMIT)
    news = get_latest_news(symbol, hours=CANDLE_LIMIT*4)
    features_list = []
    for i in range(len(candles) - 100, len(candles)):
        candle_slice = candles.iloc[:i + 1]
        news_slice = news[news['published_at'] <= pd.to_datetime(candle_slice.iloc[-1]['timestamp'], unit='s')] if not news.empty else pd.DataFrame()
        feat = build_features(candle_slice, news_slice, symbol)
        if isinstance(feat, pd.DataFrame):
            feat = feat.iloc[0].to_dict()
        elif isinstance(feat, pd.Series):
            feat = feat.to_dict()
        features_list.append(feat)
    X = pd.DataFrame(features_list)
    monitor = FeatureMonitor(model, all_feature_names)
    monitor.evaluate_features(X)
    importlib.reload(importlib.import_module("feature_engineering.feature_config"))
    from feature_engineering.feature_config import FEATURE_CONFIG
    return [f for f, v in FEATURE_CONFIG.items() if v]

def fetch_and_store_latest_data(symbol):
    candles = fetch_candles_binance(symbol, interval="4h", limit=200)
    save_candles_to_db(candles)
    keep_last_200_candles(symbol)
    print(f"[{symbol}] Saved {len(candles)} candles to DB.")
    news = fetch_news_newsapi(symbol, limit=25, api_key=NEWSAPI_KEY)
    for n in news:
        n["sentiment_score"] = analyze_sentiment_finbert((n.get("title") or "") + " " + (n.get("content") or ""))
    save_news_to_db(news)
    print(f"[{symbol}] Saved {len(news)} news to DB.")

def save_trades_log():
    if trades_log:
        pd.DataFrame(trades_log).to_csv("trades.csv", index=False)

def price_updater():
    global status_texts, latest_prices
    while True:
        for symbol in LIVE_SYMBOLS:
            try:
                price_now = get_realtime_price(symbol)
                latest_prices[symbol] = price_now
                lines = [
                    f"Symbol: {symbol}",
                    f"Price: {price_now:.2f}",
                ]
                if status_texts[symbol]:
                    old_lines = status_texts[symbol].split('\n')
                    for old_line in old_lines:
                        if (
                            old_line.startswith("Signal:") or
                            old_line.startswith("Balance:") or
                            old_line.startswith("Entry:") or
                            old_line.startswith("SL:") or
                            old_line.startswith("TP") or
                            old_line.startswith("TP idx") or
                            old_line.startswith("QTY left") or
                            old_line.startswith("Position") or
                            old_line.startswith("No active position")
                        ):
                            lines.append(old_line)
                status_texts[symbol] = '\n'.join(lines)
            except Exception as e:
                status_texts[symbol] = f"Symbol: {symbol}\nPrice: --\n(Price error: {e})"
        time.sleep(1)

def tp_sl_watcher():
    global status_texts, latest_prices, positions, sl_price, tp_prices, tp_idx, qty_left, entry_price, balance, trades_log
    while True:
        for symbol in LIVE_SYMBOLS:
            price_now = latest_prices.get(symbol, 0.0)
            try:
                if positions[symbol] == "LONG":
                    if price_now <= sl_price[symbol]:
                        pnl = -BALANCE * qty_left[symbol] * SL_PCT
                        fee = abs(pnl) * 0.001
                        balance[symbol] += pnl - fee
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'SL',
                            'side': 'LONG',
                            'entry_price': entry_price[symbol],
                            'exit_price': price_now,
                            'qty': qty_left[symbol],
                            'pnl': pnl,
                            'fee': fee,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                        positions[symbol] = None
                        qty_left[symbol] = 1.0
                    elif tp_idx[symbol] < len(tp_prices[symbol]) and price_now >= tp_prices[symbol][tp_idx[symbol]]:
                        tp_qty = TP_QTYS[tp_idx[symbol]]
                        pnl = BALANCE * tp_qty * (TP_STEPS[tp_idx[symbol]])
                        fee = abs(pnl) * 0.001
                        balance[symbol] += pnl - fee
                        trades_log.append({
                            'symbol': symbol,
                            'type': f'TP{tp_idx[symbol]+1}',
                            'side': 'LONG',
                            'entry_price': entry_price[symbol],
                            'exit_price': price_now,
                            'qty': tp_qty,
                            'pnl': pnl,
                            'fee': fee,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                        qty_left[symbol] -= tp_qty
                        tp_idx[symbol] += 1
                        if qty_left[symbol] <= 0:
                            positions[symbol] = None
                            qty_left[symbol] = 1.0
                elif positions[symbol] == "SHORT":
                    if price_now >= sl_price[symbol]:
                        pnl = -BALANCE * qty_left[symbol] * SL_PCT
                        fee = abs(pnl) * 0.001
                        balance[symbol] += pnl - fee
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'SL',
                            'side': 'SHORT',
                            'entry_price': entry_price[symbol],
                            'exit_price': price_now,
                            'qty': qty_left[symbol],
                            'pnl': pnl,
                            'fee': fee,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                        positions[symbol] = None
                        qty_left[symbol] = 1.0
                    elif tp_idx[symbol] < len(tp_prices[symbol]) and price_now <= tp_prices[symbol][tp_idx[symbol]]:
                        tp_qty = TP_QTYS[tp_idx[symbol]]
                        pnl = BALANCE * tp_qty * (TP_STEPS[tp_idx[symbol]])
                        fee = abs(pnl) * 0.001
                        balance[symbol] += pnl - fee
                        trades_log.append({
                            'symbol': symbol,
                            'type': f'TP{tp_idx[symbol]+1}',
                            'side': 'SHORT',
                            'entry_price': entry_price[symbol],
                            'exit_price': price_now,
                            'qty': tp_qty,
                            'pnl': pnl,
                            'fee': fee,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                        qty_left[symbol] -= tp_qty
                        tp_idx[symbol] += 1
                        if qty_left[symbol] <= 0:
                            positions[symbol] = None
                            qty_left[symbol] = 1.0
            except Exception as e:
                print(f"[{symbol}] TP/SL Watcher ERROR: {e}")
        time.sleep(1)

def live_test():
    global status_texts, positions, entry_price, sl_price, tp_prices, qty_left, tp_idx, balance, trades_log, active_feature_names_per_symbol

    model, all_feature_names = load_or_train_model()
    activate_all_features()
    model_per_symbol = {}

    # دریافت و ذخیره دیتا پیش از تحلیل
    for symbol in LIVE_SYMBOLS:
        fetch_and_store_latest_data(symbol)

    # اجرای فیچر مانیتور و فعال‌سازی فیچرهای مناسب و ریترین مدل
    for symbol in LIVE_SYMBOLS:
        print(f"Selecting best features for {symbol} ...")
        active_features = auto_feature_selection(symbol, model, all_feature_names)
        print(f"Active features for {symbol}: {active_features}")

        candles_train = get_latest_candles(symbol, limit=None)
        news_train = get_latest_news(symbol, hours=None)
        if not news_train.empty:
            news_train['published_at'] = pd.to_datetime(news_train['published_at'])
        X_full = []
        y_full = []
        for i in range(100, len(candles_train)-1):
            candle_slice = candles_train.iloc[i-99:i+1]
            candle_time = pd.to_datetime(candles_train.iloc[i]['timestamp'], unit='s')
            news_slice = news_train[news_train['published_at'] <= candle_time] if not news_train.empty else pd.DataFrame()
            features = build_features(candle_slice, news_slice, symbol)
            if isinstance(features, pd.DataFrame):
                features = features.iloc[0].to_dict()
            elif isinstance(features, pd.Series):
                features = features.to_dict()
            X_full.append({f: features.get(f, 0.0) for f in active_features})
            y_full.append(candles_train.iloc[i].get("label", 1))
        X_full_df = pd.DataFrame(X_full)
        y_full_arr = np.array(y_full)
        model_active = retrain_active_model(X_full_df, y_full_arr, active_features)
        # بعد از ریترین، حتماً مدل فعال و فیچرهای داینامیک را از فایل لود کن
        model_active, dynamic_features = load_dynamic_model()
        if model_active is None or dynamic_features is None:
            print(f"[{symbol}] ERROR: Active model or features not found after retrain!")
            continue
        model_per_symbol[symbol] = model_active
        active_feature_names_per_symbol[symbol] = dynamic_features

    print("===== Starting LIVE trading/test =====")
    last_main_loop = 0

    while True:
        now = time.time()
        if now - last_main_loop < 60:
            time.sleep(1)
            continue
        last_main_loop = now

        for symbol in LIVE_SYMBOLS:
            try:
                fetch_and_store_latest_data(symbol)
                candles = get_latest_candles(symbol, CANDLE_LIMIT)
                price_now = latest_prices.get(symbol, 0.0)
                news = get_latest_news(symbol, hours=CANDLE_LIMIT*4)
                if not candles.empty:
                    start_ts = candles.iloc[-CANDLE_LIMIT]['timestamp']
                    end_ts = candles.iloc[-1]['timestamp']
                    news = get_news_for_range(symbol, start_ts, end_ts)
                    news = pd.DataFrame(news)
                new_candle = candles.iloc[-1].copy()
                new_candle['close'] = price_now
                candle_slice = candles.copy()
                candle_slice.iloc[-1] = new_candle

                features = build_features(candle_slice, news, symbol)
                if isinstance(features, pd.DataFrame):
                    features = features.iloc[0].to_dict()
                elif isinstance(features, pd.Series):
                    features = features.to_dict()
                active_features = active_feature_names_per_symbol.get(symbol, [])
                if not active_features:
                    print(f"[{symbol}] ERROR: No active features found for prediction!")
                    continue
                row = {f: features.get(f, 0.0) for f in active_features}
                X = pd.DataFrame([row])
                model_active = model_per_symbol.get(symbol)
                if model_active is None:
                    print(f"[{symbol}] ERROR: No active model for prediction!")
                    continue
                signal, analysis = predict_signals(model_active, active_features, X)
                confidence = analysis.get("confidence", 0.0)

                status_lines = [
                    f"Symbol: {symbol}",
                    f"Price: {price_now:.2f}",
                    f"Signal: {signal} | Confidence: {confidence:.2f}",
                    f"Balance: {balance[symbol]:.2f}"
                ]
                if positions[symbol] is not None:
                    status_lines.append(f"Entry: {entry_price[symbol]:.2f}")
                    status_lines.append(f"SL: {sl_price[symbol]:.2f}")
                    tps = [f"TP{idx+1}: {tp:.2f} | QTY: {TP_QTYS[idx]}" for idx, tp in enumerate(tp_prices[symbol])]
                    status_lines.extend(tps)
                    status_lines.append(f"TP idx (پله فعال): {tp_idx[symbol]}")
                    status_lines.append(f"QTY left: {qty_left[symbol]:.2f}")
                    status_lines.append(f"Position: {positions[symbol]}")
                else:
                    status_lines.append("No active position")
                recent_trades = [trade for trade in trades_log if trade['symbol'] == symbol][-5:]
                if recent_trades:
                    status_lines.append("Last Trades:")
                    for t in recent_trades:
                        entry_val = t.get('entry_price', t.get('price', 0))
                        exit_val = t.get('exit_price', '-')
                        qty_val = t.get('qty', '-')
                        status_lines.append(
                            f" {t['type']} | {t.get('side','')} | Entry: {entry_val:.2f} | Exit: {exit_val if isinstance(exit_val, float) else exit_val} | QTY: {qty_val}"
                        )

                status_texts[symbol] = '\n'.join(status_lines)

                # فقط اگر اعتماد سیگنال بالای آستانه بود معامله انجام بده
                if positions[symbol] is None and confidence >= THRESHOLD:
                    if signal == "Buy":
                        positions[symbol] = "LONG"
                        entry_price[symbol] = price_now
                        sl_price[symbol] = price_now * (1 - SL_PCT)
                        tp_prices[symbol] = [price_now * (1 + x) for x in TP_STEPS]
                        qty_left[symbol] = 1.0
                        tp_idx[symbol] = 0
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'ENTRY',
                            'side': 'LONG',
                            'price': price_now,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                    elif signal == "Sell":
                        positions[symbol] = "SHORT"
                        entry_price[symbol] = price_now
                        sl_price[symbol] = price_now * (1 + SL_PCT)
                        tp_prices[symbol] = [price_now * (1 - x) for x in TP_STEPS]
                        qty_left[symbol] = 1.0
                        tp_idx[symbol] = 0
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'ENTRY',
                            'side': 'SHORT',
                            'price': price_now,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
            except Exception as e:
                print(f"[{symbol}] ERROR: {e}")

def update_gui(label):
    def gui_loop():
        all_status = '\n\n'.join([status_texts[symbol] for symbol in LIVE_SYMBOLS])
        label.config(text=all_status)
        label.after(1000, gui_loop)
    return gui_loop

def main():
    root = tk.Tk()
    root.title("Live TP/SL & Price")
    label = tk.Label(root, text="Waiting for updates...", font=("Arial", 13), justify="left")
    label.pack(padx=20, pady=20)
    threading.Thread(target=price_updater, daemon=True).start()
    threading.Thread(target=tp_sl_watcher, daemon=True).start()
    threading.Thread(target=live_test, daemon=True).start()
    update_func = update_gui(label)
    label.after(1000, update_func)
    root.mainloop()

if __name__ == "__main__":
    main()
