import time
import pandas as pd
import threading
import tkinter as tk
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
NEWS_HOURS = 48
CANDLE_LIMIT = 120

# --- پنجره ساده برای نمایش قیمت و وضعیت معامله ---
class TradeStatusWindow(threading.Thread):
    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.root.title("Live TP/SL & Price")
        self.label = tk.Label(self.root, text="Waiting for updates...", font=("Arial", 14), justify="left")
        self.label.pack(padx=20, pady=20)
        self.status_text = ""
        self.daemon = True

    def run(self):
        self.update_label()
        self.root.mainloop()

    def update_status(self, text):
        self.status_text = text

    def update_label(self):
        self.label.config(text=self.status_text)
        self.root.after(1000, self.update_label)

trade_window = TradeStatusWindow()
trade_window.start()
# --- پایان تعریف پنجره ---

trades_log = []

def run_feature_monitor(model, all_feature_names, symbol):
    candles = get_latest_candles(symbol, CANDLE_LIMIT)
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
        # تبدیل به dict برای ساخت DataFrame صحیح
        if isinstance(features, pd.DataFrame):
            features = features.iloc[0].to_dict()
        elif isinstance(features, pd.Series):
            features = features.to_dict()
        elif isinstance(features, dict):
            pass
        else:
            print("features type not supported:", type(features))
            continue
        features_list.append(features)
    X = pd.DataFrame(features_list)
    monitor = FeatureMonitor(model, all_feature_names)
    monitor.evaluate_features(X, y=None)
    return monitor.get_active_feature_names()

def save_trades_log():
    if trades_log:
        pd.DataFrame(trades_log).to_csv("trades.csv", index=False)

def live_test():
    model, all_feature_names = load_or_train_model()
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
    last_main_loop = 0

    while True:
        now = time.time()
        if now - last_main_loop < 300:
            time.sleep(5)
            continue
        last_main_loop = now

        for symbol in LIVE_SYMBOLS:
            try:
                candles = get_latest_candles(symbol, CANDLE_LIMIT)
                price_now = get_realtime_price(symbol)
                news = get_latest_news(symbol, hours=NEWS_HOURS)

                new_candle = candles.iloc[-1].copy()
                new_candle['close'] = price_now
                candle_slice = candles.copy()
                candle_slice.iloc[-1] = new_candle

                if not news.empty:
                    candle_time = pd.to_datetime(new_candle['timestamp'], unit='s')
                    news_slice = news[news['published_at'] <= candle_time]
                else:
                    news_slice = pd.DataFrame()

                features = build_features(candle_slice, news_slice, symbol)
                if isinstance(features, pd.DataFrame):
                    features = features.iloc[0].to_dict()
                elif isinstance(features, pd.Series):
                    features = features.to_dict()
                elif isinstance(features, dict):
                    pass
                else:
                    print("features type not supported:", type(features))
                    continue
                X = pd.DataFrame([features])
                if symbol in symbol_features:
                    feature_names = symbol_features[symbol]
                else:
                    feature_names = X.columns.tolist()
                signal = predict_signals(model, feature_names, X)[0]

                # فقط در CMD این اطلاعات ساده را نمایش بده
                print(f"[{symbol}] Price: {price_now:.2f} | Signal: {signal} | Balance: {balance[symbol]:.2f}")

                # متن وضعیت برای نمایش پنجره گرافیکی
                status_lines = [
                    f"Symbol: {symbol}",
                    f"Price: {price_now:.2f}",
                    f"Signal: {signal}",
                    f"Balance: {balance[symbol]:.2f}",
                ]
                if positions[symbol] is not None:
                    status_lines.append(f"Entry: {entry_price[symbol]:.2f}")
                    status_lines.append(f"SL: {sl_price[symbol]:.2f}")
                    status_lines.append(f"TPs: {', '.join([f'{tp:.2f}' for tp in tp_prices[symbol]])}")
                    status_lines.append(f"TP idx: {tp_idx[symbol]}")
                    status_lines.append(f"QTY left: {qty_left[symbol]:.2f}")
                trade_window.update_status('\n'.join(status_lines))

                if positions[symbol] is None:
                    if signal == 2:
                        positions[symbol] = "LONG"
                        entry_price[symbol] = price_now
                        sl_price[symbol] = price_now * (1 - SL_PCT)
                        tp_prices[symbol] = [price_now * (1 + x) for x in TP_STEPS]
                        qty_left[symbol] = 1.0
                        tp_idx[symbol] = 0
                        print(f"[{symbol}] BUY at {price_now:.2f}, SL={sl_price[symbol]:.2f}, TP={tp_prices[symbol]}")
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'ENTRY',
                            'side': 'LONG',
                            'price': price_now,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                    elif signal == 0:
                        positions[symbol] = "SHORT"
                        entry_price[symbol] = price_now
                        sl_price[symbol] = price_now * (1 + SL_PCT)
                        tp_prices[symbol] = [price_now * (1 - x) for x in TP_STEPS]
                        qty_left[symbol] = 1.0
                        tp_idx[symbol] = 0
                        print(f"[{symbol}] SELL at {price_now:.2f}, SL={sl_price[symbol]:.2f}, TP={tp_prices[symbol]}")
                        trades_log.append({
                            'symbol': symbol,
                            'type': 'ENTRY',
                            'side': 'SHORT',
                            'price': price_now,
                            'balance': balance[symbol],
                            'timestamp': time.time()
                        })
                        save_trades_log()
                else:
                    fee_rate = 0.001
                    if positions[symbol] == "LONG":
                        if price_now <= sl_price[symbol]:
                            pnl = -BALANCE * qty_left[symbol] * SL_PCT
                            fee = abs(pnl) * fee_rate
                            balance[symbol] += pnl - fee
                            print(f"[{symbol}] SL HIT! CLOSE LONG at {price_now:.2f}")
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
                            fee = abs(pnl) * fee_rate
                            balance[symbol] += pnl - fee
                            print(f"[{symbol}] TP{tp_idx[symbol]+1} HIT! PARTIAL CLOSE LONG at {price_now:.2f}, qty={tp_qty}")
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
                                print(f"[{symbol}] ALL TP HIT! FULL CLOSE LONG")
                                positions[symbol] = None
                                qty_left[symbol] = 1.0
                    elif positions[symbol] == "SHORT":
                        if price_now >= sl_price[symbol]:
                            pnl = -BALANCE * qty_left[symbol] * SL_PCT
                            fee = abs(pnl) * fee_rate
                            balance[symbol] += pnl - fee
                            print(f"[{symbol}] SL HIT! CLOSE SHORT at {price_now:.2f}")
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
                            fee = abs(pnl) * fee_rate
                            balance[symbol] += pnl - fee
                            print(f"[{symbol}] TP{tp_idx[symbol]+1} HIT! PARTIAL CLOSE SHORT at {price_now:.2f}, qty={tp_qty}")
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
                                print(f"[{symbol}] ALL TP HIT! FULL CLOSE SHORT")
                                positions[symbol] = None
                                qty_left[symbol] = 1.0

            except Exception as e:
                print(f"[{symbol}] ERROR: {e}")

if __name__ == "__main__":
    live_test()
