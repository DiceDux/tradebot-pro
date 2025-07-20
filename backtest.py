import pandas as pd
import numpy as np
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, retrain_active_model, predict_signals
from feature_engineering.feature_monitor import FeatureMonitor
from feature_engineering.feature_config import FEATURE_CONFIG
import importlib

TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.6
FEE_RATE = 0.001
TRADE_BALANCE_RATIO = 0.5

def backtest(symbol, initial_balance=100):
    candles = get_latest_candles(symbol, limit=None)  # همه کندل‌ها
    news = get_latest_news(symbol, hours=None)        # همه اخبار
    try:
        model, feature_names = load_or_train_model()
    except Exception as e:
        print(f"Model load error: {e}")
        return

    if candles is None or candles.empty or len(candles) < 120:
        print(f"{symbol}: Not enough data for backtest.")
        return None

    if not news.empty:
        news['published_at'] = pd.to_datetime(news['published_at'])

    candles_for_monitor = candles.iloc[:200]
    last_candle_time = pd.to_datetime(candles_for_monitor.iloc[-1]['timestamp'], unit='s')
    news_for_monitor = news[news['published_at'] <= last_candle_time]
    X_monitor = []
    for i in range(len(candles_for_monitor) - 100, len(candles_for_monitor)):
        candle_slice = candles_for_monitor.iloc[:i + 1]
        candle_time = pd.to_datetime(candle_slice.iloc[-1]['timestamp'], unit='s')
        news_slice = news_for_monitor[news_for_monitor['published_at'] <= candle_time] if not news_for_monitor.empty else pd.DataFrame()
        feat = build_features(candle_slice, news_slice, symbol)
        if isinstance(feat, pd.DataFrame):
            feat = feat.iloc[0].to_dict()
        elif isinstance(feat, pd.Series):
            feat = feat.to_dict()
        X_monitor.append(feat)
    X_monitor_df = pd.DataFrame(X_monitor)
    monitor = FeatureMonitor(model, feature_names)
    monitor.evaluate_features(X_monitor_df)
    importlib.reload(importlib.import_module("feature_engineering.feature_config"))
    from feature_engineering.feature_config import FEATURE_CONFIG
    active_features = [f for f, v in FEATURE_CONFIG.items() if v]
    print(f"Active features for {symbol} (backtest): {active_features}")

    # ریترین مدل با همه داده‌های دیتابیس و فقط فیچرهای فعال
    X_full = []
    y_full = []
    for symbol_train in SYMBOLS:
        candles_train = get_latest_candles(symbol_train, limit=None)
        news_train = get_latest_news(symbol_train, hours=None)
        if candles_train is None or candles_train.empty or len(candles_train) < 120:
            continue
        if not news_train.empty:
            news_train['published_at'] = pd.to_datetime(news_train['published_at'])
        for i in range(100, len(candles_train)-1):
            candle_slice = candles_train.iloc[i-99:i+1]
            candle_time = pd.to_datetime(candles_train.iloc[i]['timestamp'], unit='s')
            news_slice = news_train[news_train['published_at'] <= candle_time] if not news_train.empty else pd.DataFrame()
            features = build_features(candle_slice, news_slice, symbol_train)
            if isinstance(features, pd.DataFrame):
                features = features.iloc[0].to_dict()
            elif isinstance(features, pd.Series):
                features = features.to_dict()
            X_full.append({f: features.get(f, 0.0) for f in active_features})
            y_full.append(candles_train.iloc[i].get("label", 1))
    X_full_df = pd.DataFrame(X_full)
    y_full_arr = np.array(y_full)
    model_active = retrain_active_model(X_full_df, y_full_arr, active_features)

    position = None
    entry_price = 0
    sl_price = 0
    tp_prices = []
    qty_left = 1.0
    tp_idx = 0
    balance = initial_balance
    balance_series = []
    returns = []
    n_trades = 0
    wins = 0
    trades = []

    for i in range(100, len(candles)-1):
        candle_slice = candles.iloc[i-99:i+1]
        candle_time = pd.to_datetime(candles.iloc[i]['timestamp'], unit='s')
        news_slice = news[news['published_at'] <= candle_time] if not news.empty else pd.DataFrame()
        features = build_features(candle_slice, news_slice, symbol)
        row = {f: features.get(f, 0.0) for f in active_features}
        features_df = pd.DataFrame([row])
        signal, analysis = predict_signals(model_active, active_features, features_df)
        confidence = analysis.get("confidence", 0.0)
        if confidence < THRESHOLD:
            signal = "Hold"

        price_now = candles.iloc[i]['close']
        if position is None:
            if signal == "Buy":
                position = "long"
                entry_price = price_now
                sl_price = entry_price * (1 - SL_PCT)
                tp_prices = [entry_price * (1 + tp) for tp in TP_STEPS]
                qty_left = 1.0
                tp_idx = 0
                trade_balance = balance * TRADE_BALANCE_RATIO
                n_trades += 1
                open_i = i
                print(f"{symbol} | {i} | LONG ENTRY | price={price_now:.2f} | conf={confidence:.2f} | used_balance={trade_balance:.2f}")
        else:
            trailing_sl = entry_price * (1 - SL_PCT)
            if tp_idx > 0:
                trailing_sl = tp_prices[tp_idx-1]
            if price_now <= trailing_sl:
                loss = (price_now - entry_price) * qty_left * trade_balance / entry_price
                fee = abs(loss) * FEE_RATE
                balance += loss - fee
                returns.append(loss - fee)
                position = None
                qty_left = 1.0
                print(f"{symbol} | {i} | STOP LOSS | price={price_now:.2f} | P/L={loss-fee:.2f}")
                trades.append({
                    "entry": entry_price, "exit": price_now, "type": "SL",
                    "ret": loss - fee, "start": open_i, "end": i
                })
            elif tp_idx < len(tp_prices) and price_now >= tp_prices[tp_idx]:
                sell_qty = TP_QTYS[tp_idx]
                profit = (tp_prices[tp_idx] - entry_price) * sell_qty * trade_balance / entry_price
                fee = abs(profit) * FEE_RATE
                balance += profit - fee
                returns.append(profit - fee)
                qty_left -= sell_qty
                print(f"{symbol} | {i} | TAKE PROFIT {tp_idx+1} | price={tp_prices[tp_idx]:.2f} | qty={sell_qty:.2f} | P/L={profit-fee:.2f}")
                trades.append({
                    "entry": entry_price, "exit": tp_prices[tp_idx], "type": f"TP{tp_idx+1}",
                    "ret": profit - fee, "start": open_i, "end": i
                })
                tp_idx += 1
                if qty_left <= 0.001 or tp_idx == len(tp_prices):
                    position = None
                    qty_left = 1.0
                    wins += 1
        balance_series.append(balance)

    print(f"======== {symbol} ========")
    print(f"Total Trades: {n_trades}")
    print(f"Profitable Trades: {wins}")
    print(f"Win Rate: {(wins/n_trades*100) if n_trades else 0:.1f}%")
    print(f"Final Balance: {balance:.2f}")
    print(f"Total Return: {100 * (balance-initial_balance)/initial_balance:.2f}%")
    try:
        import matplotlib.pyplot as plt
        plt.plot(balance_series)
        plt.title(f"{symbol} Balance Curve")
        plt.xlabel("Steps")
        plt.ylabel("Balance")
        plt.show()
    except Exception:
        pass
    return {
        "symbol": symbol,
        "trades": trades,
        "returns": returns,
        "balance_series": balance_series,
        "final_balance": balance,
        "win_rate": (wins/n_trades) if n_trades else 0
    }

if __name__ == "__main__":
    for symbol in SYMBOLS:
        print(f"--- Backtest for {symbol} ---")
        backtest(symbol, initial_balance=20)