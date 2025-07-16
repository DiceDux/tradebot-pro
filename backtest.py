import pandas as pd
import numpy as np
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals

TP_STEPS = [0.03, 0.05, 0.07]   # پله‌های سود (3٪، 5٪، 7٪)
TP_QTYS = [0.3, 0.3, 0.4]       # نسبت حجم فروش هر پله
SL_PCT = 0.02                   # حد ضرر 2 درصد
THRESHOLD = 0.6                 # حداقل اطمینان برای تریگر سیگنال (درخواست کاربر)
FEE_RATE = 0.001                # کارمزد صرافی (مثلاً CoinEx، 0.1%)
TRADE_BALANCE_RATIO = 0.5       # فقط 50 درصد بالانس وارد هر معامله شود

def backtest(symbol, initial_balance=100):
    candles = get_latest_candles(symbol, limit=3000)
    news = get_latest_news(symbol, hours=365*24)
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

    fund_features = [
        'news_count_24h',
        'news_sentiment_mean_24h',
        'news_sentiment_max_24h',
        'news_sentiment_min_24h',
        'news_shock_24h',
        'news_pos_ratio_24h',
        'news_neg_ratio_24h'
    ]

    position = None
    entry_price = 0
    sl_price = 0
    tp_prices = []
    qty_left = 1.0
    tp_idx = 0
    balance = initial_balance  # سرمایه اولیه دلاری (قابل تنظیم)
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

        # تضمین DataFrame بودن یک ردیفی و سازگار با مدل
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            features_df = features.to_frame().T
        else:
            features_df = features.copy()

        signal, analysis = predict_signals(model, feature_names, features_df)
        confidence = analysis.get("confidence", 0.0)

        # اگر سیگنال اطمینان کافی نداشت، Hold کن
        if confidence < THRESHOLD:
            signal = "Hold"

        price_now = candles.iloc[i]['close']

        # مدیریت پوزیشن‌ها
        if position is None:
            if signal == "Buy":
                position = "long"
                entry_price = price_now
                sl_price = entry_price * (1 - SL_PCT)
                tp_prices = [entry_price * (1 + tp) for tp in TP_STEPS]
                qty_left = 1.0
                tp_idx = 0
                trade_balance = balance * TRADE_BALANCE_RATIO  # فقط 50 درصد بالانس وارد معامله
                n_trades += 1
                open_i = i
                print(f"{symbol} | {i} | LONG ENTRY | price={price_now:.2f} | conf={confidence:.2f} | used_balance={trade_balance:.2f}")
        else:
            # Trailing Stop: حد ضرر بعد هر TP می‌رود روی TP قبلی
            trailing_sl = entry_price * (1 - SL_PCT)
            if tp_idx > 0:
                trailing_sl = tp_prices[tp_idx-1]
            # حد ضرر
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
            # حد سود پله‌ای
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

    # جمع‌بندی نتایج
    print(f"======== {symbol} ========")
    print(f"Total Trades: {n_trades}")
    print(f"Profitable Trades: {wins}")
    print(f"Win Rate: {(wins/n_trades*100) if n_trades else 0:.1f}%")
    print(f"Final Balance: {balance:.2f}")
    print(f"Total Return: {100 * (balance-initial_balance)/initial_balance:.2f}%")
    # رسم منحنی سرمایه (در صورت نیاز)
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
        backtest(symbol, initial_balance=20)  # هر عددی خواستی تست بگیر
