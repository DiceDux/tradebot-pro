import pandas as pd
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model
import numpy as np

# تنظیمات پله‌ای
TP_STEPS = [0.03, 0.05, 0.07]  # پله‌های سود (3%، 5%، 7%)
TP_QTYS = [0.3, 0.3, 0.4]      # نسبت حجم فروش هر پله
SL_PCT = 0.02                  # حد ضرر 2 درصد
THRESHOLD = 0.7                # حداقل درصد اطمینان برای ورود/خروج

def backtest(symbol):
    candles = get_latest_candles(symbol, limit=3000)
    news = get_latest_news(symbol, hours=365*24)
    model = load_or_train_model()

    if candles is None or candles.empty or len(candles) < 120:
        print(f"{symbol}: Not enough data for backtest.")
        return None

    position = None
    entry_price = 0
    sl_price = 0
    tp_prices = []
    qty_left = 1.0
    tp_idx = 0
    balance = 10000  # سرمایه اولیه
    balance_series = []
    returns = []
    n_trades = 0
    wins = 0
    trades = []

    if not news.empty:
        news['published_at'] = pd.to_datetime(news['published_at'])

    for i in range(100, len(candles)-1):
        candle_slice = candles.iloc[i-99:i+1]
        news_slice = news[news['published_at'] <= candles.iloc[i]['timestamp']]
        features = build_features(candle_slice, news_slice, symbol)
        signal, confidence = "Hold", 0.0
        candle_time = pd.to_datetime(candles.iloc[i]['timestamp'], unit='s')
        news_slice = news[news['published_at'] <= candle_time]
        features = build_features(candle_slice, news_slice, symbol)

        # تشخیص فیچرهای فاندامنتال و تکنیکال (بر اساس نام ستون‌ها)
        fund_keys = [k for k in features.columns if 'news' in k]
        tech_keys = [k for k in features.columns if 'news' not in k]
        fund_score = abs(features[fund_keys]).sum(axis=1).values[0] if fund_keys else 0
        tech_score = abs(features[tech_keys]).sum(axis=1).values[0] if tech_keys else 0

        print(f"{symbol} | {i} | signal={signal} | conf={confidence*100:.1f}% | fund={fund_score:.2f} | tech={tech_score:.2f}")

        try:
            proba = model.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            confidence = proba[pred_idx]
            signal = ["Sell", "Hold", "Buy"][pred_idx]
        except Exception:
            signal = "Hold"
            confidence = 0.0
            print(f"{symbol} | i={i} | signal error")

        close = candles.iloc[i]['close']
        open_next = candles.iloc[i+1]['open']

        # ورود به معامله: فقط اگر سیگنال Buy و اطمینان >= 70%
        if position is None and signal == "Buy" and confidence >= THRESHOLD:
            position = "long"
            entry_price = open_next
            sl_price = entry_price * (1 - SL_PCT)
            tp_prices = [entry_price * (1 + x) for x in TP_STEPS]
            qty_left = 1.0
            tp_idx = 0
            n_trades += 1
            print(f'{symbol} | {i} | BUY {confidence*100:.1f}% | entry: {entry_price:.2f}')

        # مدیریت پوزیشن باز (پله‌ای)
        elif position == "long":
            # پله‌های سود
            while tp_idx < len(tp_prices) and close >= tp_prices[tp_idx]:
                sell_qty = qty_left * TP_QTYS[tp_idx]
                profit = (tp_prices[tp_idx] - entry_price) / entry_price
                balance *= (1 + profit * sell_qty)
                returns.append(profit * sell_qty)
                trades.append({
                    'type': f'TP{tp_idx+1}',
                    'entry': entry_price,
                    'exit': tp_prices[tp_idx],
                    'profit_pct': profit*100,
                    'qty': sell_qty,
                    'confidence': confidence
                })
                print(f'{symbol} | {i} | TP{tp_idx+1} hit at {tp_prices[tp_idx]:.2f} ({sell_qty*100:.1f}%)')
                qty_left -= sell_qty
                tp_idx += 1
                # بعد اولین پله، SL به نقطه ورود (BreakEven)
                if tp_idx == 1:
                    sl_price = entry_price

            # حد ضرر
            if close <= sl_price:
                profit = (close - entry_price) / entry_price
                balance *= (1 + profit * qty_left)
                returns.append(profit * qty_left)
                trades.append({
                    'type': 'SL',
                    'entry': entry_price,
                    'exit': close,
                    'profit_pct': profit*100,
                    'qty': qty_left,
                    'confidence': confidence
                })
                print(f'{symbol} | {i} | SL hit at {close:.2f} ({qty_left*100:.1f}%)')
                if profit > 0:
                    wins += 1
                position = None
                entry_price = 0
                qty_left = 1.0
                tp_idx = 0

            # سیگنال Sell و اطمینان >= 70%
            elif signal == "Sell" and confidence >= THRESHOLD:
                profit = (open_next - entry_price) / entry_price
                balance *= (1 + profit * qty_left)
                returns.append(profit * qty_left)
                trades.append({
                    'type': 'SELL',
                    'entry': entry_price,
                    'exit': open_next,
                    'profit_pct': profit*100,
                    'qty': qty_left,
                    'confidence': confidence
                })
                print(f'{symbol} | {i} | SELL {confidence*100:.1f}% | exit: {open_next:.2f} ({qty_left*100:.1f}%)')
                if profit > 0:
                    wins += 1
                position = None
                entry_price = 0
                qty_left = 1.0
                tp_idx = 0

            # همه پله‌ها گرفته شده؟
            if tp_idx >= len(tp_prices):
                position = None
                qty_left = 1.0
                tp_idx = 0

        balance_series.append(balance)

    # اگر معامله باز مانده، با آخرین قیمت می‌بندیم
    if position == "long":
        exit_price = candles.iloc[-1]['close']
        profit = (exit_price - entry_price) / entry_price
        balance *= (1 + profit * qty_left)
        returns.append(profit * qty_left)
        trades.append({
            'type': 'FORCE_EXIT',
            'entry': entry_price,
            'exit': exit_price,
            'profit_pct': profit*100,
            'qty': qty_left,
            'confidence': confidence
        })
        if profit > 0:
            wins += 1

    winrate = wins / n_trades if n_trades > 0 else 0
    total_return = (balance / 10000 - 1) * 100
    print(f"\nBacktest {symbol}:")
    print(f"  سرمایه اولیه: 10000")
    print(f"  سرمایه نهایی: {balance:.2f}")
    print(f"  سود/ضرر درصدی: {total_return:.2f}%")
    print(f"  تعداد معاملات: {n_trades} | درصد برد: {winrate*100:.2f}%\n")

    # ذخیره نتایج سری سرمایه
    pd.Series(balance_series).to_csv(f"backtest_{symbol}_balance.csv", index=False)
    # ذخیره معاملات پله‌ای
    pd.DataFrame(trades).to_csv(f"backtest_{symbol}_trades.csv", index=False)
    return balance, total_return

if __name__ == "__main__":
    total_init = 0
    total_final = 0
    results = []
    for symbol in SYMBOLS:
        result = backtest(symbol)
        if result:
            final_balance, total_return = result
            total_init += 10000
            total_final += final_balance
            results.append((symbol, final_balance, total_return))
    print("\n----- خلاصه کل پورتفوی -----")
    print(f"سرمایه اولیه کل: {total_init}")
    print(f"سرمایه نهایی کل: {total_final:.2f}")
    print(f"سود/ضرر کل: {((total_final/total_init)-1)*100:.2f}%")
    print("جزئیات هر ارز:")
    for symbol, final_balance, total_return in results:
        print(f"  {symbol}: سرمایه نهایی = {final_balance:.2f} | سود/ضرر = {total_return:.2f}%")
