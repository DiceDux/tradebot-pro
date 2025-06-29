import pandas as pd
from utils.config import SYMBOLS
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model
import numpy as np

def backtest(symbol):
    candles = get_latest_candles(symbol, limit=3000)
    news = get_latest_news(symbol, hours=365*24)
    model = load_or_train_model()

    if candles is None or candles.empty or len(candles) < 120:
        print(f"{symbol}: Not enough data for backtest.")
        return

    position = None
    entry_price = 0
    balance = 10000  # فرض سرمایه اولیه
    balance_series = []
    returns = []
    n_trades = 0
    wins = 0

    for i in range(100, len(candles)-1):
        candle_slice = candles.iloc[i-99:i+1]
        news_slice = news[news['published_at'] <= candles.iloc[i]['timestamp']]
        features = build_features(candle_slice, news_slice, symbol)
        signal, analysis = "Hold", None
        try:
            y_pred = model.predict(features)[0]
            signal = ["Sell", "Hold", "Buy"][int(y_pred)]
        except Exception:
            signal = "Hold"
            print(f"{symbol} | i={i} | signal={signal}")

        # ورود/خروج به معامله
        if position is None and signal == "Buy":
            position = "long"
            entry_price = candles.iloc[i+1]['open']  # ورود با قیمت باز کندل بعدی
            n_trades += 1
        elif position == "long" and signal == "Sell":
            exit_price = candles.iloc[i+1]['open']
            profit = (exit_price - entry_price) / entry_price
            balance *= (1 + profit)
            returns.append(profit)
            if profit > 0:
                wins += 1
            position = None
            entry_price = 0
        balance_series.append(balance)

    # اگر معامله باز مانده، با آخرین قیمت می‌بندیم
    if position == "long":
        exit_price = candles.iloc[-1]['close']
        profit = (exit_price - entry_price) / entry_price
        balance *= (1 + profit)
        returns.append(profit)
        if profit > 0:
            wins += 1

    winrate = wins / n_trades if n_trades > 0 else 0
    total_return = (balance / 10000 - 1) * 100
    print(f"Backtest {symbol}: Trades: {n_trades} | Winrate: {winrate:.2%} | Total Return: {total_return:.2f}%")

    # ذخیره نتایج سری سرمایه
    pd.Series(balance_series).to_csv(f"backtest_{symbol}_balance.csv", index=False)

if __name__ == "__main__":
    for symbol in SYMBOLS:
        backtest(symbol)