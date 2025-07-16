import time
from utils.config import SYMBOLS, PREDICTION_INTERVAL
from data.candle_manager import get_latest_candles
from data.news_manager import get_latest_news
from feature_engineering.feature_engineer import build_features
from model.catboost_model import load_or_train_model, predict_signals
from data.db_manager import save_analysis



def main():
    print("TradeBot Pro is running...")
    model, feature_names = load_or_train_model()
    while True:
        for symbol in SYMBOLS:
            candles = get_latest_candles(symbol)
            news = get_latest_news(symbol)
            features = build_features(candles, news, symbol)
            features_df = features.copy()
            if isinstance(features_df, dict):
                import pandas as pd
                features_df = pd.DataFrame([features_df])
            signal, analysis = predict_signals(model, feature_names, features_df)
            save_analysis(symbol, analysis)
            print(f"{symbol}: {signal} | تحلیل ذخیره شد")
        time.sleep(PREDICTION_INTERVAL)

if __name__ == "__main__":
    main()
