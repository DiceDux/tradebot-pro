"""
نسخه خط فرمان (بدون UI) برای اجرا بدون مشکل Tkinter
"""
import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
import threading
import argparse
import sqlite3

from data.candle_manager import get_latest_candles, keep_last_200_candles
from data.news_manager import get_latest_news
from data.fetch_online import fetch_candles_binance, save_candles_to_db, fetch_news_newsapi, save_news_to_db
from feature_engineering.feature_engineer import build_features
from feature_engineering.sentiment_finbert import analyze_sentiment_finbert
from utils.price_fetcher import get_realtime_price
from model.enhanced_base_model import EnhancedBaseModel
from feature_engineering.adaptive_feature_selector import AdaptiveFeatureSelector
from rl_models.dqn_tpsl_manager import DQNTPSLManager

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BALANCE = 100
TP_STEPS = [0.03, 0.05, 0.07]
TP_QTYS = [0.3, 0.3, 0.4]
SL_PCT = 0.02
THRESHOLD = 0.7
CANDLE_LIMIT = 200

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "ede1e0b0db7140fdbbd20f6f1b440cb9")

class SmartTraderCLI:
    def _clear_last_lines(self, n=6):
        """پاک کردن چند خط قبلی در ترمینال"""
        for _ in range(n):
            print("\033[A\033[K", end='')  # بالا رفتن یک خط و پاک کردن آن
            
    """نسخه خط فرمان ربات معامله‌گر هوشمند"""
    def __init__(self):
        self.base_model = EnhancedBaseModel()
        self.feature_selector = None
        self.tpsl_manager = None
        
        # وضعیت معاملات
        self.trades_log = []
        self.positions = {symbol: None for symbol in SYMBOLS}
        self.entry_price = {symbol: 0 for symbol in SYMBOLS}
        self.sl_price = {symbol: 0 for symbol in SYMBOLS}
        self.tp_prices = {symbol: [] for symbol in SYMBOLS}
        self.qty_left = {symbol: 1.0 for symbol in SYMBOLS}
        self.tp_idx = {symbol: 0 for symbol in SYMBOLS}
        self.balance = {symbol: BALANCE for symbol in SYMBOLS}
        
        # نگهداری قیمت‌های فعلی
        self.latest_prices = {symbol: 0.0 for symbol in SYMBOLS}
        
    def initialize(self):
        """راه‌اندازی اولیه ربات"""
        print("Initializing Smart Trading Bot (CLI version)...")
        
        # بارگذاری مدل پایه یا ساخت آن اگر وجود ندارد
        try:
            self.base_model.load()
        except:
            print("Base model not found, need to train it first.")
            self._train_base_model()
        
        # ساخت سلکتور فیچر
        all_feature_names = self.base_model.feature_names
        self.feature_selector = AdaptiveFeatureSelector(
            self.base_model.model, 
            all_feature_names,
            top_n=30,  # تعداد فیچرهای انتخابی
            update_interval_hours=48  # بروزرسانی هر 48 ساعت
        )
        
        # ساخت مدیریت TP/SL با یادگیری تقویتی
        self.tpsl_manager = DQNTPSLManager(
            state_size=10,  # تعداد ویژگی‌های حالت
            action_size=5,   # تعداد اکشن‌های ممکن
            memory_size=10000
        )
        
        # دریافت اولیه داده‌ها
        for symbol in SYMBOLS:
            self._fetch_and_store_data(symbol)
        
        print("Smart Trading Bot initialized successfully!")
        return self
        
    def _train_base_model(self):
        """آموزش مدل پایه با تمام فیچرها و داده‌های موجود"""
        print("Training base model with all features...")
        
        all_features = []
        all_labels = []
        
        # جمع‌آوری داده‌ها از همه نمادها
        for symbol in SYMBOLS:
            print(f"Processing data for {symbol}...")
            
            # دریافت کندل‌ها و اخبار
            candles = get_latest_candles(symbol, limit=None)
            if candles is None or candles.empty:
                print(f"No candles found for {symbol}, skipping")
                continue
                
            news = get_latest_news(symbol, hours=None)
            if not news.empty:
                news['published_at'] = pd.to_datetime(news['published_at'])
            
            # برچسب‌گذاری داده‌ها
            labeled_candles = self._make_labels(candles)
            
            # ساخت فیچرها
            for i in range(100, len(labeled_candles)):
                candle_slice = labeled_candles.iloc[i-99:i+1]
                candle_time = pd.to_datetime(labeled_candles.iloc[i]['timestamp'], unit='s')
                news_slice = news[news['published_at'] <= candle_time] if not news.empty else pd.DataFrame()
                
                features = build_features(candle_slice, news_slice, symbol)
                if isinstance(features, pd.DataFrame):
                    features_dict = features.iloc[0].to_dict()
                else:
                    features_dict = features.to_dict()
                
                all_features.append(features_dict)
                all_labels.append(labeled_candles.iloc[i]['label'])
                
                if i % 500 == 0:
                    print(f"Processed {i}/{len(labeled_candles)} candles for {symbol}")
        
        # تبدیل به DataFrame
        X = pd.DataFrame(all_features)
        y = np.array(all_labels)
        
        print(f"Training base model with {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: {np.bincount(y)}")
        
        # آموزش مدل
        self.base_model.train(X, y, class_balance=True)
        
        return self
    
    def _make_labels(self, candles, threshold=0.01, future_steps=6):
        """برچسب‌گذاری داده‌ها برای آموزش"""
        labeled = candles.copy()
        labels = []
        
        for i in range(len(candles) - future_steps):
            current = candles.iloc[i]['close']
            future_prices = candles.iloc[i+1:i+future_steps+1]['close'].values
            
            # محاسبه بیشترین سود و ضرر آینده
            max_profit = (max(future_prices) - current) / current
            max_loss = (current - min(future_prices)) / current
            
            if max_profit > threshold and max_profit > max_loss:
                # سیگنال خرید
                labels.append(2)
            elif max_loss > threshold and max_loss > max_profit:
                # سیگنال فروش
                labels.append(0)
            else:
                # سیگنال نگهداری
                labels.append(1)
        
        # برچسب‌ها را به داده‌ها اضافه می‌کنیم
        labeled = labeled.iloc[:len(labels)].copy()
        labeled['label'] = labels
        
        return labeled
    
    def _fetch_and_store_data(self, symbol):
        """دریافت و ذخیره داده‌های جدید"""
        print(f"Fetching latest data for {symbol}...")
        
        # دریافت کندل‌ها
        candles = fetch_candles_binance(symbol, interval="4h", limit=200)
        save_candles_to_db(candles)
        keep_last_200_candles(symbol)
        
        # دریافت اخبار
        news = fetch_news_newsapi(symbol, limit=25, api_key=NEWSAPI_KEY)
        for n in news:
            n["sentiment_score"] = analyze_sentiment_finbert((n.get("title") or "") + " " + (n.get("content") or ""))
        save_news_to_db(news)
        
        print(f"Saved {len(candles)} candles and {len(news)} news items for {symbol}")
    
    def analyze_market(self, symbol):
        """تحلیل بازار و صدور سیگنال با معماری دو لایه"""
        print(f"Analyzing market for {symbol}...")
        
        # دریافت داده‌های بازار
        self._fetch_and_store_data(symbol)
        candles = get_latest_candles(symbol, CANDLE_LIMIT)
        news = get_latest_news(symbol, hours=CANDLE_LIMIT * 4)
        
        if candles is None or candles.empty:
            print(f"No candles found for {symbol}, skipping analysis")
            return "Hold", 0.0
        
        # اضافه کردن قیمت فعلی به آخرین کندل
        price_now = self.latest_prices.get(symbol, 0.0) or candles.iloc[-1]['close']
        new_candle = candles.iloc[-1].copy()
        new_candle['close'] = price_now
        candle_slice = candles.copy()
        candle_slice.iloc[-1] = new_candle
        
        # ساخت تمام فیچرها
        all_features = build_features(candle_slice, news, symbol)
        if isinstance(all_features, pd.DataFrame):
            features_dict = all_features.iloc[0].to_dict()
        else:
            features_dict = all_features.to_dict()
        
        # انتخاب فیچرهای مناسب برای شرایط فعلی بازار
        market_data = pd.DataFrame([features_dict])
        selected_features = self.feature_selector.select_features(market_data)
        
        # استفاده از مدل پایه با فیچرهای منتخب
        X_filtered = pd.DataFrame({f: [features_dict.get(f, 0.0)] for f in selected_features})
        
        # پیش‌بینی با مدل پایه
        pred_class, pred_proba, confidence = self.base_model.predict(X_filtered)
        
        # مهم: تبدیل مقدار numpy به مقادیر پایتون
        class_idx = int(pred_class[0])
        conf_value = float(confidence[0])
        
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        signal = signal_map.get(class_idx, "Hold")
        
        print(f"Signal for {symbol}: {signal} with confidence {conf_value:.2f}")
        
        return signal, conf_value
    
    def execute_trades(self):
        """اجرای معاملات بر اساس سیگنال‌ها"""
        for symbol in SYMBOLS:
            try:
                # بروزرسانی قیمت فعلی
                try:
                    price_now = get_realtime_price(symbol)
                    self.latest_prices[symbol] = price_now
                except Exception as e:
                    print(f"Error getting price for {symbol}: {e}")
                    price_now = 0.0
                
                # فقط اگر پوزیشن فعال نداریم، تحلیل انجام می‌دهیم
                if self.positions[symbol] is None:
                    signal, confidence = self.analyze_market(symbol)
                    
                    # اگر اطمینان کافی وجود دارد، معامله می‌کنیم
                    if confidence >= THRESHOLD:
                        if signal == "Buy":
                            self._open_position(symbol, "LONG", price_now)
                        elif signal == "Sell":
                            self._open_position(symbol, "SHORT", price_now)
                
                # مدیریت TP/SL
                self._manage_positions(symbol, price_now)
                
            except Exception as e:
                import traceback
                print(f"Error in execute_trades for {symbol}: {e}")
                print(traceback.format_exc())
    
    def _open_position(self, symbol, direction, price):
        """باز کردن پوزیشن جدید"""
        print(f"Opening {direction} position for {symbol} at {price}")
        
        self.positions[symbol] = direction
        self.entry_price[symbol] = price
        self.qty_left[symbol] = 1.0
        self.tp_idx[symbol] = 0
        
        if direction == "LONG":
            self.sl_price[symbol] = price * (1 - SL_PCT)
            self.tp_prices[symbol] = [price * (1 + tp) for tp in TP_STEPS]
        else:  # SHORT
            self.sl_price[symbol] = price * (1 + SL_PCT)
            self.tp_prices[symbol] = [price * (1 - tp) for tp in TP_STEPS]
        
        # ثبت در لاگ معاملات
        self.trades_log.append({
            'symbol': symbol,
            'type': 'ENTRY',
            'side': direction,
            'price': price,
            'balance': self.balance[symbol],
            'timestamp': time.time()
        })
        
        # ذخیره لاگ معاملات
        self._save_trades_log()
    
    def _manage_positions(self, symbol, price_now):
        """مدیریت پوزیشن‌های فعلی (TP/SL)"""
        if self.positions[symbol] == "LONG":
            # چک SL
            if price_now <= self.sl_price[symbol]:
                self._close_position(symbol, price_now, "SL")
            
            # چک TP
            elif self.tp_idx[symbol] < len(self.tp_prices[symbol]) and price_now >= self.tp_prices[symbol][self.tp_idx[symbol]]:
                self._take_profit(symbol, price_now)
                
        elif self.positions[symbol] == "SHORT":
            # چک SL
            if price_now >= self.sl_price[symbol]:
                self._close_position(symbol, price_now, "SL")
            
            # چک TP
            elif self.tp_idx[symbol] < len(self.tp_prices[symbol]) and price_now <= self.tp_prices[symbol][self.tp_idx[symbol]]:
                self._take_profit(symbol, price_now)
    
    def _close_position(self, symbol, price, reason):
        """بستن کامل پوزیشن"""
        direction = self.positions[symbol]
        
        if reason == "SL":
            # محاسبه زیان
            pnl = -self.balance[symbol] * self.qty_left[symbol] * SL_PCT
            fee = abs(pnl) * 0.001
            self.balance[symbol] += pnl - fee
            
            # ثبت در لاگ
            self.trades_log.append({
                'symbol': symbol,
                'type': reason,
                'side': direction,
                'entry_price': self.entry_price[symbol],
                'exit_price': price,
                'qty': self.qty_left[symbol],
                'pnl': pnl,
                'fee': fee,
                'balance': self.balance[symbol],
                'timestamp': time.time()
            })
            
        # ریست وضعیت
        self.positions[symbol] = None
        self.qty_left[symbol] = 1.0
        self._save_trades_log()
    
    def _take_profit(self, symbol, price):
        """برداشت سود در نقطه TP"""
        tp_qty = TP_QTYS[self.tp_idx[symbol]]
        direction = self.positions[symbol]
        
        # محاسبه سود
        pnl = self.balance[symbol] * tp_qty * TP_STEPS[self.tp_idx[symbol]]
        fee = abs(pnl) * 0.001
        self.balance[symbol] += pnl - fee
        
        # ثبت در لاگ
        self.trades_log.append({
            'symbol': symbol,
            'type': f'TP{self.tp_idx[symbol]+1}',
            'side': direction,
            'entry_price': self.entry_price[symbol],
            'exit_price': price,
            'qty': tp_qty,
            'pnl': pnl,
            'fee': fee,
            'balance': self.balance[symbol],
            'timestamp': time.time()
        })
        
        # بروزرسانی مقدار باقی‌مانده
        self.qty_left[symbol] -= tp_qty
        self.tp_idx[symbol] += 1
        
        # اگر پوزیشن کامل بسته شده
        if self.qty_left[symbol] <= 0 or self.tp_idx[symbol] >= len(self.tp_prices[symbol]):
            self.positions[symbol] = None
            self.qty_left[symbol] = 1.0
        
        self._save_trades_log()
    
    def _save_trades_log(self):
        """ذخیره لاگ معاملات در فایل"""
        if self.trades_log:
            pd.DataFrame(self.trades_log).to_csv("trades.csv", index=False)
    
    def run(self):
        """اجرای ربات در خط فرمان"""
        self.initialize()
        
        # شروع ترد آپدیت قیمت‌ها
        threading.Thread(target=self._update_price_thread, daemon=True).start()
        
        last_execution = 0
        interval = 60  # بررسی هر 60 ثانیه
        
        print("Starting main trading loop...")
        try:
            while True:
                now = time.time()
                if now - last_execution >= interval:
                    self.execute_trades()
                    self._print_status()
                    last_execution = now
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("Trading bot stopped by user")
        except Exception as e:
            import traceback
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())
    
    def _update_price_thread(self):
        """thread آپدیت قیمت‌ها"""
        while True:
            try:
                for symbol in SYMBOLS:
                    try:
                        price = get_realtime_price(symbol)
                        self.latest_prices[symbol] = price
                    except Exception as e:
                        print(f"Price fetch error for {symbol}: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Error in price update thread: {e}")
                time.sleep(5)
    
    def _print_status(self):
        """چاپ وضعیت فعلی در خط فرمان"""
        print("\n" + "="*50)
        print(f"STATUS UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        for symbol in SYMBOLS:
            price = self.latest_prices.get(symbol, 0.0)
            bal = self.balance.get(symbol, BALANCE)
            
            print(f"{symbol}: ${price:.2f} | Balance: ${bal:.2f}")
            
            if self.positions[symbol]:
                print(f"  Position: {self.positions[symbol]}")
                print(f"  Entry: {self.entry_price[symbol]:.2f} | SL: {self.sl_price[symbol]:.2f}")
                print(f"  TPs: {[f'{tp:.2f}' for tp in self.tp_prices[symbol]]}")
                print(f"  Current TP: {self.tp_idx[symbol]+1}/{len(TP_STEPS)} | Qty left: {self.qty_left[symbol]:.2f}")
            else:
                print("  No active position")
        
        print("="*50 + "\n")
    
    def run_backtest(self):
        """اجرای بک‌تست روی داده‌های تاریخی"""
        self.initialize()
        print("Running backtest on historical data...")
        
        # تنظیم پارامترهای بک‌تست
        start_date = "2018-01-01"  # از ابتدای 2021 (3 سال به جای 2.5 سال)
        end_date = "2025-07-28"    # تا دیروز
        initial_balance = 1000.0  # سرمایه اولیه برای هر نماد
        
        backtest_results = {}
        
        for symbol in SYMBOLS:
            print(f"\n===== Backtesting {symbol} from {start_date} to {end_date} =====")
            
            # تنظیم مجدد متغیرهای حساب برای بک‌تست
            # دانلود داده‌های تاریخی جدید
            self._download_historical_data_for_backtest(symbol, start_date, end_date)
            self.balance[symbol] = initial_balance
            self.positions[symbol] = None
            self.trades_log = []
            
            # دریافت داده‌های تاریخی
            historical_candles = self._get_historical_data(symbol, start_date, end_date)
            if historical_candles is None or historical_candles.empty:
                print(f"No historical data available for {symbol}")
                continue
                
            print(f"Loaded {len(historical_candles)} historical candles")
            
            # اجرای بک‌تست روی هر شمع
            total_candles = len(historical_candles)
            lookback = 100  # تعداد شمع‌های قبلی برای تحلیل
            
            for i in range(lookback, total_candles):
                if i % 50 == 0:
                    print(f"Processing candle {i}/{total_candles} ({(i/total_candles*100):.1f}%)")
                
                # شبیه‌سازی شرایط فعلی بازار با دیتای تاریخی
                current_candle = historical_candles.iloc[i]
                current_time = pd.to_datetime(current_candle['timestamp'], unit='s')
                
                # آخرین قیمت در این لحظه زمانی
                current_price = current_candle['close']
                self.latest_prices[symbol] = current_price
                
                # فقط اگر پوزیشن فعال نداریم، تحلیل انجام می‌دهیم
                if self.positions[symbol] is None:
                    # گرفتن بخشی از داده‌های تاریخی تا این لحظه
                    candles_slice = historical_candles.iloc[i-lookback:i+1]
                    
                    # شبیه‌سازی اخبار (یا می‌توانیم از دیتاست واقعی اخبار استفاده کنیم)
                    dummy_news = pd.DataFrame()
                    
                    # شبیه‌سازی ساخت فیچرها
                    features = build_features(candles_slice, dummy_news, symbol)
                    if isinstance(features, pd.DataFrame):
                        features_dict = features.iloc[0].to_dict()
                    else:
                        features_dict = features.to_dict()
                    
                    # انتخاب فیچرهای مناسب
                    market_data = pd.DataFrame([features_dict])
                    selected_features = self.feature_selector.select_features(market_data)
                    
                    # استفاده از مدل با فیچرهای منتخب
                    X_filtered = pd.DataFrame({f: [features_dict.get(f, 0.0)] for f in selected_features})
                    pred_class, pred_proba, confidence = self.base_model.predict(X_filtered)
                    
                    # تبدیل به مقادیر اسکالر
                    class_idx = int(pred_class[0]) if isinstance(pred_class[0], (np.ndarray, np.generic)) else int(pred_class[0])
                    conf_value = float(confidence[0]) if isinstance(confidence[0], (np.ndarray, np.generic)) else float(confidence[0])
                    
                    signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
                    signal = signal_map.get(class_idx, "Hold")
                    
                    # اگر اطمینان کافی وجود دارد، معامله می‌کنیم
                    if conf_value >= THRESHOLD:
                        if signal == "Buy":
                            self._open_position(symbol, "LONG", current_price, current_time.timestamp())
                        elif signal == "Sell":
                            self._open_position(symbol, "SHORT", current_price, current_time.timestamp())
                
                # مدیریت TP/SL
                self._manage_positions(symbol, current_price, current_time.timestamp())
                
                # هر 500 شمع وضعیت را نمایش بده
                if i % 500 == 0 and i > 0:
                    self._print_backtest_status(symbol)
            
            # آنالیز نتایج بک‌تست برای این نماد
            wins, losses = self._analyze_backtest_results(symbol)
            backtest_results[symbol] = {
                "initial_balance": initial_balance,
                "final_balance": self.balance[symbol],
                "profit_loss": self.balance[symbol] - initial_balance,
                "profit_percent": ((self.balance[symbol] / initial_balance) - 1) * 100,
                "total_trades": len(self.trades_log),
                "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0
            }
            
            # نمایش نتایج
            self._print_backtest_results(symbol, backtest_results[symbol])
            
            # ذخیره تاریخچه معاملات
            if self.trades_log:
                pd.DataFrame(self.trades_log).to_csv(f"backtest_{symbol}_trades.csv", index=False)
        
        print("\n===== Overall Backtest Summary =====")
        total_profit = sum(r["profit_loss"] for r in backtest_results.values())
        avg_win_rate = sum(r["win_rate"] for r in backtest_results.values()) / len(backtest_results) if backtest_results else 0
        
        print(f"Total profit across all symbols: ${total_profit:.2f}")
        print(f"Average win rate: {avg_win_rate:.2f}%")
        
        return backtest_results

    def _get_historical_news(self, symbol, start_ts, end_ts):
        """دریافت اخبار تاریخی برای بک‌تست"""
        try:
            import pymysql
            from utils.config import DB_CONFIG
            
            # اتصال به MySQL
            conn = pymysql.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_CONFIG["database"],
                port=DB_CONFIG["port"]
            )
            
            # کوئری برای دریافت اخبار
            query = """
            SELECT * FROM news 
            WHERE symbol = %s 
            AND timestamp BETWEEN %s AND %s 
            ORDER BY timestamp
            """
            
            # اجرای کوئری
            df = pd.read_sql(query, conn, params=[symbol, start_ts, end_ts])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
                
            # تبدیل ستون timestamp به datetime
            if 'timestamp' in df.columns:
                df['published_at'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
            
        except Exception as e:
            print(f"Error getting historical news: {e}")
            return pd.DataFrame()

    def _download_historical_data_for_backtest(self, symbol, start_date, end_date):
        """دانلود داده‌های تاریخی برای بک‌تست و ذخیره در MySQL"""
        print(f"Downloading historical data for {symbol} from {start_date} to {end_date}...")
        
        # تبدیل تاریخ‌ها به timestamp
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        try:
            # دانلود داده‌های تاریخی از بایننس (مثلاً 4 ساعته)
            from binance.client import Client
            client = Client("", "")  # کلید‌های API اختیاری هستند برای داده‌های تاریخی
            
            # دانلود داده‌ها در چند بخش (هر بخش 1000 کندل)
            all_candles = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                klines = client.get_historical_klines(
                    symbol=symbol, 
                    interval=Client.KLINE_INTERVAL_4HOUR,
                    start_str=current_ts,
                    end_str=end_ts,
                    limit=1000
                )
                
                if not klines:
                    break
                    
                print(f"Downloaded {len(klines)} candles")
                all_candles.extend(klines)
                
                # آخرین timestamp دریافت شده + 1 برای دریافت بخش بعدی
                current_ts = klines[-1][0] + 1
                
            # تبدیل به DataFrame
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignored'
            ])
            
            # تبدیل انواع داده
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                            'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # تبدیل timestamp از میلی‌ثانیه به ثانیه
            df['timestamp'] = df['timestamp'].apply(lambda x: int(x / 1000))
            df['symbol'] = symbol
            
            # ذخیره در MySQL
            print(f"Saving {len(df)} candles to MySQL database")
            
            import pymysql
            from utils.config import DB_CONFIG
            
            # اتصال به MySQL
            conn = pymysql.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_CONFIG["database"],
                port=DB_CONFIG["port"]
            )
            
            cursor = conn.cursor()
            
            # بررسی ساختار جدول
            cursor.execute("SHOW COLUMNS FROM candles")
            columns = [col[0] for col in cursor.fetchall()]
            
            # فقط ستون‌های موجود در جدول را انتخاب می‌کنیم
            df_columns = [col for col in df.columns if col in columns]
            df_filtered = df[df_columns]
            
            # ذخیره هر رکورد
            for _, row in df_filtered.iterrows():
                placeholders = ", ".join(["%s"] * len(df_columns))
                columns_str = ", ".join(df_columns)
                
                # بررسی وجود تکراری
                check_query = f"SELECT COUNT(*) FROM candles WHERE symbol = %s AND timestamp = %s"
                cursor.execute(check_query, (row['symbol'], row['timestamp']))
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # اگر تکراری نبود، اضافه کن
                    insert_query = f"INSERT INTO candles ({columns_str}) VALUES ({placeholders})"
                    cursor.execute(insert_query, tuple(row[col] for col in df_columns))
            
            conn.commit()
            conn.close()
            print(f"Data saved successfully!")
            
            return df
            
        except Exception as e:
            print(f"Error downloading historical data: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _get_historical_data(self, symbol, start_date, end_date):
        """دریافت داده‌های تاریخی برای بک‌تست از MySQL"""
        try:
            import pymysql
            from utils.config import DB_CONFIG
            
            # تبدیل تاریخ به timestamp
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())
            
            # اتصال به MySQL
            conn = pymysql.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_CONFIG["database"],
                port=DB_CONFIG["port"]
            )
            
            # کوئری برای دریافت داده‌های تاریخی با محدوده زمانی
            query = """
            SELECT * FROM candles
            WHERE symbol = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            # اجرای کوئری با pandas
            df = pd.read_sql(query, conn, params=[symbol, start_ts, end_ts])
            conn.close()
            
            if df.empty:
                print(f"No historical data found for {symbol} between {start_date} and {end_date}")
                return None
                
            print(f"Found {len(df)} candles between {start_date} and {end_date}")
            return df
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _open_position(self, symbol, direction, price, timestamp=None):
        """باز کردن پوزیشن جدید (با پشتیبانی از timestamp برای بک‌تست)"""
        print(f"Opening {direction} position for {symbol} at {price}")
        
        self.positions[symbol] = direction
        self.entry_price[symbol] = price
        self.qty_left[symbol] = 1.0
        self.tp_idx[symbol] = 0
        
        if direction == "LONG":
            self.sl_price[symbol] = price * (1 - SL_PCT)
            self.tp_prices[symbol] = [price * (1 + tp) for tp in TP_STEPS]
        else:  # SHORT
            self.sl_price[symbol] = price * (1 + SL_PCT)
            self.tp_prices[symbol] = [price * (1 - tp) for tp in TP_STEPS]
        
        # ثبت در لاگ معاملات
        self.trades_log.append({
            'symbol': symbol,
            'type': 'ENTRY',
            'side': direction,
            'price': price,
            'balance': self.balance[symbol],
            'timestamp': timestamp or time.time()
        })
        
        # ذخیره لاگ معاملات
        self._save_trades_log()

    def _manage_positions(self, symbol, price_now, timestamp=None):
        """مدیریت پوزیشن‌های فعلی (TP/SL) - با پشتیبانی از timestamp برای بک‌تست"""
        if self.positions[symbol] == "LONG":
            # چک SL
            if price_now <= self.sl_price[symbol]:
                self._close_position(symbol, price_now, "SL", timestamp)
            
            # چک TP
            elif self.tp_idx[symbol] < len(self.tp_prices[symbol]) and price_now >= self.tp_prices[symbol][self.tp_idx[symbol]]:
                self._take_profit(symbol, price_now, timestamp)
                
        elif self.positions[symbol] == "SHORT":
            # چک SL
            if price_now >= self.sl_price[symbol]:
                self._close_position(symbol, price_now, "SL", timestamp)
            
            # چک TP
            elif self.tp_idx[symbol] < len(self.tp_prices[symbol]) and price_now <= self.tp_prices[symbol][self.tp_idx[symbol]]:
                self._take_profit(symbol, price_now, timestamp)

    def _close_position(self, symbol, price, reason, timestamp=None):
        """بستن کامل پوزیشن - با پشتیبانی از timestamp برای بک‌تست"""
        direction = self.positions[symbol]
        
        if reason == "SL":
            # محاسبه زیان
            pnl = -self.balance[symbol] * self.qty_left[symbol] * SL_PCT
            fee = abs(pnl) * 0.001
            self.balance[symbol] += pnl - fee
            
            # ثبت در لاگ
            self.trades_log.append({
                'symbol': symbol,
                'type': reason,
                'side': direction,
                'entry_price': self.entry_price[symbol],
                'exit_price': price,
                'qty': self.qty_left[symbol],
                'pnl': pnl,
                'fee': fee,
                'balance': self.balance[symbol],
                'timestamp': timestamp or time.time()
            })
            
        # ریست وضعیت
        self.positions[symbol] = None
        self.qty_left[symbol] = 1.0
        self._save_trades_log()

    def _take_profit(self, symbol, price, timestamp=None):
        """برداشت سود در نقطه TP - با پشتیبانی از timestamp برای بک‌تست"""
        tp_qty = TP_QTYS[self.tp_idx[symbol]]
        direction = self.positions[symbol]
        
        # محاسبه سود
        pnl = self.balance[symbol] * tp_qty * TP_STEPS[self.tp_idx[symbol]]
        fee = abs(pnl) * 0.001
        self.balance[symbol] += pnl - fee
        
        # ثبت در لاگ
        self.trades_log.append({
            'symbol': symbol,
            'type': f'TP{self.tp_idx[symbol]+1}',
            'side': direction,
            'entry_price': self.entry_price[symbol],
            'exit_price': price,
            'qty': tp_qty,
            'pnl': pnl,
            'fee': fee,
            'balance': self.balance[symbol],
            'timestamp': timestamp or time.time()
        })
        
        # بروزرسانی مقدار باقی‌مانده
        self.qty_left[symbol] -= tp_qty
        self.tp_idx[symbol] += 1
        
        # اگر پوزیشن کامل بسته شده
        if self.qty_left[symbol] <= 0 or self.tp_idx[symbol] >= len(self.tp_prices[symbol]):
            self.positions[symbol] = None
            self.qty_left[symbol] = 1.0
        
        self._save_trades_log()

    def _print_backtest_status(self, symbol, i=0, total=0, current_time=None, current_price=0):
        """نمایش وضعیت فعلی بک‌تست با بروزرسانی در جای ثابت"""
        trades_count = len([t for t in self.trades_log if t['symbol'] == symbol])
        win_trades = len([t for t in self.trades_log if t['symbol'] == symbol and t.get('pnl', 0) > 0])
        loss_trades = len([t for t in self.trades_log if t['symbol'] == symbol and t.get('pnl', 0) < 0])
        
        # پاک کردن نمایش قبلی
        if hasattr(self, 'last_status_lines'):
            self._clear_last_lines(self.last_status_lines)
        
        # نمایش وضعیت
        print("\n" + "="*50)
        print(f"BACKTEST STATUS: {symbol} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        if i > 0 and total > 0:
            progress = i / total * 100
            print(f"Progress: {i}/{total} candles ({progress:.1f}%)")
        
        if current_time:
            print(f"Current date: {current_time.strftime('%Y-%m-%d %H:%M')} - Price: ${current_price:.2f}")
        
        print(f"Balance: ${self.balance[symbol]:.2f}")
        print(f"Trades: {trades_count} (Win: {win_trades}, Loss: {loss_trades})")
        print(f"Win rate: {(win_trades/trades_count*100) if trades_count > 0 else 0:.2f}%")
        
        if self.positions[symbol]:
            print(f"Active position: {self.positions[symbol]}")
            print(f"Entry: ${self.entry_price[symbol]:.2f} | SL: ${self.sl_price[symbol]:.2f}")
            print(f"TPs: {[f'${tp:.2f}' for tp in self.tp_prices[symbol]]}")
            print(f"Current TP: {self.tp_idx[symbol]+1}/{len(TP_STEPS)} | Qty left: {self.qty_left[symbol]:.2f}")
        else:
            print("No active position")
        
        print("="*50)
        
        # تعداد خطوط چاپ شده را ذخیره می‌کنیم
        self.last_status_lines = 12 if self.positions[symbol] else 9

    def _analyze_backtest_results(self, symbol):
        """تحلیل نتایج بک‌تست"""
        wins = len([t for t in self.trades_log if t['symbol'] == symbol and t.get('pnl', 0) > 0])
        losses = len([t for t in self.trades_log if t['symbol'] == symbol and t.get('pnl', 0) < 0])
        return wins, losses

    def _print_backtest_results(self, symbol, results):
        """نمایش نتایج بک‌تست"""
        print(f"\n========== {symbol} Backtest Results ==========")
        print(f"Initial balance: ${results['initial_balance']:.2f}")
        print(f"Final balance: ${results['final_balance']:.2f}")
        print(f"Total P/L: ${results['profit_loss']:.2f} ({results['profit_percent']:.2f}%)")
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']*100:.2f}%")
        print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart Trading Bot CLI')
    parser.add_argument('--train_base', action='store_true', help='Train the base model with all features')
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of live trading')
    args = parser.parse_args()
    
    bot = SmartTraderCLI()
    
    if args.train_base:
        print("Training base model with all features...")
        bot._train_base_model()
    elif args.backtest:
        print("Running backtest mode...")
        bot.run_backtest()
    else:
        print("Starting live trading...")
        bot.run()
