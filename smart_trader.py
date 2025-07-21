import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
import threading
import tkinter as tk

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

class SmartTraderBot:
    """
    ربات معامله‌گر هوشمند با معماری دو لایه‌ای:
    1. مدل پایه (تمام فیچرها)
    2. انتخاب‌کننده فیچر هوشمند
    """
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
        
        # متغیرهای UI
        self.status_texts = {symbol: "" for symbol in SYMBOLS}
        self.latest_prices = {symbol: 0.0 for symbol in SYMBOLS}
    
    def initialize(self):
        """راه‌اندازی اولیه ربات"""
        print("Initializing Smart Trading Bot...")
        
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
        
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        signal = signal_map.get(pred_class[0], "Hold")
        
        print(f"Signal for {symbol}: {signal} with confidence {confidence[0]:.2f}")
        
        return signal, confidence[0]
    
    def execute_trades(self):
        """اجرای معاملات بر اساس سیگنال‌ها"""
        for symbol in SYMBOLS:
            try:
                # بروزرسانی قیمت فعلی
                price_now = get_realtime_price(symbol)
                self.latest_prices[symbol] = price_now
                
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
                
                # بروزرسانی وضعیت UI
                self._update_status(symbol, price_now)
            
            except Exception as e:
                print(f"Error in execute_trades for {symbol}: {e}")
    
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
    
    def _update_status(self, symbol, price_now):
        """بروزرسانی وضعیت برای UI"""
        status_lines = [
            f"Symbol: {symbol}",
            f"Price: {price_now:.2f}",
            f"Balance: {self.balance[symbol]:.2f}"
        ]
        
        # اضافه کردن اطلاعات پوزیشن فعال
        if self.positions[symbol] is not None:
            status_lines.extend([
                f"Position: {self.positions[symbol]}",
                f"Entry: {self.entry_price[symbol]:.2f}",
                f"SL: {self.sl_price[symbol]:.2f}",
            ])
            
            # اطلاعات TP
            tps = [f"TP{idx+1}: {tp:.2f} | QTY: {TP_QTYS[idx]}" for idx, tp in enumerate(self.tp_prices[symbol])]
            status_lines.extend(tps)
            status_lines.append(f"TP idx: {self.tp_idx[symbol]}")
            status_lines.append(f"QTY left: {self.qty_left[symbol]:.2f}")
        else:
            status_lines.append("No active position")
        
        # اطلاعات معاملات اخیر
        recent_trades = [t for t in self.trades_log if t['symbol'] == symbol][-5:]
        if recent_trades:
            status_lines.append("Recent trades:")
            for t in recent_trades:
                entry = t.get('entry_price', t.get('price', 0))
                exit_val = t.get('exit_price', '-')
                status_lines.append(
                    f" {t['type']} | {t['side']} | Entry: {entry:.2f} | Exit: {exit_val}"
                )
        
        self.status_texts[symbol] = '\n'.join(status_lines)
    
    def run(self):
        """اجرای ربات در حلقه اصلی"""
        self.initialize()
        
        # راه‌اندازی UI
        self._start_ui()
        
        last_execution = 0
        interval = 60  # بررسی هر 60 ثانیه
        
        print("Starting main trading loop...")
        try:
            while True:
                now = time.time()
                if now - last_execution >= interval:
                    self.execute_trades()
                    last_execution = now
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("Trading bot stopped by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
    
    def _start_ui(self):
        """راه‌اندازی رابط کاربری ساده"""
        def update_price_thread():
            while True:
                for symbol in SYMBOLS:
                    try:
                        price = get_realtime_price(symbol)
                        self.latest_prices[symbol] = price
                    except Exception as e:
                        print(f"Price fetch error for {symbol}: {e}")
                time.sleep(1)
        
        def update_gui(label):
            def gui_update():
                all_status = '\n\n'.join([self.status_texts[symbol] for symbol in SYMBOLS])
                label.config(text=all_status)
                label.after(1000, gui_update)
            return gui_update
        
        # راه‌اندازی GUI
        root = tk.Tk()
        root.title("Smart Trading Bot")
        label = tk.Label(root, text="Initializing...", font=("Arial", 13), justify="left")
        label.pack(padx=20, pady=20)
        
        # شروع ترد آپدیت قیمت‌ها
        threading.Thread(target=update_price_thread, daemon=True).start()
        
        # تنظیم آپدیت GUI
        update_func = update_gui(label)
        label.after(1000, update_func)
        
        # شروع حلقه اصلی UI
        threading.Thread(target=root.mainloop, daemon=True).start()

# فقط در صورت اجرا به عنوان اسکریپت اصلی
if __name__ == "__main__":
    bot = SmartTraderBot()
    bot.run()
