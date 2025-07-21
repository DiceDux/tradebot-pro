import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNTPSLManager:
    """مدیریت TP/SL با یادگیری تقویتی Deep Q-Network"""
    def __init__(self, state_size, action_size, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        
        # پارامترهای DQN
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # ساخت مدل‌های Q-network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """ساخت شبکه عصبی برای DQN"""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """بروزرسانی مدل هدف با وزن‌های مدل اصلی"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """ذخیره تجربه در حافظه"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """انتخاب اکشن بر اساس state فعلی"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """آموزش مدل با نمونه‌های تصادفی از حافظه"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        # پیش‌بینی Q-values برای state فعلی و state بعدی
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # بروزرسانی Q-values با Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                q_values[i, action] = reward
            else:
                q_values[i, action] = reward + self.gamma * np.max(next_q_values[i])
        
        # آموزش مدل
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        # کاهش epsilon برای exploration کمتر
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_optimal_tpsl(self, market_state):
        """دریافت بهینه‌ترین TP/SL بر اساس شرایط بازار"""
        # تبدیل اطلاعات بازار به حالت مناسب برای مدل
        state = self._prepare_state(market_state)
        
        # پیش‌بینی اکشن
        action = self.act(state, training=False)
        
        # ترجمه اکشن به TP/SL
        return self._action_to_tpsl(action)
    
    def _prepare_state(self, market_data):
        """آماده‌سازی state از داده‌های بازار"""
        # این تابع باید اطلاعات مهم بازار را به بردار state تبدیل کند
        state = np.zeros(self.state_size)
        
        # مثال: استفاده از اطلاعات فنی و معامله
        if isinstance(market_data, dict):
            # قیمت فعلی نسبت به قیمت ورود
            state[0] = market_data.get('price_entry_ratio', 1.0)
            
            # تغییرات اخیر قیمت
            state[1] = market_data.get('recent_price_change', 0.0)
            
            # نوسانات اخیر
            state[2] = market_data.get('volatility', 0.0)
            
            # جهت روند
            state[3] = market_data.get('trend_direction', 0.0)
            
            # زمان در پوزیشن
            state[4] = market_data.get('time_in_position', 0.0)
            
            # سود/زیان فعلی
            state[5] = market_data.get('current_pnl', 0.0)
            
            # اطلاعات حجم
            state[6] = market_data.get('volume_ratio', 1.0)
            
            # شاخص‌های تکنیکال
            state[7] = market_data.get('rsi', 50.0)
            state[8] = market_data.get('macd', 0.0)
            state[9] = market_data.get('bb_position', 0.0)
        
        # تبدیل به shape مناسب برای شبکه
        return state.reshape(1, -1)
    
    def _action_to_tpsl(self, action):
        """ترجمه اکشن به TP/SL"""
        # طبقه‌بندی اکشن‌ها
        if action == 0:
            # حفظ TP/SL فعلی
            return {'adjust_tp': 0, 'adjust_sl': 0}
        elif action == 1:
            # افزایش TP
            return {'adjust_tp': 0.01, 'adjust_sl': 0}
        elif action == 2:
            # افزایش SL (trailing)
            return {'adjust_tp': 0, 'adjust_sl': 0.005}
        elif action == 3:
            # کاهش TP (برای خروج سریع‌تر)
            return {'adjust_tp': -0.005, 'adjust_sl': 0}
        elif action == 4:
            # بستن فوری پوزیشن
            return {'close_position': True}
    
    def train_from_market_data(self, historical_trades):
        """آموزش مدل با داده‌های تاریخی معاملات"""
        if not historical_trades:
            return
        
        print(f"Training RL model with {len(historical_trades)} historical trades...")
        
        for trade in historical_trades:
            # استخراج حالات و پاداش‌ها از معاملات تاریخی
            initial_state = self._extract_state_from_trade(trade, 'entry')
            final_state = self._extract_state_from_trade(trade, 'exit')
            
            # انتخاب اکشن مناسب (برای آموزش نظارت‌شده)
            if trade.get('result') == 'win':
                # اگر معامله سودآور بوده، اکشنی که به آن منجر شده
                action = self._determine_winning_action(trade)
                reward = trade.get('profit', 1.0)
            else:
                # اگر معامله ضررده بوده، اکشنی که باید انتخاب می‌شد
                action = self._determine_better_action(trade)
                reward = trade.get('loss', -1.0)
            
            # ذخیره در حافظه
            self.remember(initial_state, action, reward, final_state, True)
        
        # آموزش مدل
        if len(self.memory) > 32:
            self.replay(32)
    
    def _extract_state_from_trade(self, trade, phase):
        """استخراج حالت از داده‌های معامله"""
        # این تابع باید state را از داده‌های معامله استخراج کند
        state = np.zeros(self.state_size)
        
        # پر کردن state بر اساس داده‌های معامله
        # مثال ساده:
        if phase == 'entry':
            state[0] = 1.0  # قیمت ورود به قیمت ورود (نسبت 1)
        else:  # exit
            state[0] = trade.get('exit_price', 0) / trade.get('entry_price', 1)
            
        return state.reshape(1, -1)
    
    def _determine_winning_action(self, trade):
        """تعیین اکشنی که به سود منجر شده"""
        # بر اساس نوع خروج (TP1, TP2, ...)
        if trade.get('exit_type', '').startswith('TP'):
            tp_level = int(trade.get('exit_type', 'TP1')[-1]) - 1
            return 1 if tp_level <= 1 else 3
        return 0
    
    def _determine_better_action(self, trade):
        """تعیین اکشنی که می‌توانست به نتیجه بهتر منجر شود"""
        # برای معاملات ضررده، معمولاً اکشن بستن زودتر یا تنظیم SL بهتر مناسب است
        return 4 if trade.get('loss', 0) > 0.1 else 2
