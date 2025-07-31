import numpy as np
import pandas as pd
import joblib
import time
import os
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import logging

SELECTOR_PATH = "model/feature_selector.pkl"
ACTIVE_FEATURES_PATH = "model/active_features.pkl"

class AdaptiveFeatureSelector:
    """
    انتخاب‌کننده هوشمند فیچرها بر اساس شرایط بازار با استفاده از الگوریتم ژنتیک
    برای یافتن بهترین ترکیب از فیچرها
    """
    def __init__(self, base_model, all_feature_names, top_n=30, 
                 backtest_update_hours=24, live_update_hours=4,
                 population_size=20, generations=10):
        self.base_model = base_model
        self.all_feature_names = all_feature_names
        self.top_n = top_n
        self.backtest_update_seconds = backtest_update_hours * 3600
        self.live_update_seconds = live_update_hours * 3600
        self.last_update_time = None
        self.active_features = None
        
        # پارامترهای الگوریتم ژنتیک
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = 0.7
        self.mutation_rate = 0.2
        self.min_features = min(15, len(all_feature_names) // 3)
        self.max_features = min(40, len(all_feature_names))
        
        # برای ذخیره اطلاعات ارزیابی
        self.feature_importance = {}
        self.combinations_evaluated = 0
        
        # فایل‌های ذخیره‌سازی
        self.last_update_file = "model/last_feature_update.txt"
        self.feature_importance_file = "model/feature_importance.pkl"
        
        # تنظیم لاگر
        self.logger = logging.getLogger('feature_selector')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            if not os.path.exists("logs"):
                os.makedirs("logs")
            handler = logging.FileHandler('logs/feature_selection.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # بارگیری وضعیت قبلی اگر موجود باشد
        self._load_state()
        
    def _load_state(self):
        """بارگیری وضعیت قبلی انتخاب فیچر"""
        try:
            if os.path.exists(ACTIVE_FEATURES_PATH):
                data = joblib.load(ACTIVE_FEATURES_PATH)
                
                if isinstance(data, dict):
                    self.active_features = data.get('features', [])
                    self.feature_importance = data.get('importance', {})
                    self.combinations_evaluated = data.get('combinations_evaluated', 0)
                    print(f"Loaded {len(self.active_features)} active features from disk")
                else:
                    self.active_features = data
                    print(f"Loaded {len(self.active_features)} active features from disk (legacy format)")
            
            if os.path.exists(self.last_update_file):
                with open(self.last_update_file, 'r') as f:
                    last_update_str = f.read().strip()
                    self.last_update_time = float(last_update_str)
                    print(f"Last feature selection update: {datetime.fromtimestamp(self.last_update_time)}")
        except Exception as e:
            print(f"Error loading selector state: {e}")
            self.logger.error(f"Error loading selector state: {e}")
    
    def _save_state(self):
        """ذخیره وضعیت فعلی انتخاب فیچر"""
        try:
            if not os.path.exists("model"):
                os.makedirs("model")
                
            if self.active_features:
                data = {
                    'features': self.active_features,
                    'importance': self.feature_importance,
                    'combinations_evaluated': self.combinations_evaluated,
                    'timestamp': datetime.now()
                }
                joblib.dump(data, ACTIVE_FEATURES_PATH)
            
            if self.last_update_time:
                with open(self.last_update_file, 'w') as f:
                    f.write(str(self.last_update_time))
                    
            # ذخیره لیست فیچرها در فایل متنی برای خوانایی بیشتر
            if self.active_features:
                with open("model/selected_features.txt", "w") as f:
                    f.write(f"Updated: {datetime.now()}\n")
                    f.write(f"Feature count: {len(self.active_features)}\n\n")
                    f.write("Selected Features:\n")
                    
                    # مرتب‌سازی بر اساس اهمیت
                    sorted_features = sorted(
                        [(f, self.feature_importance.get(f, 0)) for f in self.active_features],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for i, (feature, importance) in enumerate(sorted_features):
                        f.write(f"{i+1}. {feature}: {importance:.6f}\n")
                
        except Exception as e:
            print(f"Error saving selector state: {e}")
            self.logger.error(f"Error saving selector state: {e}")
    
    def _is_update_needed(self, is_backtest=True):
        """بررسی نیاز به بروزرسانی فیچرها"""
        if self.last_update_time is None:
            return True
        
        time_passed = time.time() - self.last_update_time
        update_interval = self.backtest_update_seconds if is_backtest else self.live_update_seconds
        
        return time_passed > update_interval
    
    def select_features(self, market_data, force_update=False, is_backtest=True):
        """
        انتخاب بهترین ترکیب فیچرها با استفاده از الگوریتم ژنتیک
        
        Args:
            market_data: داده‌های اخیر بازار به صورت DataFrame
            force_update: اجبار به بروزرسانی حتی اگر زمان آن نرسیده باشد
            is_backtest: آیا برای بک‌تست است یا لایو ترید
            
        Returns:
            list: فیچرهای انتخاب‌شده
        """
        # اگر هنوز زمان بروزرسانی نرسیده و فیچرهای فعال داریم
        if not force_update and not self._is_update_needed(is_backtest) and self.active_features:
            print("Using cached feature selection")
            return self.active_features
        
        print(f"Selecting optimal feature combination for {'backtest' if is_backtest else 'live trading'}...")
        self.logger.info(f"Starting feature selection for {'backtest' if is_backtest else 'live trading'}")
        
        start_time = datetime.now()
        
        # آماده‌سازی داده‌ها برای انتخاب فیچر
        try:
            X, y = self._prepare_data_for_feature_selection(market_data, is_backtest)
            self.logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
            
            # اجرای الگوریتم ژنتیک
            self._run_genetic_algorithm(X, y)
            
        except Exception as e:
            print(f"Error during feature selection: {e}")
            self.logger.error(f"Error during feature selection: {e}")
            
            # اگر فیچرهای قبلی داریم، آنها را برگردانیم
            if self.active_features:
                return self.active_features
            
            # در غیر این صورت، از روش ساده‌تر استفاده کنیم
            try:
                self.active_features = self._fallback_feature_selection(market_data)
            except:
                # اگر هیچ روشی کار نکرد، یک زیرمجموعه پیش‌فرض را برگردانیم
                print("Using default features as fallback")
                essential_features = ['close', 'open', 'high', 'low', 'volume', 'rsi', 'macd', 'ema20', 'ema50']
                self.active_features = [f for f in essential_features if f in self.all_feature_names]
                
                # اضافه کردن فیچرهای دیگر تا رسیدن به top_n
                remaining_features = [f for f in self.all_feature_names if f not in self.active_features]
                needed = max(0, self.top_n - len(self.active_features))
                if needed > 0 and remaining_features:
                    self.active_features.extend(random.sample(remaining_features, min(needed, len(remaining_features))))
                
        # بروزرسانی وضعیت
        self.last_update_time = time.time()
        self._save_state()
        
        duration = datetime.now() - start_time
        print(f"Feature selection completed in {duration.total_seconds()//60} minutes {duration.total_seconds()%60:.0f} seconds")
        print(f"Selected {len(self.active_features)} features")
        
        # نمایش 10 فیچر برتر
        self._print_top_features(10)
        
        return self.active_features
    
    def _print_top_features(self, n=10):
        """نمایش فیچرهای برتر"""
        if not self.feature_importance:
            return
            
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop feature importance:")
        for i, (feature, importance) in enumerate(sorted_features[:n]):
            print(f"{i+1}. {feature}: {importance:.6f}")
    
    def _prepare_data_for_feature_selection(self, market_data, is_backtest):
        """آماده‌سازی داده‌ها برای انتخاب فیچر"""
        # اگر market_data کافی است، از آن استفاده می‌کنیم
        if isinstance(market_data, pd.DataFrame) and len(market_data) >= 100:
            X = market_data.copy()
            
            # اطمینان از وجود تمام فیچرها
            for feature in self.all_feature_names:
                if feature not in X.columns:
                    X[feature] = 0
                    
            # انتخاب فقط فیچرهای در all_feature_names
            X = X[self.all_feature_names]
            
            # برچسب‌ها - از آنجا که market_data معمولاً برچسب ندارد،
            # می‌توانیم از داده‌های تاریخی استفاده کنیم یا برچسب‌های موقت بسازیم
            y = pd.Series(np.ones(len(X)))
            return X, y
        
        # در غیر این صورت، داده‌ها را از دیتابیس دریافت می‌کنیم
        from data.candle_manager import get_latest_candles
        from data.news_manager import get_latest_news
        from feature_engineering.feature_engineer import build_features
        
        samples = []
        labels = []
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        for symbol in symbols:
            # تعداد کندل‌های متفاوت برای بک‌تست و لایو
            candle_count = 1000 if is_backtest else 500
            candles = get_latest_candles(symbol, candle_count)
            
            if candles is None or len(candles) < 100:
                continue
            
            print(f"Processing {len(candles)} candles for {symbol}")
            
            # ساخت برچسب‌ها و فیچرها
            for i in range(50, len(candles) - 20):
                current_price = candles.iloc[i]['close']
                future_price = candles.iloc[i+20]['close']
                price_change = (future_price - current_price) / current_price
                
                # برچسب‌گذاری
                if price_change > 0.01:  # افزایش بیش از 1%
                    label = 2  # خرید
                elif price_change < -0.01:  # کاهش بیش از 1%
                    label = 0  # فروش
                else:
                    label = 1  # نگهداری
                
                # استخراج فیچرها
                try:
                    candle_slice = candles.iloc[i-50:i+1].copy()
                    news = get_latest_news(symbol, hours=48 if is_backtest else 24)
                    features = build_features(candle_slice, news, symbol)
                    
                    if isinstance(features, pd.DataFrame) and not features.empty:
                        row = features.iloc[0].to_dict()
                        samples.append(row)
                        labels.append(label)
                except Exception as e:
                    self.logger.error(f"Error building features: {e}")
        
        if not samples:
            raise ValueError("No data could be prepared for feature selection")
        
        # تبدیل به DataFrame
        X = pd.DataFrame(samples)
        y = pd.Series(labels)
        
        # اطمینان از وجود تمام فیچرها
        for feature in self.all_feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # انتخاب فقط فیچرهای موجود در all_feature_names
        X = X[self.all_feature_names]
        
        print(f"Prepared {len(X)} samples with {len(self.all_feature_names)} features")
        
        return X, y
    
    def _run_genetic_algorithm(self, X, y):
        """اجرای الگوریتم ژنتیک برای یافتن بهترین ترکیب فیچرها"""
        # ایجاد جمعیت اولیه
        population = self._create_initial_population()
        
        best_chromosome = None
        best_fitness = -float('inf')
        
        print(f"Running genetic algorithm with population={self.population_size}, generations={self.generations}")
        self.logger.info(f"Running genetic algorithm with {self.population_size} chromosomes for {self.generations} generations")
        
        for generation in range(self.generations):
            # ارزیابی هر کروموزوم
            fitness_scores = []
            for chromosome in population:
                fitness = self._evaluate_fitness(chromosome, X, y)
                fitness_scores.append(fitness)
            
            # یافتن بهترین کروموزوم در این نسل
            best_idx = np.argmax(fitness_scores)
            gen_best_chromosome = population[best_idx]
            gen_best_fitness = fitness_scores[best_idx]
            
            # به‌روزرسانی بهترین کروموزوم کلی
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = gen_best_chromosome
                
                # تبدیل به لیست فیچرها
                selected_indices = [i for i, gene in enumerate(best_chromosome) if gene == 1]
                self.active_features = [self.all_feature_names[i] for i in selected_indices]
                
                print(f"Generation {generation+1}: New best score {best_fitness:.4f} with {len(self.active_features)} features")
                self.logger.info(f"Generation {generation+1}: New best score {best_fitness:.4f} with {len(self.active_features)} features")
            
            # ایجاد نسل بعدی
            new_population = []
            
            # elitism: حفظ بهترین‌ها
            elite_count = max(1, int(self.population_size * 0.1))
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]])
            
            # ایجاد فرزندان جدید
            while len(new_population) < self.population_size:
                # انتخاب والدین
                parent1 = self._select_by_tournament(population, fitness_scores)
                parent2 = self._select_by_tournament(population, fitness_scores)
                
                # ترکیب و جهش
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                new_population.append(child)
            
            # جایگزینی نسل
            population = new_population
            
            # گزارش پیشرفت
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Generation {generation+1}/{self.generations}: Avg fitness={avg_fitness:.4f}, Best fitness={gen_best_fitness:.4f}")
        
        # محاسبه اهمیت فیچرها با استفاده از بهترین ترکیب
        if self.active_features:
            self._calculate_feature_importance(X, y)
    
    def _create_initial_population(self):
        """ایجاد جمعیت اولیه برای الگوریتم ژنتیک"""
        population = []
        
        # افزودن کروموزوم فعال قبلی (اگر وجود دارد)
        if self.active_features:
            chromosome = [1 if feature in self.active_features else 0 for feature in self.all_feature_names]
            population.append(chromosome)
        
        # ایجاد کروموزوم‌های تصادفی با تراکم‌های متفاوت
        densities = [0.2, 0.3, 0.4, 0.5]
        
        while len(population) < self.population_size:
            density = random.choice(densities)
            chromosome = []
            
            for _ in range(len(self.all_feature_names)):
                if random.random() < density:
                    chromosome.append(1)
                else:
                    chromosome.append(0)
            
            # اطمینان از محدوده تعداد فیچرها
            feature_count = sum(chromosome)
            if self.min_features <= feature_count <= self.max_features:
                population.append(chromosome)
        
        # اگر هنوز به تعداد کافی نرسیدیم، روش دیگری استفاده کنیم
        while len(population) < self.population_size:
            n_features = random.randint(self.min_features, self.max_features)
            selected_indices = random.sample(range(len(self.all_feature_names)), n_features)
            chromosome = [1 if i in selected_indices else 0 for i in range(len(self.all_feature_names))]
            population.append(chromosome)
        
        return population
    
    def _evaluate_fitness(self, chromosome, X, y):
        """ارزیابی میزان مناسب بودن یک ترکیب فیچر"""
        # انتخاب فیچرهای فعال
        selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
        
        # اگر تعداد فیچرها خارج از محدوده است، امتیاز منفی بدهیم
        feature_count = len(selected_indices)
        if feature_count < self.min_features or feature_count > self.max_features:
            return -1
        
        # انتخاب فیچرها
        selected_features = [self.all_feature_names[i] for i in selected_indices]
        X_selected = X[selected_features]
        
        # امتیازدهی با اعتبارسنجی متقابل
        try:
            # مدل سبک برای سرعت بیشتر
            model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
            
            # اعتبارسنجی متقابل زمانی
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_selected):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
            
            # میانگین امتیازها
            avg_score = np.mean(scores)
            
            # جریمه برای تعداد زیاد فیچر (تشویق به انتخاب فیچرهای کمتر)
            penalty = 0.001 * feature_count
            
            # افزایش تعداد ارزیابی‌ها
            self.combinations_evaluated += 1
            
            return avg_score - penalty
            
        except Exception as e:
            self.logger.error(f"Error in fitness evaluation: {e}")
            return -1
    
    def _crossover(self, parent1, parent2):
        """ترکیب دو والد برای ایجاد فرزند"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # تقاطع دو نقطه‌ای
        points = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        
        # اطمینان از محدوده تعداد فیچرها
        feature_count = sum(child)
        
        if self.min_features <= feature_count <= self.max_features:
            return child
        
        # تنظیم تعداد فیچرها
        if feature_count < self.min_features:
            # افزایش تعداد فیچرها
            zeros = [i for i, gene in enumerate(child) if gene == 0]
            if zeros:
                to_add = min(self.min_features - feature_count, len(zeros))
                for i in random.sample(zeros, to_add):
                    child[i] = 1
        elif feature_count > self.max_features:
            # کاهش تعداد فیچرها
            ones = [i for i, gene in enumerate(child) if gene == 1]
            if ones:
                to_remove = min(feature_count - self.max_features, len(ones))
                for i in random.sample(ones, to_remove):
                    child[i] = 0
        
        return child
    
    def _mutate(self, chromosome):
        """اعمال جهش روی کروموزوم"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # معکوس کردن بیت
        
        # اطمینان از محدوده تعداد فیچرها
        feature_count = sum(mutated)
        
        if self.min_features <= feature_count <= self.max_features:
            return mutated
        
        # تنظیم تعداد فیچرها
        if feature_count < self.min_features:
            # افزایش تعداد فیچرها
            zeros = [i for i, gene in enumerate(mutated) if gene == 0]
            if zeros:
                to_add = min(self.min_features - feature_count, len(zeros))
                for i in random.sample(zeros, to_add):
                    mutated[i] = 1
        elif feature_count > self.max_features:
            # کاهش تعداد فیچرها
            ones = [i for i, gene in enumerate(mutated) if gene == 1]
            if ones:
                to_remove = min(feature_count - self.max_features, len(ones))
                for i in random.sample(ones, to_remove):
                    mutated[i] = 0
        
        return mutated
    
    def _select_by_tournament(self, population, fitness_scores, tournament_size=3):
        """انتخاب یک کروموزوم با روش مسابقه"""
        if not population:
            return None
            
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [(fitness_scores[i], population[i]) for i in tournament_indices]
        return max(tournament_fitness, key=lambda x: x[0])[1]
    
    def _calculate_feature_importance(self, X, y):
        """محاسبه اهمیت فیچرهای انتخاب شده"""
        try:
            if not self.active_features:
                return
                
            X_selected = X[self.active_features]
            
            # استفاده از Random Forest برای محاسبه اهمیت فیچرها
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_selected, y)
            
            importances = model.feature_importances_
            self.feature_importance = dict(zip(self.active_features, importances))
            
            self.logger.info("Feature importance calculated successfully")
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            print(f"Error calculating feature importance: {e}")
    
    def _fallback_feature_selection(self, market_data):
        """روش ساده‌تر برای انتخاب فیچر در صورت خطا"""
        print("Using fallback feature selection method")
        
        # ترکیب چند روش برای انتخاب فیچر
        importance_methods = {
            # اهمیت فیچرها از مدل پایه
            'model_importance': self._get_model_importance(),
            
            # تغییرپذیری فیچرها در شرایط فعلی بازار
            'variance': self._get_market_variance(market_data),
            
            # همبستگی متقابل (برای کاهش فیچرهای همبسته)
            'correlation': self._get_feature_correlation(market_data)
        }
        
        # ترکیب معیارها با وزن‌های مختلف
        combined_scores = {}
        weights = {'model_importance': 0.5, 'variance': 0.3, 'correlation': 0.2}
        
        for feature in self.all_feature_names:
            score = 0
            for method, importance in importance_methods.items():
                if feature in importance:
                    score += weights[method] * importance[feature]
            combined_scores[feature] = score
        
        # مرتب‌سازی و انتخاب بهترین فیچرها
        sorted_features = sorted(combined_scores.items(), key=lambda x: -x[1])
        selected_features = [f for f, _ in sorted_features[:self.top_n]]
        
        # اطمینان از وجود حداقل فیچرهای اساسی
        essential_features = ['close', 'ema20', 'ema50', 'rsi', 'volume']
        for feat in essential_features:
            if feat in self.all_feature_names and feat not in selected_features:
                selected_features.append(feat)
                
        return selected_features
    
    def _get_model_importance(self):
        """دریافت اهمیت فیچرها از مدل پایه"""
        if not hasattr(self.base_model, 'get_feature_importance'):
            return {}
        
        try:
            importances = self.base_model.get_feature_importance()
            return {f: imp for f, imp in zip(self.all_feature_names, importances)}
        except Exception as e:
            self.logger.error(f"Error getting model importance: {e}")
            return {}
    
    def _get_market_variance(self, market_data):
        """محاسبه واریانس/تغییرات فیچرها در بازار فعلی"""
        result = {}
        for feat in self.all_feature_names:
            if feat in market_data.columns:
                result[feat] = market_data[feat].std()
        return result
    
    def _get_feature_correlation(self, market_data):
        """محاسبه همبستگی بین فیچرها (برای کاهش فیچرهای همبسته)"""
        result = {}
        common_features = [f for f in self.all_feature_names if f in market_data.columns]
        
        if not common_features:
            return result
            
        corr_matrix = market_data[common_features].corr().abs()
        
        for feat in common_features:
            # 1 - average correlation (هر چه همبستگی کمتر باشد، اهمیت بیشتر است)
            result[feat] = 1 - (corr_matrix[feat].sum() - 1) / (len(common_features) - 1)
            
        return result
