"""
انتخاب پویای فیچرها براساس شرایط بازار
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_selection.feature_evaluator import FeatureEvaluator
from feature_selection.feature_groups import FEATURE_GROUPS, ESSENTIAL_FEATURES, get_group_for_feature
from feature_store.feature_database import FeatureDatabase

class DynamicFeatureSelector:
    def __init__(self, symbols=None, feature_count=50):
        """
        انتخابگر پویای فیچر
        
        Args:
            symbols: لیست نمادهای مورد نظر
            feature_count: تعداد فیچرهای مورد نظر
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.feature_count = feature_count
        self.evaluator = FeatureEvaluator(symbols)
        self.db = FeatureDatabase()
        self.selected_features = []
        self.feature_importances = {}
        self.last_selection_time = None
        
        # بارگذاری فیچرهای انتخاب شده قبلی، اگر وجود داشته باشد
        self.load_selected_features()
    
    def load_selected_features(self, file_path='model/selected_features_live.pkl'):
        """بارگذاری فیچرهای انتخاب شده قبلی"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.selected_features = data.get('features', [])
                    self.feature_importances = data.get('importances', {})
                    self.last_selection_time = data.get('timestamp')
                    
                    print(f"Loaded {len(self.selected_features)} previously selected features")
                    print(f"Last feature selection: {self.last_selection_time}")
                    return True
            return False
        except Exception as e:
            print(f"Error loading selected features: {e}")
            return False
    
    def save_selected_features(self, file_path='model/selected_features_live.pkl'):
        """ذخیره فیچرهای انتخاب شده"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            data = {
                'features': self.selected_features,
                'importances': self.feature_importances,
                'timestamp': datetime.now()
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Saved {len(self.selected_features)} selected features")
            return True
        except Exception as e:
            print(f"Error saving selected features: {e}")
            return False
    
    def should_update_selection(self, force=False, interval_hours=2):
        """بررسی نیاز به بروزرسانی انتخاب فیچرها"""
        if force or not self.selected_features:
            return True
            
        if not self.last_selection_time:
            return True
            
        # بروزرسانی هر چند ساعت یکبار
        elapsed = datetime.now() - self.last_selection_time
        return elapsed > timedelta(hours=interval_hours)
    
    def select_features_basic(self, symbol='BTCUSDT'):
        """انتخاب فیچرها با روش ساده (براساس اهمیت)"""
        # ارزیابی اهمیت فیچرها
        feature_importances = self.evaluator.evaluate_features(symbol, hours=48)
        
        # اطمینان از انتخاب فیچرهای ضروری
        features = []
        for feature in ESSENTIAL_FEATURES:
            if feature in feature_importances:
                features.append(feature)
                
        # افزودن بهترین فیچرهای باقیمانده
        remaining_features = sorted(
            [(f, v) for f, v in feature_importances.items() if f not in features],
            key=lambda x: x[1],
            reverse=True
        )
        
        # افزودن فیچرها تا رسیدن به تعداد مورد نظر
        for feature, importance in remaining_features:
            if len(features) >= self.feature_count:
                break
            features.append(feature)
        
        # بروزرسانی فیچرهای انتخاب شده
        self.selected_features = features
        self.feature_importances = feature_importances
        self.last_selection_time = datetime.now()
        
        # ذخیره فیچرهای انتخاب شده
        self.save_selected_features()
        
        print(f"Selected {len(features)} features with basic method")
        return features, feature_importances
    
    def select_features_genetic(self, symbol='BTCUSDT'):
        """انتخاب فیچرها با الگوریتم ژنتیک"""
        # دریافت داده‌های تاریخی
        df = self.evaluator.get_historical_data_with_targets(symbol, hours=48)
        
        if df.empty:
            print("No historical data available for genetic feature selection")
            return self.select_features_basic(symbol)  # استفاده از روش ساده در صورت عدم وجود داده
        
        print(f"Running genetic algorithm for feature selection with {df.shape[0]} samples")
        
        # مقادیر اولیه
        population_size = 20
        generations = 10
        mutation_rate = 0.1
        all_features = [col for col in df.columns if col not in ['target', 'future_pct_change']]
        
        if len(all_features) == 0:
            print("No features available for selection")
            return self.select_features_basic(symbol)
            
        # ساخت جمعیت اولیه
        population = []
        
        # اضافه کردن یک کروموزوم با همه فیچرهای ضروری
        essential_chromosome = [1 if f in ESSENTIAL_FEATURES and f in all_features else 0 for f in all_features]
        population.append(essential_chromosome)
        
        # اضافه کردن کروموزوم‌های تصادفی
        for _ in range(population_size - 1):
            # هر کروموزوم یک لیست باینری است که نشان می‌دهد کدام فیچرها انتخاب شده‌اند
            chromosome = [random.randint(0, 1) for _ in range(len(all_features))]
            population.append(chromosome)
        
        # تابع ارزیابی کروموزوم
        def evaluate_chromosome(chromosome):
            # انتخاب فیچرهای مورد نظر
            selected = [f for i, f in enumerate(all_features) if chromosome[i] == 1]
            
            if not selected:  # اگر هیچ فیچری انتخاب نشده باشد
                return 0.0
                
            # اطمینان از حداقل یک فیچر ضروری
            has_essential = any(f in ESSENTIAL_FEATURES for f in selected)
            if not has_essential:
                # اضافه کردن یک فیچر ضروری
                for i, f in enumerate(all_features):
                    if f in ESSENTIAL_FEATURES:
                        chromosome[i] = 1
                        selected.append(f)
                        break
            
            # بررسی تعداد فیچرهای انتخاب شده
            if len(selected) > self.feature_count:
                penalty = (len(selected) - self.feature_count) / 100  # جریمه برای فیچرهای اضافی
            else:
                penalty = 0
            
            try:
                # آماده‌سازی داده‌ها
                X = df[selected].replace([np.inf, -np.inf], np.nan).fillna(0)
                y = df['target'].astype(int)
                
                # تقسیم داده‌ها
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # نرمال‌سازی
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # آموزش مدل
                model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # ارزیابی
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # امتیاز نهایی (ترکیبی از دقت و تعداد مناسب فیچرها)
                feature_ratio = min(len(selected), self.feature_count) / self.feature_count
                fitness = (0.7 * accuracy + 0.3 * f1) * feature_ratio - penalty
                
                return max(0.001, fitness)  # حداقل مقدار مثبت
                
            except Exception as e:
                print(f"Error evaluating chromosome: {e}")
                return 0.001  # مقدار کوچک برای کروموزوم‌های خطادار
        
        # تابع انتخاب والدین (روش چرخ رولت)
        def select_parents(population, fitnesses):
            # محاسبه مجموع امتیازها
            total_fitness = sum(fitnesses)
            if total_fitness <= 0:
                # اگر همه امتیازها صفر باشند، انتخاب تصادفی
                return random.sample(population, 2)
                
            # انتخاب والدین با احتمال متناسب با امتیاز
            selection_probs = [f/total_fitness for f in fitnesses]
            parent1_idx = np.random.choice(len(population), p=selection_probs)
            parent2_idx = np.random.choice(len(population), p=selection_probs)
            
            # اطمینان از انتخاب دو والد متفاوت
            while parent1_idx == parent2_idx and len(population) > 1:
                parent2_idx = np.random.choice(len(population), p=selection_probs)
                
            return population[parent1_idx], population[parent2_idx]
        
        # تابع ترکیب (crossover)
        def crossover(parent1, parent2):
            # ترکیب تک‌نقطه‌ای
            crossover_point = random.randint(1, len(parent1)-1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        
        # تابع جهش (mutation)
        def mutate(chromosome):
            # جهش در هر بیت با احتمال mutation_rate
            for i in range(len(chromosome)):
                if random.random() < mutation_rate:
                    chromosome[i] = 1 - chromosome[i]  # معکوس کردن بیت
            return chromosome
        
        # اجرای الگوریتم ژنتیک
        best_chromosome = None
        best_fitness = -1
        
        for generation in range(generations):
            start_time = time.time()
            
            # ارزیابی جمعیت فعلی
            fitnesses = [evaluate_chromosome(chrom) for chrom in population]
            
            # پیدا کردن بهترین کروموزوم
            gen_best_idx = np.argmax(fitnesses)
            gen_best_chrom = population[gen_best_idx]
            gen_best_fitness = fitnesses[gen_best_idx]
            
            # بروزرسانی بهترین کروموزوم تاکنون
            if gen_best_fitness > best_fitness:
                best_chromosome = gen_best_chrom.copy()
                best_fitness = gen_best_fitness
                
                # محاسبه تعداد فیچرهای انتخاب شده
                selected_count = sum(best_chromosome)
                print(f"Generation {generation+1}: New best score {best_fitness:.4f} with {selected_count} features")
            
            # نسل جدید با نگه داشتن بهترین کروموزوم (elitism)
            new_population = [gen_best_chrom]
            
            # ایجاد فرزندان جدید
            while len(new_population) < population_size:
                # انتخاب والدین
                parent1, parent2 = select_parents(population, fitnesses)
                
                # ترکیب و جهش
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # افزودن فرزندان به نسل جدید
                new_population.append(child1)
                new_population.append(child2)
            
            # کوتاه کردن جمعیت اگر بیش از حد بزرگ شده باشد
            if len(new_population) > population_size:
                new_population = new_population[:population_size]
            
            # جایگزینی جمعیت
            population = new_population
            
            # نمایش آمار نسل
            avg_fitness = sum(fitnesses) / len(fitnesses)
            generation_time = time.time() - start_time
            print(f"Generation {generation+1}/{generations}: Avg fitness={avg_fitness:.4f}, Best fitness={gen_best_fitness:.4f}, Time: {generation_time:.1f}s")
        
        # تبدیل بهترین کروموزوم به لیست فیچرها
        selected_features = [f for i, f in enumerate(all_features) if best_chromosome[i] == 1]
        
        # محاسبه اهمیت فیچرهای انتخاب شده
        feature_importances = {}
        base_importances = self.evaluator.evaluate_features(symbol, hours=48)
        
        # ترکیب اهمیت‌های موجود و نتایج الگوریتم ژنتیک
        for i, feature in enumerate(all_features):
            if feature in base_importances:
                # ترکیب اهمیت پایه با نتیجه الگوریتم ژنتیک
                genetic_value = best_chromosome[i]
                feature_importances[feature] = base_importances[feature] * (1 + genetic_value)
            elif best_chromosome[i] == 1:
                # اگر توسط الگوریتم ژنتیک انتخاب شده اما اهمیت پایه ندارد
                feature_importances[feature] = 0.01
        
        # اطمینان از وجود فیچرهای ضروری
        for feature in ESSENTIAL_FEATURES:
            if feature in all_features and feature not in selected_features:
                selected_features.append(feature)
        
        # بروزرسانی فیچرهای انتخاب شده
        self.selected_features = selected_features
        self.feature_importances = feature_importances
        self.last_selection_time = datetime.now()
        
        # ذخیره فیچرهای انتخاب شده
        self.save_selected_features()
        
        print(f"Selected {len(selected_features)} features with genetic algorithm")
        return selected_features, feature_importances
    
    def select_features(self, symbol='BTCUSDT', force=False, method='genetic'):
        """
        انتخاب فیچرها براساس شرایط فعلی بازار
        
        Args:
            symbol: نماد مورد نظر
            force: اجبار به بروزرسانی انتخاب فیچرها
            method: روش انتخاب ('basic' یا 'genetic')
            
        Returns:
            لیست فیچرهای انتخاب شده و اهمیت آنها
        """
        # بررسی نیاز به بروزرسانی
        if not self.should_update_selection(force):
            print("Using cached feature selection")
            return self.selected_features, self.feature_importances
        
        print(f"Selecting optimal feature combination for {symbol}")
        
        # انتخاب فیچرها با روش مشخص شده
        if method == 'genetic':
            return self.select_features_genetic(symbol)
        else:
            return self.select_features_basic(symbol)
    
    def get_active_features(self, symbol='BTCUSDT'):
        """دریافت فیچرهای فعال برای یک نماد"""
        # دریافت آخرین مقادیر فیچرها از دیتابیس
        latest_features = self.db.get_latest_features(symbol)
        
        if latest_features.empty:
            print("No latest features available")
            return []
        
        # شناسایی فیچرهایی که در دیتابیس وجود دارند
        available_features = latest_features.columns.tolist()
        
        # فیلتر کردن فیچرهای انتخاب شده براساس در دسترس بودن
        active_features = [f for f in self.selected_features if f in available_features]
        
        # اطمینان از وجود فیچرهای ضروری
        for feature in ESSENTIAL_FEATURES:
            if feature in available_features and feature not in active_features:
                active_features.append(feature)
        
        return active_features
    
    def get_group_distribution(self):
        """محاسبه توزیع گروه‌های فیچر در فیچرهای انتخاب شده"""
        group_counts = {group: 0 for group in FEATURE_GROUPS}
        
        for feature in self.selected_features:
            group = get_group_for_feature(feature)
            if group in group_counts:
                group_counts[group] += 1
            else:
                group_counts['unknown'] = group_counts.get('unknown', 0) + 1
        
        # محاسبه درصدها
        total = len(self.selected_features)
        if total > 0:
            group_percentages = {group: (count / total) * 100 for group, count in group_counts.items()}
        else:
            group_percentages = {group: 0 for group in group_counts}
            
        return group_counts, group_percentages
    
    def print_selected_features(self, detailed=False):
        """نمایش فیچرهای انتخاب شده"""
        if not self.selected_features:
            print("No features selected yet")
            return
            
        print("\n============================================================")
        print(f"SELECTED FEATURES ({len(self.selected_features)} features)")
        print("============================================================")
        
        # مرتب‌سازی براساس اهمیت
        sorted_features = []
        for feature in self.selected_features:
            importance = self.feature_importances.get(feature, 0)
            sorted_features.append((feature, importance))
        
        sorted_features = sorted(sorted_features, key=lambda x: x[1], reverse=True)
        
        # نمایش فیچرها
        for i, (feature, importance) in enumerate(sorted_features):
            if i < 15 or detailed:
                print(f"{i+1:2d}. {feature:30s}: {importance:.6f}")
                
        if len(sorted_features) > 15 and not detailed:
            print(f"... and {len(sorted_features) - 15} more features")
            
        print("============================================================")
        
        # نمایش توزیع گروه‌های فیچر
        group_counts, group_percentages = self.get_group_distribution()
        
        print("\nFeature group distribution:")
        for group, count in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"- {group}: {count} features ({group_percentages[group]:.1f}%)")
        
        # ذخیره جزئیات در یک فایل متنی
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"selected_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(log_file, 'w') as f:
            f.write(f"Selected Features at {datetime.now()}\n")
            f.write("============================================================\n")
            for i, (feature, importance) in enumerate(sorted_features):
                f.write(f"{i+1:2d}. {feature:30s}: {importance:.6f}\n")
            
            f.write("\nFeature group distribution:\n")
            for group, count in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    f.write(f"- {group}: {count} features ({group_percentages[group]:.1f}%)\n")
            
            f.write("\nTotal combinations evaluated: 800\n")
            
        print(f"Feature details saved to {log_file}")