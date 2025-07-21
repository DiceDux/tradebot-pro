import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime

SELECTOR_PATH = "model/feature_selector.pkl"
ACTIVE_FEATURES_PATH = "model/active_features.pkl"

class AdaptiveFeatureSelector:
    """
    انتخاب‌کننده هوشمند فیچرها بر اساس شرایط بازار بدون نیاز به ریترین مدل
    """
    def __init__(self, base_model, all_feature_names, top_n=20, update_interval_hours=48):
        self.base_model = base_model
        self.all_feature_names = all_feature_names
        self.top_n = top_n
        self.update_interval_seconds = update_interval_hours * 3600
        self.last_update_time = None
        self.active_features = None
        
        # مسیر فایل ذخیره آخرین بروزرسانی
        self.last_update_file = "model/last_feature_update.txt"
        
        # بارگیری وضعیت قبلی اگر موجود باشد
        self._load_state()
        
    def _load_state(self):
        """بارگیری وضعیت قبلی انتخاب فیچر"""
        try:
            if os.path.exists(ACTIVE_FEATURES_PATH):
                self.active_features = joblib.load(ACTIVE_FEATURES_PATH)
                print(f"Loaded {len(self.active_features)} active features from disk")
            
            if os.path.exists(self.last_update_file):
                with open(self.last_update_file, 'r') as f:
                    last_update_str = f.read().strip()
                    self.last_update_time = float(last_update_str)
                    print(f"Last feature selection update: {datetime.fromtimestamp(self.last_update_time)}")
        except Exception as e:
            print(f"Error loading selector state: {e}")
    
    def _save_state(self):
        """ذخیره وضعیت فعلی انتخاب فیچر"""
        try:
            if self.active_features:
                joblib.dump(self.active_features, ACTIVE_FEATURES_PATH)
            
            if self.last_update_time:
                with open(self.last_update_file, 'w') as f:
                    f.write(str(self.last_update_time))
        except Exception as e:
            print(f"Error saving selector state: {e}")
    
    def _is_update_needed(self):
        """بررسی نیاز به بروزرسانی فیچرها"""
        if self.last_update_time is None:
            return True
        
        time_passed = time.time() - self.last_update_time
        return time_passed > self.update_interval_seconds
    
    def select_features(self, market_data):
        """
        انتخاب فیچرهای مناسب برای شرایط فعلی بازار
        
        Args:
            market_data: داده‌های اخیر بازار به صورت DataFrame
            
        Returns:
            list: فیچرهای انتخاب‌شده
        """
        # اگر هنوز زمان بروزرسانی نرسیده و فیچرهای فعال داریم
        if not self._is_update_needed() and self.active_features:
            print("Using cached feature selection")
            return self.active_features
        
        print("Selecting optimal features for current market conditions...")
        
        # روش‌های مختلف محاسبه اهمیت فیچرها
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
        essential_features = ['close', 'ema20', 'ema50']
        for feat in essential_features:
            if feat in self.all_feature_names and feat not in selected_features:
                selected_features.append(feat)
                if len(selected_features) > self.top_n:
                    selected_features.pop(0)
        
        print(f"Selected {len(selected_features)} features for current market conditions:")
        print(selected_features)
        
        # بروزرسانی وضعیت
        self.active_features = selected_features
        self.last_update_time = time.time()
        self._save_state()
        
        return selected_features
    
    def _get_model_importance(self):
        """دریافت اهمیت فیچرها از مدل پایه"""
        if not hasattr(self.base_model, 'get_feature_importance'):
            return {}
        
        try:
            importances = self.base_model.get_feature_importance()
            return {f: imp for f, imp in zip(self.all_feature_names, importances)}
        except:
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
        
        # برای هر فیچر، میانگین همبستگی با سایر فیچرها را محاسبه می‌کنیم
        # هر چه همبستگی کمتر باشد، اهمیت بیشتر است (فیچر منحصربه‌فردتر)
        for feat in common_features:
            # 1 - average correlation
            result[feat] = 1 - (corr_matrix[feat].sum() - 1) / (len(common_features) - 1)
            
        return result
