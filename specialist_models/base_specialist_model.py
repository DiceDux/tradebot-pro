"""
کلاس پایه برای مدل‌های متخصص
"""
import pickle
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# تنظیم لاگر
logger = logging.getLogger("specialist_models")
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/models.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class BaseSpecialistModel:
    """کلاس پایه برای مدل‌های متخصص"""
    
    def __init__(self, model_name='base', feature_group=''):
        """مقداردهی اولیه"""
        self.model_name = model_name
        self.feature_group = feature_group
        self.model = None
        self.scaler = None
        self.training_history = []
        self.model_file = f"model/specialists/{model_name}.pkl"
        
    def create_model(self):
        """
        ایجاد مدل پیشرفته با استفاده از XGBoost
        """
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )
        return model
        
    def get_required_features(self):
        """دریافت لیست فیچرهای مورد نیاز این مدل"""
        from feature_selection.feature_groups import FEATURE_GROUPS
        return FEATURE_GROUPS.get(self.feature_group, [])
        
    def preprocess_features(self, X):
        """
        پیش‌پردازش فیچرها قبل از آموزش یا پیش‌بینی
        - مدیریت مقادیر گمشده
        - نرمال‌سازی داده‌ها
        """
        # کپی از داده‌ها
        X_processed = X.copy()
        
        # مدیریت مقادیر گمشده
        X_processed = X_processed.fillna(0)
        
        # نرمال‌سازی داده‌ها
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
            
        return X_scaled
        
    def train(self, X, y):
        """
        آموزش مدل متخصص
        
        Args:
            X: فیچرهای ورودی (DataFrame)
            y: برچسب‌های هدف (0: فروش، 1: نگهداری، 2: خرید)
        """
        start_time = datetime.now()
        
        # بررسی وجود داده کافی
        if len(X) < 10:
            logger.warning(f"Not enough data for training {self.model_name} (samples: {len(X)})")
            print(f"Warning: Not enough data for training {self.model_name} model (only {len(X)} samples)")
            return False
            
        logger.info(f"Training {self.model_name} specialist model with {len(X.columns)} features and {len(X)} samples")
        print(f"Training {self.model_name} specialist model with {len(X.columns)} features and {len(X)} samples")
        
        # پیش‌پردازش فیچرها
        X_scaled = self.preprocess_features(X)
        
        # ایجاد مدل
        if self.model is None:
            self.model = self.create_model()
            
        # آموزش مدل
        try:
            self.model.fit(X_scaled, y)
            
            # ارزیابی روی داده‌های آموزش
            train_accuracy = self.model.score(X_scaled, y)
            
            # محاسبه F1-score
            from sklearn.metrics import f1_score
            y_pred = self.model.predict(X_scaled)
            f1 = f1_score(y, y_pred, average='macro')
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # ذخیره تاریخچه آموزش
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples': len(X),
                'features': len(X.columns),
                'accuracy': train_accuracy,
                'f1_score': f1,
                'training_time': training_time
            })
            
            logger.info(f"{self.model_name} model trained with accuracy: {train_accuracy:.4f}, F1: {f1:.4f}")
            print(f"{self.model_name} model trained with accuracy: {train_accuracy:.4f}, F1: {f1:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error training {self.model_name} model: {e}")
            print(f"Error training {self.model_name} model: {e}")
            return False
            
    def predict(self, X):
        """
        پیش‌بینی با استفاده از مدل آموزش دیده
        
        Args:
            X: فیچرهای ورودی (DataFrame)
            
        Returns:
            (predictions, probabilities): پیش‌بینی‌ها و احتمالات کلاس‌ها
        """
        if self.model is None:
            logger.warning(f"{self.model_name} model not trained yet")
            raise ValueError(f"{self.model_name} model not trained yet")
            
        # پیش‌پردازش فیچرها
        X_scaled = self.preprocess_features(X)
        
        # پیش‌بینی
        try:
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error predicting with {self.model_name} model: {e}")
            raise
            
    def save(self):
        """ذخیره مدل در فایل"""
        if self.model is None:
            logger.warning(f"Cannot save {self.model_name} model: model not trained")
            return False
            
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            # ذخیره مدل و scaler
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'training_history': self.training_history,
                    'last_updated': datetime.now().isoformat()
                }, f)
                
            logger.info(f"{self.model_name} model saved to {self.model_file}")
            print(f"{self.model_name} model saved to {self.model_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.model_name} model: {e}")
            print(f"Error saving {self.model_name} model: {e}")
            return False
            
    def load(self):
        """بارگیری مدل از فایل"""
        if not os.path.exists(self.model_file):
            logger.warning(f"{self.model_name} model file not found: {self.model_file}")
            return self
            
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data.get('scaler')  # ممکن است در مدل‌های قدیمی‌تر موجود نباشد
                self.training_history = data.get('training_history', [])
                
            logger.info(f"{self.model_name} model loaded from {self.model_file}")
            return self
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {e}")
            return self
