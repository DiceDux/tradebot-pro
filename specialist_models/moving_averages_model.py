"""
مدل متخصص میانگین‌های متحرک
"""
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("moving_averages_model")

class MovingAveragesModel:
    """مدل متخصص برای تحلیل میانگین‌های متحرک"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.model = None
        self.model_name = "moving_averages"
        self.feature_group = "moving_averages"
        self.model_file = f"model/specialists/{self.model_name}.pkl"
        
    def create_model(self):
        """ایجاد مدل با پارامترهای بهینه‌تر"""
        model = RandomForestClassifier(
            n_estimators=500,  # افزایش تعداد درخت‌ها
            max_depth=15,      # افزایش عمق درخت‌ها
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',  # انتخاب ویژگی‌ها با روش sqrt
            bootstrap=True,
            class_weight='balanced',
            criterion='gini',
            n_jobs=-1,
            random_state=42
        )
        return model
        
    def train(self, X, y):
        """
        آموزش مدل با استفاده از داده‌های تاریخی
        
        Args:
            X: فیچرهای ورودی
            y: برچسب‌های هدف (0=فروش، 1=هولد، 2=خرید)
        """
        print(f"Training {self.model_name} specialist model with {X.shape[1]} features and {len(X)} samples")
        
        if self.model is None:
            self.model = self.create_model()
            
        try:
            # آموزش مدل
            self.model.fit(X, y)
            
            # ارزیابی روی داده‌های آموزش
            train_accuracy = self.model.score(X, y)
            
            # محاسبه F1-score
            from sklearn.metrics import f1_score, classification_report
            y_pred = self.model.predict(X)
            f1 = f1_score(y, y_pred, average='macro')
            
            print(f"{self.model_name} model trained with accuracy: {train_accuracy:.4f}, F1: {f1:.4f}")
            
            # گزارش طبقه‌بندی
            print("Classification Report:")
            print(classification_report(y, y_pred))
            
            # اهمیت فیچرها
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                print("Top 10 most important features:")
                feature_names = X.columns.tolist()
                for i in range(min(10, len(feature_names))):
                    idx = indices[i]
                    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {self.model_name} model: {e}", exc_info=True)
            raise
            
    def predict(self, X):
        """
        پیش‌بینی با استفاده از مدل آموزش‌دیده
        
        Args:
            X: فیچرهای ورودی
            
        Returns:
            (predictions, probabilities): پیش‌بینی‌ها و احتمالات
        """
        if self.model is None:
            raise ValueError(f"{self.model_name} model not trained yet")
            
        # پیش‌بینی
        predictions = self.model.predict(X)
        
        # احتمالات کلاس‌ها
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
        
    def save(self):
        """ذخیره مدل در فایل"""
        if self.model is None:
            logger.warning(f"Cannot save {self.model_name} model: model not trained")
            return False
            
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
                
            print(f"{self.model_name} model saved to {self.model_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.model_name} model: {e}")
            return False
            
    def load(self):
        """بارگیری مدل از فایل"""
        if not os.path.exists(self.model_file):
            logger.warning(f"{self.model_name} model file not found: {self.model_file}")
            return self
            
        try:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
                
            print(f"{self.model_name} model loaded from {self.model_file}")
            return self
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {e}")
            return self
    
    def get_required_features(self):
        """لیست فیچرهای مورد نیاز این مدل"""
        return [
            'ema5', 'ema9', 'ema10', 'ema20', 'ema21', 'ema50', 'ema100', 'ema200',
            'sma20', 'sma50', 'sma100', 'sma200',
            'price_to_ema50', 'price_to_ema200', 'price_to_sma50', 'price_to_sma200',
            'ema_cross_5_20', 'ema_cross_9_21', 'ema_cross_50_200',
            'sma_cross_20_50', 'sma_cross_50_200'
        ]
