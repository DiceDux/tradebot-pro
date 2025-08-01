import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("model_combiner")

class ModelCombiner:
    """کلاس برای ترکیب نتایج مدل‌های متخصص"""
    
    def __init__(self, specialist_models=None, model_name="meta_model"):
        """مقداردهی اولیه"""
        self.specialist_models = specialist_models if specialist_models else []
        self.model = None
        self.model_weights = None
        self.training_history = []
        self.model_file = f"model/{model_name}.pkl"
        
    def create_model(self):
        """ایجاد مدل متا"""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        return model
        
    def train(self, X, y):
        """
        آموزش مدل متا با استفاده از پیش‌بینی‌های مدل‌های متخصص
        
        Args:
            X: فیچرهای پیش‌بینی شده از مدل‌های متخصص
            y: برچسب‌های هدف
        """
        print(f"Training meta model with {len(X.columns)} features and {len(X)} samples")
        
        if self.model is None:
            self.model = self.create_model()
            
        try:
            # آموزش مدل
            self.model.fit(X, y)
            
            # ارزیابی روی داده‌های آموزش
            train_accuracy = self.model.score(X, y)
            
            # محاسبه F1-score
            from sklearn.metrics import f1_score
            y_pred = self.model.predict(X)
            f1 = f1_score(y, y_pred, average='macro')
            
            print(f"Meta model trained with accuracy: {train_accuracy:.4f}, F1: {f1:.4f}")
            
            # ذخیره تاریخچه آموزش
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples': len(X),
                'accuracy': train_accuracy,
                'f1_score': f1,
            })
            
            # محاسبه اهمیت مدل‌های متخصص
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                model_importance = {}
                
                # محاسبه میانگین اهمیت برای هر مدل متخصص
                n_classes = 3  # تعداد کلاس‌ها
                n_models = len(self.specialist_models)
                
                if n_models > 0:
                    self.model_weights = np.zeros(n_models)
                    class_features_per_model = n_classes  # هر مدل 3 ویژگی تولید می‌کند (برای کلاس 0، 1 و 2)
                    
                    # برای هر مدل، میانگین اهمیت فیچرهایش را محاسبه می‌کنیم
                    for i in range(n_models):
                        start_idx = i * class_features_per_model
                        end_idx = (i + 1) * class_features_per_model
                        if end_idx <= len(importances):
                            self.model_weights[i] = np.mean(importances[start_idx:end_idx])
                    
                    # نرمال‌سازی وزن‌ها
                    if np.sum(self.model_weights) > 0:
                        self.model_weights = self.model_weights / np.sum(self.model_weights)
                    else:
                        self.model_weights = np.ones(n_models) / n_models
                else:
                    self.model_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])  # وزن یکسان برای 5 مدل
                
                # نمایش وزن‌های مدل‌های متخصص
                print("Specialist model weights:")
                for i, model in enumerate(self.specialist_models):
                    # استفاده از model_name به جای name
                    print(f"- {model.model_name}: {self.model_weights[i]:.4f}")
                    
            return True
        except Exception as e:
            print(f"Error training meta model: {e}")
            return False
    
    def predict(self, X):
        """
        پیش‌بینی با استفاده از مدل متا
        
        Args:
            X: فیچرهای ورودی
            
        Returns:
            (predictions, probabilities): پیش‌بینی‌ها و احتمالات
        """
        if self.model is None:
            raise ValueError("Meta model not trained yet")
            
        try:
            # پیش‌بینی با مدل متا
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            return predictions, probabilities
        except Exception as e:
            print(f"Error predicting with meta model: {e}")
            raise
    
    def save(self):
        """ذخیره مدل متا در فایل"""
        if self.model is None:
            print("Cannot save meta model: model not trained")
            return False
            
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            # ذخیره مدل و اطلاعات آن
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_weights': self.model_weights,
                    'training_history': self.training_history,
                    'last_updated': datetime.now().isoformat()
                }, f)
                
            print(f"Meta model saved to {self.model_file}")
            return True
        except Exception as e:
            print(f"Error saving meta model: {e}")
            return False
    
    def load(self):
        """بارگیری مدل متا از فایل"""
        if not os.path.exists(self.model_file):
            print(f"Meta model file not found: {self.model_file}")
            return self
            
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.model_weights = data.get('model_weights')
                self.training_history = data.get('training_history', [])
                
            print(f"Meta model loaded from {self.model_file}")
            return self
        except Exception as e:
            print(f"Error loading meta model: {e}")
            return self
