import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import logging

logger = logging.getLogger("model_combiner")

class ModelCombiner:
    """کلاس برای ترکیب نتایج مدل‌های متخصص"""
    
    def __init__(self, specialist_models=None, model_name="meta_model"):
        """مقداردهی اولیه"""
        self.specialist_models = specialist_models if specialist_models else []
        self.model = None
        self.ensemble_models = {}  # مدل‌های انسمبل مختلف
        self.best_model_name = None  # نام بهترین مدل
        self.model_weights = None
        self.training_history = []
        self.model_file = f"model/{model_name}.pkl"
        
    def create_models(self):
        """ایجاد چندین مدل متا با پیکربندی‌های مختلف"""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=1000,  # تعداد درخت‌های بسیار بیشتر
                max_depth=12,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                criterion='gini',
                n_jobs=-1,
                random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=30,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42
            )
        }
        
        # ایجاد مدل ترکیبی با رأی‌گیری وزن‌دار
        models["voting"] = VotingClassifier(
            estimators=[
                ('rf', models["random_forest"]),
                ('gb', models["gradient_boosting"])
            ],
            voting='soft',  # استفاده از احتمالات برای رأی‌گیری
            weights=[0.6, 0.4]  # وزن بیشتر به RandomForest
        )
        
        return models
        
    def train(self, X, y):
        """
        آموزش مدل متا با استفاده از پیش‌بینی‌های مدل‌های متخصص
        
        Args:
            X: فیچرهای پیش‌بینی شده از مدل‌های متخصص
            y: برچسب‌های هدف
        """
        print(f"Training meta model with {len(X.columns)} features and {len(X)} samples")
        
        # ایجاد مدل‌های متا
        self.ensemble_models = self.create_models()
        
        best_accuracy = 0
        best_f1 = 0
        
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.metrics import f1_score, classification_report, confusion_matrix
            
            print("Evaluating meta models with cross-validation...")
            
            # ارزیابی هر مدل با اعتبارسنجی متقابل
            cv_results = {}
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in self.ensemble_models.items():
                # اعتبارسنجی متقابل برای دقت
                cv_accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
                # اعتبارسنجی متقابل برای F1
                cv_f1 = cross_val_score(model, X, y, cv=kf, scoring='f1_macro', n_jobs=-1)
                
                cv_results[name] = {
                    'accuracy': cv_accuracy.mean(),
                    'f1': cv_f1.mean()
                }
                
                print(f"Model {name}: CV accuracy = {cv_accuracy.mean():.4f}, CV F1 = {cv_f1.mean():.4f}")
            
            # انتخاب بهترین مدل (بر اساس F1)
            for name, metrics in cv_results.items():
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_accuracy = metrics['accuracy']
                    self.best_model_name = name
            
            print(f"Best model: {self.best_model_name} with F1: {best_f1:.4f}, accuracy: {best_accuracy:.4f}")
            
            # آموزش بهترین مدل روی کل داده‌ها
            self.model = self.ensemble_models[self.best_model_name]
            self.model.fit(X, y)
            
            # ارزیابی روی داده‌های آموزش
            train_accuracy = self.model.score(X, y)
            y_pred = self.model.predict(X)
            f1 = f1_score(y, y_pred, average='macro')
            
            print(f"Meta model ({self.best_model_name}) trained with accuracy: {train_accuracy:.4f}, F1: {f1:.4f}")
            
            # گزارش طبقه‌بندی
            print("Classification Report:")
            print(classification_report(y, y_pred))
            
            # ماتریس آشفتگی
            cm = confusion_matrix(y, y_pred)
            print("Confusion Matrix:")
            print(cm)
            
            # ذخیره تاریخچه آموزش
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples': len(X),
                'accuracy': train_accuracy,
                'f1_score': f1,
                'best_model': self.best_model_name
            })
            
            # محاسبه اهمیت مدل‌های متخصص
            n_classes = 3  # تعداد کلاس‌ها
            n_models = len(self.specialist_models)
            
            if n_models > 0:
                if hasattr(self.model, 'feature_importances_'):
                    # برای RandomForest و GradientBoosting
                    importances = self.model.feature_importances_
                    
                    self.model_weights = np.zeros(n_models)
                    class_features_per_model = n_classes
                    
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
                    # برای مدل‌های دیگر، وزن یکسان
                    self.model_weights = np.ones(n_models) / n_models
                
                # نمایش وزن‌های مدل‌های متخصص
                print("Specialist model weights:")
                for i, model in enumerate(self.specialist_models):
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
                    'best_model_name': self.best_model_name,
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
                self.best_model_name = data.get('best_model_name')
                self.model_weights = data.get('model_weights')
                self.training_history = data.get('training_history', [])
                
            print(f"Meta model loaded from {self.model_file} (type: {self.best_model_name})")
            return self
        except Exception as e:
            print(f"Error loading meta model: {e}")
            return self
