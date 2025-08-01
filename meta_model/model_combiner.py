"""
ترکیب‌کننده مدل‌های متخصص و ایجاد مدل متا
"""
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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
        self.model_weights = None  # وزن‌های مدل‌های متخصص
        self.training_history = []  # تاریخچه آموزش
        self.model_file = f"model/{model_name}.pkl"
        
    def create_models(self):
        """ایجاد چندین مدل متا با پیکربندی‌های مختلف"""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=1000,
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
        
        # اطمینان از استفاده از ویژگی‌های صحیح
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.metrics import f1_score, classification_report, confusion_matrix
            from specialist_models.feature_harmonizer import FeatureHarmonizer
            
            # ایجاد مدل‌های متا
            self.ensemble_models = self.create_models()
            
            print("Evaluating meta models with cross-validation...")
            
            # اعتبارسنجی هر مدل با اعتبارسنجی متقابل
            cv_results = {}
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # آموزش و ارزیابی هر مدل
            for name, model in self.ensemble_models.items():
                try:
                    # اعتبارسنجی متقابل برای دقت
                    cv_accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
                    # اعتبارسنجی متقابل برای F1
                    cv_f1 = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
                    
                    cv_results[name] = {
                        'accuracy': cv_accuracy.mean(),
                        'f1': cv_f1.mean()
                    }
                    
                    print(f"Model {name}: CV accuracy = {cv_accuracy.mean():.4f}, CV F1 = {cv_f1.mean():.4f}")
                except Exception as e:
                    print(f"Error evaluating model {name}: {e}")
                    cv_results[name] = {'f1': 0.0}
            
            # انتخاب بهترین مدل (بر اساس F1)
            best_f1 = 0
            best_accuracy = 0
            self.best_model_name = None
            
            for name, metrics in cv_results.items():
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_accuracy = metrics['accuracy']
                    self.best_model_name = name
            
            # اگر هیچ مدلی موفق نبود، از gradient_boosting استفاده کنیم
            if self.best_model_name is None:
                self.best_model_name = "gradient_boosting"
                print("Warning: No successful model found, falling back to gradient_boosting")
            else:
                print(f"Best model: {self.best_model_name} with F1: {best_f1:.4f}, accuracy: {best_accuracy:.4f}")
            
            # آموزش بهترین مدل روی کل داده‌ها
            self.model = self.ensemble_models[self.best_model_name]
            self.model.fit(X, y)
            
            # ارزیابی روی داده‌های آموزش
            y_pred = self.model.predict(X)
            train_accuracy = accuracy_score(y, y_pred)
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
            if len(self.specialist_models) > 0:
                self._calculate_model_weights()
                
                # نمایش وزن‌های مدل‌های متخصص
                print("Specialist model weights:")
                for i, model in enumerate(self.specialist_models):
                    print(f"- {model.model_name}: {self.model_weights[i]:.4f}")
                    
            return True
            
        except Exception as e:
            print(f"Error training meta model: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _calculate_model_weights(self):
        """محاسبه اهمیت مدل‌های متخصص"""
        n_models = len(self.specialist_models)
        
        if hasattr(self.model, 'feature_importances_'):
            # برای RandomForest و GradientBoosting
            importances = self.model.feature_importances_
            
            self.model_weights = np.zeros(n_models)
            class_features_per_model = 3  # تعداد کلاس‌ها
            
            if len(importances) == n_models * class_features_per_model:
                for i in range(n_models):
                    start_idx = i * class_features_per_model
                    end_idx = (i + 1) * class_features_per_model
                    self.model_weights[i] = np.mean(importances[start_idx:end_idx])
                    
                # نرمال‌سازی وزن‌ها
                total_weight = np.sum(self.model_weights)
                if total_weight > 0:
                    self.model_weights = self.model_weights / total_weight
                else:
                    self.model_weights = np.ones(n_models) / n_models
            else:
                # اگر تعداد ویژگی‌ها متفاوت بود، وزن یکسان اختصاص می‌دهیم
                print(f"Warning: Feature count mismatch. Expected {n_models * class_features_per_model}, got {len(importances)}")
                self.model_weights = np.ones(n_models) / n_models
        else:
            # برای مدل‌های دیگر، وزن یکسان
            self.model_weights = np.ones(n_models) / n_models
    
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
            # اطمینان از تطابق ویژگی‌ها
            from specialist_models.feature_harmonizer import FeatureHarmonizer
            
            if hasattr(self.model, 'feature_names_in_'):
                # بررسی و اصلاح ویژگی‌های ورودی
                X_compatible = FeatureHarmonizer.ensure_feature_compatibility(self.model, X)
            else:
                X_compatible = X
                
            # پیش‌بینی با مدل متا
            predictions = self.model.predict(X_compatible)
            probabilities = self.model.predict_proba(X_compatible)
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error predicting with meta model: {e}")
            import traceback
            traceback.print_exc()
            
            # در صورت خطا، مقدار پیش‌فرض برگردان
            dummy_pred = np.array([1])  # پیش‌فرض: هولد
            dummy_proba = np.array([[0.2, 0.6, 0.2]])  # احتمالات پیش‌فرض
            return dummy_pred, dummy_proba
    
    def save(self):
        """ذخیره مدل متا در فایل"""
        if self.model is None:
            logger.warning("Cannot save meta model: model not trained")
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
                
            logger.info(f"Meta model saved to {self.model_file}")
            print(f"Meta model saved to {self.model_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving meta model: {e}")
            print(f"Error saving meta model: {e}")
            return False
    
    def load(self):
        """بارگیری مدل متا از فایل"""
        if not os.path.exists(self.model_file):
            logger.warning(f"Meta model file not found: {self.model_file}")
            print(f"Meta model file not found: {self.model_file}")
            return self
            
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.best_model_name = data.get('best_model_name')
                self.model_weights = data.get('model_weights')
                self.training_history = data.get('training_history', [])
                
            logger.info(f"Meta model loaded from {self.model_file} (type: {self.best_model_name})")
            print(f"Meta model loaded from {self.model_file}")
            return self
        except Exception as e:
            logger.error(f"Error loading meta model: {e}")
            print(f"Error loading meta model: {e}")
            return self
