"""
ترکیب نتایج مدل‌های متخصص
"""
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

class ModelCombiner:
    def __init__(self, specialist_models=None):
        """
        مدل ترکیب‌کننده برای ادغام نتایج مدل‌های متخصص
        
        Args:
            specialist_models: لیست مدل‌های متخصص
        """
        self.specialist_models = specialist_models or []
        self.meta_model = None
        self.scaler = None
        self.feature_names = None
        self.class_weights = None
        self.model_weights = None
        self.training_date = None
        self.accuracy = None
        self.f1_score = None
    
    def add_specialist_model(self, model):
        """افزودن مدل متخصص"""
        self.specialist_models.append(model)
    
    def predict_with_specialists(self, X):
        """
        پیش‌بینی با تمام مدل‌های متخصص
        
        Args:
            X: داده‌های ورودی
            
        Returns:
            پیش‌بینی‌های مدل‌های متخصص و احتمالات
        """
        if not self.specialist_models:
            raise ValueError("No specialist models available")
            
        predictions = []
        probabilities = []
        
        for model in self.specialist_models:
            if model.model is None:
                print(f"Warning: {model.name} model not trained, skipping")
                predictions.append(None)
                probabilities.append(None)
                continue
                
            try:
                # انتخاب فیچرهای مورد نیاز برای این مدل
                required_features = model.get_required_features()
                X_model = {f: X[f] for f in required_features if f in X}
                
                # تبدیل به DataFrame
                if isinstance(X, dict):
                    X_model = pd.DataFrame([X_model])
                else:
                    X_model = X[required_features]
                    
                # پیش‌بینی
                pred, proba = model.predict(X_model)
                predictions.append(pred)
                probabilities.append(proba)
                
            except Exception as e:
                print(f"Error in {model.name} prediction: {e}")
                predictions.append(None)
                probabilities.append(None)
                
        return predictions, probabilities
    
    def combine_predictions(self, specialist_preds, specialist_probas):
        """
        ترکیب پیش‌بینی‌های مدل‌های متخصص
        
        Args:
            specialist_preds: پیش‌بینی‌های مدل‌های متخصص
            specialist_probas: احتمالات مدل‌های متخصص
            
        Returns:
            پیش‌بینی ترکیبی و احتمالات
        """
        # اگر مدل متا آموزش دیده باشد
        if self.meta_model is not None and self.model_weights is not None:
            # ایجاد ویژگی‌های ورودی برای مدل متا
            meta_features = []
            
            # استخراج احتمالات از هر مدل متخصص
            for i, probas in enumerate(specialist_probas):
                if probas is not None:
                    for j in range(probas.shape[1]):
                        meta_features.append(probas[0][j])  # احتمال هر کلاس
                else:
                    # اگر پیش‌بینی یک مدل وجود نداشته باشد، احتمال یکسان برای همه کلاس‌ها
                    meta_features.extend([0.33, 0.33, 0.34])
            
            # تبدیل به آرایه NumPy
            meta_features = np.array(meta_features).reshape(1, -1)
            
            # نرمال‌سازی
            if self.scaler:
                meta_features = self.scaler.transform(meta_features)
                
            # پیش‌بینی با مدل متا
            meta_pred = self.meta_model.predict(meta_features)
            meta_probas = self.meta_model.predict_proba(meta_features)
            
            return meta_pred[0], meta_probas[0]
            
        else:
            # استفاده از میانگین وزن‌دار
            weights = [1.0] * len(specialist_probas)
            if self.model_weights:
                weights = self.model_weights
                
            # ترکیب احتمالات
            combined_probas = np.zeros(3)  # برای 3 کلاس: Sell, Hold, Buy
            total_weight = 0.0
            
            for i, probas in enumerate(specialist_probas):
                if probas is not None:
                    combined_probas += probas[0] * weights[i]
                    total_weight += weights[i]
            
            # نرمال‌سازی
            if total_weight > 0:
                combined_probas /= total_weight
                
            # انتخاب کلاس با بیشترین احتمال
            pred_class = np.argmax(combined_probas)
            
            return pred_class, combined_probas
    
    def predict(self, X):
        """
        پیش‌بینی با مدل ترکیبی
        
        Args:
            X: داده‌های ورودی
            
        Returns:
            پیش‌بینی نهایی، احتمالات و اطمینان
        """
        # پیش‌بینی با مدل‌های متخصص
        specialist_preds, specialist_probas = self.predict_with_specialists(X)
        
        # ترکیب پیش‌بینی‌ها
        pred, probas = self.combine_predictions(specialist_preds, specialist_probas)
        
        # محاسبه اطمینان
        confidence = np.max(probas)
        
        return pred, probas, confidence
    
    def train(self, X_meta, y_meta):
        """
        آموزش مدل متا
        
        Args:
            X_meta: داده‌های ورودی (خروجی مدل‌های متخصص)
            y_meta: برچسب‌های هدف
        """
        print(f"Training meta model with {X_meta.shape[1]} features and {X_meta.shape[0]} samples")
        
        # حذف مقادیر NaN و inf
        X_meta = X_meta.replace([np.inf, -np.inf], np.nan)
        X_meta = X_meta.fillna(0)
        
        # ذخیره نام فیچرها
        self.feature_names = list(X_meta.columns)
        
        # نرمال‌سازی داده‌ها
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_meta)
        
        # محاسبه وزن کلاس‌ها
        class_counts = np.bincount(y_meta)
        total_samples = len(y_meta)
        self.class_weights = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in enumerate(class_counts)
        }
        
        # آموزش مدل
        self.meta_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight=self.class_weights,
            random_state=42
        )
        self.meta_model.fit(X_scaled, y_meta)
        
        # ارزیابی مدل
        y_pred = self.meta_model.predict(X_scaled)
        self.accuracy = accuracy_score(y_meta, y_pred)
        self.f1_score = f1_score(y_meta, y_pred, average='weighted')
        self.training_date = datetime.now()
        
        print(f"Meta model trained with accuracy: {self.accuracy:.4f}, F1: {self.f1_score:.4f}")
        
        # محاسبه وزن مدل‌های متخصص
        if hasattr(self.meta_model, 'feature_importances_'):
            # فرض می‌کنیم هر 3 فیچر متوالی مربوط به یک مدل است (3 کلاس)
            model_weights = []
            n_specialists = len(self.specialist_models)
            features_per_model = len(self.feature_names) // n_specialists
            
            for i in range(n_specialists):
                start_idx = i * features_per_model
                end_idx = (i + 1) * features_per_model
                model_weight = np.sum(self.meta_model.feature_importances_[start_idx:end_idx])
                model_weights.append(model_weight)
            
            # نرمال‌سازی وزن‌ها
            total_weight = sum(model_weights)
            if total_weight > 0:
                self.model_weights = [w / total_weight for w in model_weights]
            else:
                self.model_weights = [1.0 / n_specialists] * n_specialists
                
            # نمایش وزن مدل‌ها
            print("Specialist model weights:")
            for i, model in enumerate(self.specialist_models):
                print(f"- {model.name}: {self.model_weights[i]:.4f}")
        
        return self
    
    def save(self, file_path='model/meta_model.pkl'):
        """ذخیره مدل متا"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights,
            'model_weights': self.model_weights,
            'training_date': self.training_date,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'specialist_model_names': [model.name for model in self.specialist_models]
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Meta model saved to {file_path}")
        return file_path
    
    def load(self, file_path='model/meta_model.pkl'):
        """بارگذاری مدل متا"""
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.meta_model = model_data.get('meta_model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names')
            self.class_weights = model_data.get('class_weights')
            self.model_weights = model_data.get('model_weights')
            self.training_date = model_data.get('training_date')
            self.accuracy = model_data.get('accuracy')
            self.f1_score = model_data.get('f1_score')
            
            print(f"Meta model loaded from {file_path}")
            if self.training_date:
                print(f"Training date: {self.training_date}")
            if self.accuracy:
                print(f"Accuracy: {self.accuracy:.4f}, F1: {self.f1_score:.4f}")
                
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
