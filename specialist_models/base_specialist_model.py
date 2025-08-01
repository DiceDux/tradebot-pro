"""
کلاس پایه برای مدل‌های متخصص
"""
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class BaseSpecialistModel:
    def __init__(self, name, feature_group=None):
        """
        مدل متخصص پایه
        
        Args:
            name: نام مدل
            feature_group: گروه فیچر مربوطه
        """
        self.name = name
        self.feature_group = feature_group
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_date = None
        self.accuracy = None
        self.f1_score = None
        self.confusion_matrix = None
    
    def train(self, X, y):
        """
        آموزش مدل متخصص
        
        Args:
            X: داده‌های ورودی
            y: برچسب‌ها
        """
        print(f"Training {self.name} specialist model with {X.shape[1]} features and {X.shape[0]} samples")
        
        # حذف مقادیر NaN و inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # ذخیره نام فیچرها
        self.feature_names = list(X.columns)
        
        # نرمال‌سازی داده‌ها
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # آموزش مدل
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # ارزیابی مدل
        y_pred = self.model.predict(X_scaled)
        self.accuracy = accuracy_score(y, y_pred)
        self.f1_score = f1_score(y, y_pred, average='weighted')
        self.confusion_matrix = confusion_matrix(y, y_pred)
        self.training_date = datetime.now()
        
        print(f"{self.name} model trained with accuracy: {self.accuracy:.4f}, F1: {self.f1_score:.4f}")
        
        return self
    
    def _create_model(self):
        """ایجاد مدل (باید در کلاس‌های فرزند پیاده‌سازی شود)"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def predict(self, X):
        """
        پیش‌بینی با مدل متخصص
        
        Args:
            X: داده‌های ورودی
            
        Returns:
            کلاس پیش‌بینی شده و احتمالات
        """
        if self.model is None:
            raise ValueError(f"{self.name} model has not been trained yet")
            
        if isinstance(X, pd.DataFrame):
            # اطمینان از وجود تمام فیچرهای مورد نیاز
            missing_features = [f for f in self.feature_names if f not in X.columns]
            extra_features = [f for f in X.columns if f not in self.feature_names]
            
            if missing_features:
                # ایجاد فیچرهای گمشده با مقدار صفر
                for f in missing_features:
                    X[f] = 0
                    
            # انتخاب فقط فیچرهای مورد نیاز
            X = X[self.feature_names]
        else:
            # تبدیل آرایه NumPy به DataFrame
            X = pd.DataFrame([X], columns=self.feature_names)
            
        # حذف مقادیر NaN و inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # نرمال‌سازی
        X_scaled = self.scaler.transform(X)
        
        # پیش‌بینی
        y_pred = self.model.predict(X_scaled)
        
        # احتمالات کلاس‌ها
        try:
            probas = self.model.predict_proba(X_scaled)
        except:
            # اگر مدل predict_proba را پشتیبانی نکند
            probas = np.zeros((len(y_pred), 3))
            probas[np.arange(len(y_pred)), y_pred] = 1
            
        return y_pred, probas
    
    def save(self, directory='model/specialists'):
        """ذخیره مدل متخصص"""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}.pkl")
        
        model_data = {
            'name': self.name,
            'feature_group': self.feature_group,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': self.training_date,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"{self.name} model saved to {file_path}")
        return file_path
    
    def load(self, directory='model/specialists'):
        """بارگذاری مدل متخصص"""
        file_path = os.path.join(directory, f"{self.name}.pkl")
        
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.name = model_data.get('name', self.name)
            self.feature_group = model_data.get('feature_group', self.feature_group)
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names')
            self.training_date = model_data.get('training_date')
            self.accuracy = model_data.get('accuracy')
            self.f1_score = model_data.get('f1_score')
            self.confusion_matrix = model_data.get('confusion_matrix')
            
            print(f"{self.name} model loaded from {file_path}")
            if self.training_date:
                print(f"Training date: {self.training_date}")
            if self.accuracy:
                print(f"Accuracy: {self.accuracy:.4f}, F1: {self.f1_score:.4f}")
                
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_required_features(self):
        """دریافت لیست فیچرهای مورد نیاز"""
        return self.feature_names if self.feature_names else []