import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

MODEL_PATH = "model/enhanced_base_model.pkl"

class EnhancedBaseModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X, y, class_balance=True):
        """
        آموزش مدل پایه با تمام فیچرها و متعادل‌سازی کلاس‌ها
        """
        print(f"Training enhanced base model with {X.shape[1]} features")
        
        # متعادل‌سازی کلاس‌ها
        if class_balance:
            print("Performing class balancing...")
            try:
                # روش اول: SMOTE
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                print(f"Class distribution after SMOTE: {np.bincount(y_balanced)}")
            except Exception as e:
                # روش دوم: Resampling
                print(f"SMOTE failed ({str(e)}), using resampling instead")
                X_balanced = pd.DataFrame()
                y_balanced = []
                
                # تعداد نمونه‌های هر کلاس
                class_counts = np.bincount(y)
                target_count = max(class_counts) * 2  # تعداد مطلوب برای هر کلاس
                
                for cls in range(len(class_counts)):
                    cls_indices = np.where(y == cls)[0]
                    if len(cls_indices) == 0:
                        continue
                        
                    X_cls = X.iloc[cls_indices]
                    y_cls = y[cls_indices]
                    
                    # اگر کلاس کمتر از تعداد هدف نمونه دارد
                    if len(X_cls) < target_count:
                        X_resampled = resample(X_cls, 
                                               replace=True, 
                                               n_samples=target_count, 
                                               random_state=42)
                        y_resampled = np.array([cls] * target_count)
                    else:
                        X_resampled = X_cls
                        y_resampled = y_cls
                        
                    X_balanced = pd.concat([X_balanced, X_resampled])
                    y_balanced.extend(y_resampled)
                
                y_balanced = np.array(y_balanced)
                print(f"Class distribution after resampling: {np.bincount(y_balanced)}")
        else:
            X_balanced = X
            y_balanced = y
            
        # تنظیم وزن کلاس‌ها
        class_weights = {}
        cls_counts = np.bincount(y)
        for cls in range(len(cls_counts)):
            if cls_counts[cls] > 0:
                # هر چه فراوانی کمتر، وزن بیشتر
                class_weights[cls] = 1.0 / cls_counts[cls]
        
        # نرمالایز کردن وزن‌ها
        sum_weights = sum(class_weights.values())
        for cls in class_weights:
            class_weights[cls] = class_weights[cls] / sum_weights * len(class_weights)
            
        print(f"Class weights: {class_weights}")
        
        # مدل پیشرفته CatBoost
        self.model = CatBoostClassifier(
            iterations=500,  # تعداد درخت‌ها
            learning_rate=0.03,
            depth=8,  # عمق درخت
            l2_leaf_reg=3,  # تنظیم‌کننده L2
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_strength=0.1,  # قدرت تصادفی
            od_type="Iter",  # نوع تشخیص over-fitting
            od_wait=50,  # تعداد تکرارهای انتظار برای early stopping
            verbose=50,  # نمایش وضعیت هر 50 تکرار
            class_weights=class_weights
        )
        
        self.model.fit(X_balanced, y_balanced)
        self.feature_names = list(X.columns)
        
        # ذخیره مدل و نام فیچرها
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.feature_names, MODEL_PATH.replace(".pkl", "_features.pkl"))
        
        print(f"Enhanced base model trained with {len(self.feature_names)} features and saved to {MODEL_PATH}")
        return self
    
    def load(self):
        """
        بارگذاری مدل از دیسک
        """
        try:
            self.model = joblib.load(MODEL_PATH)
            self.feature_names = joblib.load(MODEL_PATH.replace(".pkl", "_features.pkl"))
            print(f"Enhanced base model loaded from {MODEL_PATH} with {len(self.feature_names)} features")
            return self
        except FileNotFoundError:
            print(f"Enhanced base model not found at {MODEL_PATH}")
            return None
    
    def predict(self, X):
        """
        پیش‌بینی با استفاده از مدل پایه
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # اطمینان از وجود تمام فیچرها
        X_valid = X.reindex(columns=self.feature_names, fill_value=0.0)
        
        # پیش‌بینی کلاس و احتمالات
        pred_class = self.model.predict(X_valid)
        pred_proba = self.model.predict_proba(X_valid)
        
        # محاسبه اطمینان (بالاترین احتمال)
        confidence = np.max(pred_proba, axis=1)
        
        return pred_class, pred_proba, confidence
