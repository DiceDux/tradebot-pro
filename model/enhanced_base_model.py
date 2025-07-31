import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

MODEL_PATH = "model/enhanced_base_model.pkl"

class EnhancedBaseModel:
    def __init__(self):
        """مقداردهی اولیه مدل پیشرفته"""
        self.model = None
        self.feature_names = None
        
        # لیست پیش‌فرض فیچرها برای مواقعی که فیچرها در دسترس نیستند
        self.default_feature_names = [
            'close', 'open', 'high', 'low', 'volume', 
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema5', 'ema10', 'ema20', 'ema50', 'ema100', 'ema200',
            'sma5', 'sma10', 'sma20', 'sma50', 'sma100', 'sma200',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr', 'adx', 'cci', 'mfi', 'obv', 'stoch_k', 'stoch_d',
            'williams_r', 'close_pct_change', 'volume_pct_change',
            'sentiment_score', 'news_count', 'price_gap', 'volatility'
        ]
        
        try:
            self.load()
        except Exception as e:
            print(f"Warning: Could not load model ({str(e)}). Need to train it first.")
        
    def train(self, X, y, class_balance=True):
        """
        آموزش مدل پایه با تمام فیچرها و متعادل‌سازی کلاس‌ها
        
        Args:
            X: داده‌های ورودی (DataFrame)
            y: برچسب‌ها (آرایه)
            class_balance: آیا متعادل‌سازی کلاس‌ها انجام شود؟
        """
        print(f"Training enhanced base model with {X.shape[1]} features")
        
        # ذخیره نام فیچرها
        self.feature_names = list(X.columns)
        
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
        
        # ذخیره مدل
        self.save()
        
        return self
    
    def save(self):
        """ذخیره مدل و فیچرها به فایل"""
        if self.model is None:
            print("Warning: No model to save!")
            return False
            
        # اطمینان از وجود دایرکتوری
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
        # ذخیره مدل و فیچرها در یک فایل
        data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        joblib.dump(data, MODEL_PATH)
        print(f"Enhanced base model trained with {len(self.feature_names)} features and saved to {MODEL_PATH}")
        return True
    
    def get_feature_names(self):
        """برگرداندن نام ویژگی‌های مدل"""
        if self.feature_names is None or len(self.feature_names) == 0:
            print("Warning: feature_names not initialized, using default feature list")
            return self.default_feature_names
        return self.feature_names
    
    def get_feature_importance(self):
        """دریافت اهمیت ویژگی‌ها"""
        if self.model is None:
            print("Warning: Model not loaded or trained!")
            return np.ones(len(self.get_feature_names())) / len(self.get_feature_names())
            
        try:
            # تلاش برای دریافت اهمیت ویژگی‌ها از مدل
            importances = self.model.feature_importances_
            return importances
        except:
            # اگر مدل اهمیت ویژگی‌ها را پشتیبانی نکند، مقدار یکسان برمی‌گردانیم
            print("Warning: Could not get feature importances from model")
            return np.ones(len(self.get_feature_names())) / len(self.get_feature_names())

    def load(self):
        """بارگذاری مدل از فایل"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
            
        try:
            # تلاش برای بارگذاری در فرمت جدید (دیکشنری)
            data = joblib.load(MODEL_PATH)
            if isinstance(data, dict):
                self.model = data.get('model')
                self.feature_names = data.get('feature_names')
            else:
                # فرمت قدیمی (فقط مدل)
                self.model = data
                # تلاش برای بارگذاری فیچرها از فایل جداگانه
                feature_path = MODEL_PATH.replace(".pkl", "_features.pkl")
                if os.path.exists(feature_path):
                    self.feature_names = joblib.load(feature_path)
                else:
                    print(f"Warning: No feature names found, using default list")
                    self.feature_names = self.default_feature_names
            
            # بررسی نهایی فیچرها
            if self.feature_names is None or len(self.feature_names) == 0:
                print(f"Warning: Empty feature list loaded, using default list")
                self.feature_names = self.default_feature_names
                
            print(f"Enhanced base model loaded with {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X):
        """پیش‌بینی کلاس، احتمالات و اطمینان"""
        if self.model is None:
            raise ValueError("Model not loaded or trained!")
            
        if isinstance(X, pd.DataFrame):
            # اطمینان از اینکه همه فیچرهای مورد نیاز وجود دارند
            features_to_use = self.get_feature_names()
            missing_features = set(features_to_use) - set(X.columns)
            if missing_features:
                # اضافه کردن فیچرهای گم‌شده با مقدار 0
                for f in missing_features:
                    X[f] = 0
            
            # استفاده فقط از فیچرهای موجود در مدل
            available_features = [f for f in features_to_use if f in X.columns]
            X = X[available_features]
        
        # پیش‌بینی
        pred_proba = self.model.predict_proba(X)
        
        # اطمینان از اینکه خروجی‌ها اسکالر هستند (برای رفع هشدار NumPy)
        pred_class_idx = np.argmax(pred_proba, axis=1)
        pred_class = np.array([int(idx) for idx in pred_class_idx])  # تبدیل صحیح به اسکالر
        
        # محاسبه اطمینان (تفاوت بین بیشترین و دومین احتمال)
        confidences = []
        for probs in pred_proba:
            sorted_probs = np.sort(probs)
            # اگر فقط یک کلاس با احتمال بالا وجود داشته باشد، اطمینان بالاست
            confidence = sorted_probs[-1] - (sorted_probs[-2] if len(sorted_probs) > 1 else 0)
            confidences.append(float(confidence))  # تبدیل به اسکالر
        
        return pred_class, pred_proba, np.array(confidences)
