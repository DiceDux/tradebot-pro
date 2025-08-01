"""
ارزیابی اهمیت فیچرها در زمان واقعی
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from feature_store.feature_database import FeatureDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

class FeatureEvaluator:
    def __init__(self, symbols=None):
        """
        ارزیابی اهمیت فیچرها
        
        Args:
            symbols: لیست نمادهای مورد نظر
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.db = FeatureDatabase()
        
    def create_target_variable(self, df):
        """ساخت متغیر هدف براساس تغییرات قیمت آینده"""
        if df.empty or 'close' not in df.columns:
            return None
            
        # محاسبه تغییر درصد قیمت برای 6 کندل بعدی
        df['future_pct_change'] = df['close'].shift(-6) / df['close'] - 1
        
        # طبقه‌بندی به سه گروه: فروش، نگهداری، خرید
        bins = [-np.inf, -0.005, 0.005, np.inf]  # آستانه‌های تغییر قیمت
        labels = [0, 1, 2]  # فروش، نگهداری، خرید
        df['target'] = pd.cut(df['future_pct_change'], bins=bins, labels=labels)
        
        # حذف ردیف‌های بدون مقدار هدف
        df = df.dropna(subset=['target']).copy()
        
        return df
    
    def get_historical_data_with_targets(self, symbol, hours=48):
        """دریافت داده‌های تاریخی با متغیرهای هدف"""
        try:
            # دریافت لیست تمام فیچرهای موجود
            all_features = self.db.get_available_features(symbol)
            
            if not all_features:
                print(f"No features found for {symbol}")
                return pd.DataFrame()
            
            # دریافت تاریخچه فیچرها
            df = self.db.get_features_history(symbol, all_features, hours)
            
            if df.empty:
                print(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # ایجاد متغیر هدف
            df = self.create_target_variable(df)
            
            if df is None or df.empty:
                print(f"Could not create target variable for {symbol}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error getting historical data with targets: {e}")
            return pd.DataFrame()
    
    def calculate_feature_importance_rf(self, df):
        """محاسبه اهمیت فیچرها با استفاده از Random Forest"""
        if df.empty or 'target' not in df.columns:
            return {}
            
        try:
            # حذف ستون‌های هدف و محاسبه شده از ویژگی‌ها
            X = df.drop(['target', 'future_pct_change'], axis=1, errors='ignore')
            y = df['target'].astype(int)
            
            # حذف ستون‌هایی با مقادیر NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # نرمال‌سازی داده‌ها
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # آموزش Random Forest
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(X_scaled, y)
            
            # محاسبه اهمیت فیچرها
            importances = rf.feature_importances_
            
            # مرتب‌سازی فیچرها براساس اهمیت
            feature_importances = {X.columns[i]: importances[i] for i in range(len(X.columns))}
            return feature_importances
            
        except Exception as e:
            print(f"Error calculating feature importance with RF: {e}")
            return {}
    
    def calculate_feature_importance_mi(self, df):
        """محاسبه اهمیت فیچرها با استفاده از Mutual Information"""
        if df.empty or 'target' not in df.columns:
            return {}
            
        try:
            # حذف ستون‌های هدف و محاسبه شده از ویژگی‌ها
            X = df.drop(['target', 'future_pct_change'], axis=1, errors='ignore')
            y = df['target'].astype(int)
            
            # حذف ستون‌هایی با مقادیر NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # محاسبه Mutual Information
            mi = mutual_info_classif(X, y, random_state=42)
            
            # مرتب‌سازی فیچرها براساس اهمیت
            feature_importances = {X.columns[i]: mi[i] for i in range(len(X.columns))}
            return feature_importances
            
        except Exception as e:
            print(f"Error calculating feature importance with MI: {e}")
            return {}
    
    def evaluate_features(self, symbol, hours=48, method='combined'):
        """
        ارزیابی اهمیت فیچرها
        
        Args:
            symbol: نماد مورد نظر
            hours: تعداد ساعت‌های گذشته برای بررسی
            method: روش ارزیابی ('rf', 'mi', یا 'combined')
            
        Returns:
            دیکشنری از فیچرها و اهمیت آنها
        """
        df = self.get_historical_data_with_targets(symbol, hours)
        
        if df.empty:
            print(f"No data available for evaluating features of {symbol}")
            return {}
            
        print(f"Evaluating features for {symbol} with {df.shape[0]} samples and {df.shape[1]-2} features")
        
        if method == 'rf':
            return self.calculate_feature_importance_rf(df)
        elif method == 'mi':
            return self.calculate_feature_importance_mi(df)
        else:  # روش ترکیبی
            rf_importances = self.calculate_feature_importance_rf(df)
            mi_importances = self.calculate_feature_importance_mi(df)
            
            # ترکیب دو روش
            combined_importances = {}
            for feature in rf_importances:
                if feature in mi_importances:
                    # میانگین وزنی (وزن بیشتر به Random Forest)
                    combined_importances[feature] = 0.7 * rf_importances[feature] + 0.3 * mi_importances[feature]
                else:
                    combined_importances[feature] = rf_importances[feature]
            
            return combined_importances
    
    def get_top_features(self, symbol, n=50, hours=48, method='combined'):
        """
        دریافت لیست بهترین فیچرها
        
        Args:
            symbol: نماد مورد نظر
            n: تعداد فیچرهای مورد نظر
            hours: تعداد ساعت‌های گذشته برای بررسی
            method: روش ارزیابی
            
        Returns:
            لیست بهترین فیچرها و اهمیت آنها
        """
        importances = self.evaluate_features(symbol, hours, method)
        
        # مرتب‌سازی براساس اهمیت
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # بازگرداندن n فیچر برتر
        return sorted_importances[:n]