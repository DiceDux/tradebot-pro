"""
هماهنگ‌ساز ویژگی‌ها بین مراحل مختلف آموزش و پیش‌بینی
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_harmonizer")

class FeatureHarmonizer:
    """کلاس هماهنگ‌کننده ویژگی‌ها بین مراحل مختلف"""
    
    @staticmethod
    def harmonize_features(X_train, X_test=None):
        """
        اطمینان از هماهنگی ویژگی‌های داده آموزش و تست
        
        Args:
            X_train: دیتافریم ویژگی‌های آموزش
            X_test: دیتافریم ویژگی‌های تست (اختیاری)
            
        Returns:
            tuple: (X_train_harmonized, X_test_harmonized)
        """
        if X_test is None:
            return X_train
            
        # ویژگی‌های مشترک
        common_features = list(set(X_train.columns) & set(X_test.columns))
        logger.info(f"Common features: {len(common_features)} out of {len(X_train.columns)}")
        
        # ویژگی‌های فقط در داده آموزش
        train_only = set(X_train.columns) - set(X_test.columns)
        if train_only:
            logger.warning(f"Features only in training data: {len(train_only)}")
            logger.debug(f"Train-only features: {sorted(list(train_only))}")
        
        # ویژگی‌های فقط در داده تست
        test_only = set(X_test.columns) - set(X_train.columns)
        if test_only:
            logger.warning(f"Features only in test data: {len(test_only)}")
            logger.debug(f"Test-only features: {sorted(list(test_only))}")
        
        # استفاده فقط از ویژگی‌های مشترک
        X_train_harmonized = X_train[common_features].copy()
        X_test_harmonized = X_test[common_features].copy()
        
        # بررسی وجود مقادیر نامعتبر
        X_train_harmonized = X_train_harmonized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_test_harmonized = X_test_harmonized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        logger.info(f"Harmonized features shape: train={X_train_harmonized.shape}, test={X_test_harmonized.shape}")
        return X_train_harmonized, X_test_harmonized
    
    @staticmethod
    def ensure_feature_compatibility(model, X):
        """
        اطمینان از سازگاری ویژگی‌های ورودی با مدل
        
        Args:
            model: مدل آموزش‌دیده
            X: دیتافریم ویژگی‌های ورودی
            
        Returns:
            DataFrame: دیتافریم ویژگی‌های سازگار با مدل
        """
        if not hasattr(model, 'feature_names_in_'):
            logger.warning("Model does not have feature_names_in_ attribute")
            return X
        
        model_features = model.feature_names_in_
        
        # ویژگی‌های گمشده
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in input data")
            
            # اضافه کردن ستون‌های گمشده با مقادیر صفر
            for feature in missing_features:
                X[feature] = 0.0
        
        # ویژگی‌های اضافی
        extra_features = set(X.columns) - set(model_features)
        if extra_features:
            logger.warning(f"Extra {len(extra_features)} features in input data")
        
        # انتخاب فقط ویژگی‌های مدل با همان ترتیب
        X_compatible = X[model_features].copy()
        
        return X_compatible
