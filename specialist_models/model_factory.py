"""
کارخانه تولید مدل‌های متخصص با بهینه‌سازی خودکار
"""
import numpy as np
import pandas as pd
import logging
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

logger = logging.getLogger("model_factory")

class ModelFactory:
    """کارخانه تولید مدل‌های متخصص با بهینه‌سازی خودکار"""
    
    @staticmethod
    def create_optimized_model(model_type, n_estimators=500):
        """
        ایجاد مدل با پارامترهای بهینه
        
        Args:
            model_type: نوع مدل ('rf' برای RandomForest یا 'gb' برای GradientBoosting)
            n_estimators: تعداد درخت‌ها
            
        Returns:
            model: مدل آماده
        """
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                criterion='gini',
                n_jobs=-1,
                random_state=42
            )
        elif model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=15,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def evaluate_models(X, y, feature_set_name):
        """
        ارزیابی چندین مدل و انتخاب بهترین
        
        Args:
            X: ویژگی‌های ورودی
            y: برچسب‌های هدف
            feature_set_name: نام مجموعه ویژگی‌ها
            
        Returns:
            best_model: بهترین مدل آموزش‌دیده
        """
        start_time = time.time()
        logger.info(f"Evaluating models for {feature_set_name} with {X.shape[1]} features and {len(X)} samples")
        print(f"Evaluating models for {feature_set_name}...")
        
        # بررسی توزیع کلاس‌ها
        class_dist = np.bincount(y)
        for i, count in enumerate(class_dist):
            logger.info(f"Class {i}: {count} samples ({count/len(y)*100:.1f}%)")
            
        # مدل‌های کاندیدا با پارامترهای مختلف
        models = {
            "rf_default": ModelFactory.create_optimized_model('rf', n_estimators=500),
            "gb_default": ModelFactory.create_optimized_model('gb', n_estimators=300),
            "rf_deep": RandomForestClassifier(
                n_estimators=300, 
                max_depth=20, 
                min_samples_split=5,
                class_weight='balanced',
                n_jobs=-1, 
                random_state=42
            ),
            "gb_shallow": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # اعتبارسنجی متقابل برای هر مدل
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            try:
                # ارزیابی با اعتبارسنجی متقابل
                start = time.time()
                cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
                
                results[name] = {
                    'accuracy_mean': cv_accuracy.mean(),
                    'accuracy_std': cv_accuracy.std(),
                    'f1_mean': cv_f1.mean(),
                    'f1_std': cv_f1.std(),
                    'time': time.time() - start
                }
                
                logger.info(f"Model {name}: CV accuracy={cv_accuracy.mean():.4f}±{cv_accuracy.std():.4f}, "
                           f"F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}, Time={results[name]['time']:.1f}s")
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
                results[name] = {'f1_mean': 0.0}
        
        # انتخاب بهترین مدل بر اساس F1
        best_model_name = max(results, key=lambda k: results[k]['f1_mean'])
        best_model = models[best_model_name]
        logger.info(f"Best model for {feature_set_name}: {best_model_name} "
                   f"(F1={results[best_model_name]['f1_mean']:.4f})")
        
        # آموزش مدل نهایی روی کل داده‌ها
        logger.info(f"Training final {best_model_name} model on all data")
        best_model.fit(X, y)
        
        # ارزیابی روی داده‌های آموزش
        y_pred = best_model.predict(X)
        train_accuracy = accuracy_score(y, y_pred)
        train_f1 = f1_score(y, y_pred, average='macro')
        
        logger.info(f"Final {feature_set_name} model trained with accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, "
                  f"Total time: {time.time()-start_time:.1f}s")
        print(f"Final {feature_set_name} model: {best_model_name}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        # گزارش طبقه‌بندی
        print(classification_report(y, y_pred))
        
        # نمایش اهمیت ویژگی‌ها اگر قابل دسترسی باشد
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importances = pd.Series(importances, index=X.columns)
            top_features = feature_importances.sort_values(ascending=False).head(15)
            
            print("\nTop 15 most important features:")
            for name, importance in top_features.items():
                print(f"- {name}: {importance:.4f}")
        
        return best_model
