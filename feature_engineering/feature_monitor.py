import numpy as np
from .feature_config import FEATURE_CONFIG

class FeatureMonitor:
    def __init__(self, model, all_feature_names):
        self.model = model
        self.all_features = all_feature_names
        self.active_features = set([f for f, v in FEATURE_CONFIG.items() if v])

    def evaluate_features(self, X, y=None, window=100):
        """ X: دیتافریم فیچرهای فعال اخیر، y: اگر داشتی لیبل واقعی، window: سایز پنجره برای ارزیابی """
        # می‌توانی از shap, permutation importance یا correlation با سیگنال خروجی استفاده کنی
        importances = []
        for f in self.all_features:
            if f in X.columns:
                if y is not None:
                    # Correlation با سیگنال خروجی (مثال ساده)
                    val = abs(np.corrcoef(X[f].values[-window:], y[-window:])[0,1]) if len(X[f]) >= window else 0
                else:
                    # اگر لیبل نداری، از واریانس یا تغییرپذیری فیچر استفاده کن
                    val = np.std(X[f].values[-window:]) if len(X[f]) >= window else 0
                importances.append((f, val))
            else:
                importances.append((f, 0))
        importances.sort(key=lambda x: -x[1])
        # فقط top N فیچر را فعال کن
        top_n = min(20, len(importances))
        selected = set([f for f, v in importances[:top_n] if v > 0])
        # بروزرسانی پارامتریک
        for f in FEATURE_CONFIG:
            FEATURE_CONFIG[f] = f in selected
        self.active_features = selected
        print(f"[FeatureMonitor] Active features: {list(selected)}")
        return self.active_features

    def get_active_feature_names(self):
        return list(self.active_features)
