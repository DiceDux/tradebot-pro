import numpy as np
from .feature_config import FEATURE_CONFIG
import os

class FeatureMonitor:
    def __init__(self, model, all_feature_names, config_path=None):
        self.model = model
        self.all_features = all_feature_names
        self.active_features = set([f for f, v in FEATURE_CONFIG.items() if v])
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "feature_config.py")

    def evaluate_features(self, X, y=None, window=100):
        importances = []
        for f in self.all_features:
            if f in X.columns:
                if y is not None:
                    val = abs(np.corrcoef(X[f].values[-window:], y[-window:])[0,1]) if len(X[f]) >= window else 0
                else:
                    val = np.std(X[f].values[-window:]) if len(X[f]) >= window else 0
                importances.append((f, val))
            else:
                importances.append((f, 0))
        importances.sort(key=lambda x: -x[1])
        top_n = min(20, len(importances))
        selected = set([f for f, v in importances[:top_n] if v > 0])
        # بروزرسانی FEATURE_CONFIG در حافظه
        for f in FEATURE_CONFIG:
            FEATURE_CONFIG[f] = f in selected
        self.active_features = selected
        print(f"[FeatureMonitor] Active features: {list(selected)}")
        # ذخیره خودکار FEATURE_CONFIG در فایل
        self.save_feature_config()
        return self.active_features

    def save_feature_config(self):
        # ذخیره FEATURE_CONFIG به صورت قابل استفاده توسط پروژه
        config_lines = ["FEATURE_CONFIG = {"]
        for k, v in FEATURE_CONFIG.items():
            config_lines.append(f'    "{k}": {str(v)},')
        config_lines.append("}")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(config_lines))

    def get_active_feature_names(self):
        return list(self.active_features)
