import numpy as np
from .feature_config import FEATURE_CONFIG

def auto_select_features(model, feature_names, top_n=20):
    """
    بر اساس feature importance، فقط top_n فیچر را فعال می‌کند.
    """
    importances = np.array(model.get_feature_importance())
    sorted_indices = importances.argsort()[::-1]
    selected = [feature_names[i] for i in sorted_indices[:top_n]]
    for key in FEATURE_CONFIG:
        FEATURE_CONFIG[key] = key in selected
    print(f"Selected features: {selected}")
    return FEATURE_CONFIG
