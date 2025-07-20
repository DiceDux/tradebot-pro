import joblib
import os
from catboost import CatBoostClassifier
import numpy as np

MODEL_PATH = "model/catboost_tradebot_pro.pkl"
FEATURES_PATH = "model/catboost_features.pkl"
DYNAMIC_MODEL_PATH = "model/catboost_tradebot_pro_active.pkl"
DYNAMIC_FEATURES_PATH = "model/catboost_active_features.pkl"

def train_model(X, y, model_path=MODEL_PATH, features_path=FEATURES_PATH):
    from collections import Counter
    # اگر y فقط یک کلاس دارد، آموزش مدل معنی ندارد!
    unique_classes = set(y)
    if len(unique_classes) < 2:
        print(f"[CatBoost][Error] Cannot train: Target contains only one unique value ({list(unique_classes)})")
        return None

    class_weights = {}
    counts = Counter(y)
    for cls in counts:
        class_weights[cls] = 1

    model = CatBoostClassifier(
        iterations=300,
        verbose=0,
        loss_function="MultiClass",
        class_weights=[class_weights.get(c,1) for c in sorted(class_weights)]
    )
    model.fit(X, y)
    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), features_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Feature columns saved to {features_path}")
    return model

def retrain_active_model(X, y, active_features):
    # آموزش مجدد مدل فقط با فیچرهای فعال
    X_active = X[active_features]
    print(f"Retraining model with features: {active_features}")
    unique_classes = set(y)
    if len(unique_classes) < 2:
        print(f"[CatBoost][Error] Cannot retrain: Target contains only one unique value ({list(unique_classes)})")
        return None
    model = train_model(X_active, y, model_path=DYNAMIC_MODEL_PATH, features_path=DYNAMIC_FEATURES_PATH)
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    else:
        raise FileNotFoundError("Trained model or feature columns not found. Run train_model.py first.")

def load_dynamic_model():
    if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(DYNAMIC_FEATURES_PATH):
        model = joblib.load(DYNAMIC_MODEL_PATH)
        feature_names = joblib.load(DYNAMIC_FEATURES_PATH)
        return model, feature_names
    else:
        return None, None

def predict_signals(model, feature_names, features_df):
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    try:
        y_pred = model.predict(features_df)[0]
        proba = model.predict_proba(features_df)[0]
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        if isinstance(y_pred, (int, np.integer)):
            signal = signal_map.get(int(y_pred), "Hold")
        else:
            signal = ["Sell", "Hold", "Buy"][int(y_pred)]
        analysis = {
            "signal": signal,
            "confidence": float(np.max(proba)),
            "proba": proba.tolist(),
            "features": features_df.to_dict(orient="records")[0]
        }
        return signal, analysis
    except Exception as e:
        return "Hold", {
            "signal": "Hold",
            "confidence": 0.0,
            "features": features_df.to_dict(orient='records')[0],
            "error": str(e)
        }
