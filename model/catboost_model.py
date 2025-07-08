import joblib
import os
from catboost import CatBoostClassifier
import numpy as np

MODEL_PATH = "model/catboost_tradebot_pro.pkl"
FEATURES_PATH = "model/catboost_features.pkl"

def train_model(X, y):
    model = CatBoostClassifier(iterations=300, verbose=0)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Feature columns saved to {FEATURES_PATH}")
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    else:
        raise FileNotFoundError("Trained model or feature columns not found. Run train_model.py first.")

def predict_signals(model, feature_names, features_df):
    # تضمین سازگاری ترتیب و نام ستون‌ها
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    try:
        y_pred = model.predict(features_df)[0]
        proba = model.predict_proba(features_df)[0]
        signal = ["Sell", "Hold", "Buy"][int(y_pred)]
        analysis = {
            "signal": signal,
            "confidence": float(np.max(proba)),
            "proba": proba.tolist(),
            "features": features_df.to_dict(orient="records")[0]
        }
        return signal, analysis
    except Exception as e:
        return "Hold", {"signal": "Hold", "confidence": 0.0, "features": features_df.to_dict(orient='records')[0], "error": str(e)}