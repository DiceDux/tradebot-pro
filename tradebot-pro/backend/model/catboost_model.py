import joblib
import os
from catboost import CatBoostClassifier
import numpy as np

MODEL_PATH = "models/catboost_tradebot_pro.pkl"

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # آموزش مدل اولیه با داده تست (در فاز بعدی با دیتا واقعی آموزش داده می‌شود)
        model = CatBoostClassifier(verbose=0)
        # فرض: X_train, y_train = ...
        # model.fit(X_train, y_train)
        # joblib.dump(model, MODEL_PATH)
        # برای شروع مدل خالی برمی‌گردد
    return model

def predict_signals(model, features_df):
    # فرض: مدل آموزش دیده است و features_df ساختار صحیح دارد
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
        # در صورت نبود مدل یا خطا
        return "Hold", {"signal": "Hold", "confidence": 0.0, "features": features_df.to_dict(orient="records")[0]}