"""
مدل متخصص الگوهای شمعی و پرایس اکشن
"""
from sklearn.ensemble import RandomForestClassifier
from .base_specialist_model import BaseSpecialistModel

class CandlestickModel(BaseSpecialistModel):
    def __init__(self):
        super().__init__('candlestick', 'candlesticks')
    
    def _create_model(self):
        """ایجاد مدل متخصص الگوهای شمعی"""
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42
        )
        
    def get_feature_importance(self):
        """دریافت اهمیت فیچرها در مدل"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return {
            self.feature_names[i]: self.model.feature_importances_[i]
            for i in range(len(self.feature_names))
        }