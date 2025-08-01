"""
مدل متخصص نوسان و حجم
"""
from sklearn.ensemble import ExtraTreesClassifier
from .base_specialist_model import BaseSpecialistModel

class VolatilityModel(BaseSpecialistModel):
    def __init__(self):
        super().__init__('volatility', 'volatility')
    
    def _create_model(self):
        """ایجاد مدل متخصص نوسان و حجم"""
        return ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            bootstrap=True,
            class_weight='balanced',
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
