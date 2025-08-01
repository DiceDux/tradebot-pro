"""
مدل متخصص اسیلاتورها
"""
from sklearn.ensemble import GradientBoostingClassifier
from .base_specialist_model import BaseSpecialistModel

class OscillatorsModel(BaseSpecialistModel):
    def __init__(self):
        super().__init__('oscillators', 'oscillators')
    
    def _create_model(self):
        """ایجاد مدل متخصص اسیلاتورها"""
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
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