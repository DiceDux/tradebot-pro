"""
مدل متخصص میانگین‌های متحرک و روندها
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from .base_specialist_model import BaseSpecialistModel

class MovingAveragesModel(BaseSpecialistModel):
    def __init__(self):
        super().__init__('moving_averages', 'moving_averages')
    
    def _create_model(self):
        """ایجاد مدل متخصص میانگین‌های متحرک"""
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )
        
    def get_feature_importance(self):
        """دریافت اهمیت فیچرها در مدل"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return {
            self.feature_names[i]: self.model.feature_importances_[i]
            for i in range(len(self.feature_names))
        }
