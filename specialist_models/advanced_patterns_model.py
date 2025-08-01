"""
مدل متخصص برای شناسایی الگوهای پیشرفته بازار
"""
from specialist_models.base_specialist_model import BaseSpecialistModel

class AdvancedPatternsModel(BaseSpecialistModel):
    """مدل متخصص برای شناسایی الگوهای پیشرفته بازار"""
    
    def __init__(self):
        super().__init__(model_name='advanced_patterns', feature_group='advanced_patterns')

    def create_model(self):
        """ایجاد مدل XGBoost برای الگوهای پیشرفته"""
        from xgboost import XGBClassifier
        
        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )
        return model
