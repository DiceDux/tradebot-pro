"""
پایپلاین مدل برای پیش‌بینی
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

from specialist_models.moving_averages_model import MovingAveragesModel
from specialist_models.oscillators_model import OscillatorsModel
from specialist_models.volatility_model import VolatilityModel
from specialist_models.candlestick_model import CandlestickModel
from specialist_models.news_model import NewsModel
from meta_model.model_combiner import ModelCombiner
from feature_selection.feature_groups import FEATURE_GROUPS

class ModelPipeline:
    def __init__(self, symbols=None):
        """
        پایپلاین مدل برای پیش‌بینی
        
        Args:
            symbols: لیست نمادهای مورد نظر
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        
        # ایجاد مدل‌های متخصص
        self.specialist_models = {
            'moving_averages': MovingAveragesModel(),
            'oscillators': OscillatorsModel(),
            'volatility': VolatilityModel(),
            'candlestick': CandlestickModel(),
            'news': NewsModel()
        }
        
        # ایجاد مدل ترکیب‌کننده
        self.combiner = ModelCombiner(list(self.specialist_models.values()))
    
    def load_models(self):
        """بارگذاری تمام مدل‌ها"""
        # بارگذاری مدل‌های متخصص
        for name, model in self.specialist_models.items():
            success = model.load()
            if not success:
                print(f"Warning: Could not load {name} model")
        
        # بارگذاری مدل متا
        self.combiner.load()
    
    def train_models(self, training_data):
        """
        آموزش تمام مدل‌ها
        
        Args:
            training_data: داده‌های آموزش
        """
        # آموزش هر مدل متخصص با فیچرهای مربوطه
        for name, model in self.specialist_models.items():
            # انتخاب فیچرهای مربوط به گروه این مدل
            group_features = FEATURE_GROUPS.get(model.feature_group, [])
            
            # فیلتر کردن فیچرهای موجود در داده‌ها
            available_features = [f for f in group_features if f in training_data.columns]
            
            if available_features:
                # آموزش مدل
                X = training_data[available_features]
                y = training_data['target']
                
                model.train(X, y)
                model.save()
            else:
                print(f"Warning: No features available for {name} model")
        
        # آموزش مدل متا
        self._train_meta_model(training_data)
    
    def _train_meta_model(self, training_data):
        """
        آموزش مدل متا
        
        Args:
            training_data: داده‌های آموزش
        """
        # ایجاد داده‌های ورودی برای مدل متا
        meta_features = []
        
        # پیش‌بینی با هر مدل متخصص
        for name, model in self.specialist_models.items():
            if model.model is None:
                print(f"Warning: {name} model not trained, skipping")
                continue
                
            # انتخاب فیچرهای مربوط به گروه این مدل
            required_features = model.get_required_features()
            available_features = [f for f in required_features if f in training_data.columns]
            
            if available_features:
                try:
                    # پیش‌بینی احتمالات کلاس‌ها
                    X = training_data[available_features]
                    _, probas = model.predict(X)
                    
                    # افزودن احتمالات به فیچرهای مدل متا
                    for i in range(len(probas)):
                        if i == 0:  # فقط برای اولین نمونه، ستون‌های جدید ایجاد می‌کنیم
                            for j in range(probas.shape[1]):
                                meta_features.append(pd.Series(probas[:, j], name=f"{name}_class{j}"))
                
                except Exception as e:
                    print(f"Error in {name} prediction for meta-model training: {e}")
        
        # ساخت DataFrame برای مدل متا
        if meta_features:
            meta_X = pd.concat(meta_features, axis=1)
            meta_y = training_data['target']
            
            # آموزش مدل متا
            self.combiner.train(meta_X, meta_y)
            self.combiner.save()
        else:
            print("Warning: No meta-features available for meta-model training")
    
    def analyze(self, symbol, features, active_features=None):
        """
        تحلیل بازار و تولید سیگنال
        
        Args:
            symbol: نماد مورد نظر
            features: داده‌های فیچر
            active_features: لیست فیچرهای فعال
            
        Returns:
            سیگنال، اطمینان و جزئیات
        """
        if features.empty:
            print("No features provided")
            return 1, 0.0, {}  # Hold با اطمینان صفر
            
        # فیلتر کردن فیچرهای فعال
        if active_features:
            # بررسی وجود فیچرهای فعال در داده‌ها
            available_features = [f for f in active_features if f in features.columns]
            
            if available_features:
                features_filtered = features[available_features]
            else:
                print("No active features found in data")
                return 1, 0.0, {}  # Hold با اطمینان صفر
        else:
            features_filtered = features
        
        # تبدیل DataFrame به دیکشنری
        features_dict = features_filtered.iloc[0].to_dict()
        
        # پیش‌بینی با مدل ترکیبی
        try:
            pred, probas, confidence = self.combiner.predict(features_dict)
            
            # تبدیل سیگنال به عدد صحیح
            signal = int(pred)
            
            # ایجاد جزئیات
            details = {
                'probabilities': {
                    'sell': float(probas[0]),
                    'hold': float(probas[1]),
                    'buy': float(probas[2])
                },
                'confidence': float(confidence),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return signal, confidence, details
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 1, 0.0, {}  # Hold با اطمینان صفر
