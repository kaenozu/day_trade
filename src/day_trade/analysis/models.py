#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Models - 機械学習モデル管理
Issue #939対応: 最適化済みハイパーパラメータの読み込み
"""

import json
from pathlib import Path
from abc import ABC, abstractmethod
import polars as pl
from polars import DataFrame
import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional

from src.day_trade.analysis.feature_enhancer import NewsSentimentAnalyzer

class TradingModel(ABC):
    """取引モデルの抽象基底クラス"""
    def __init__(self):
        self.sentiment_analyzer = NewsSentimentAnalyzer()

    @abstractmethod
    def predict(self, data: DataFrame, symbol: str, company_name: str) -> Dict[str, Any]:
        pass

    def _create_features(self, data: DataFrame, symbol: str, company_name: str) -> DataFrame:
        # ... (implementation no change) ...
        features = data.with_columns([
            pl.col("Close").rolling_mean(window_size=5).alias("sma5"),
            pl.col("Close").rolling_mean(window_size=25).alias("sma25"),
            pl.col("Close").diff().clip_lower(0).rolling_mean(14).alias("gain"),
            (-pl.col("Close").diff().clip_upper(0)).rolling_mean(14).alias("loss"),
        ])
        features = features.with_columns([(100 - (100 / (1 + pl.col("gain") / pl.col("loss")))).alias("rsi")])
        sentiment_data = self.sentiment_analyzer.get_sentiment_for_symbol(symbol, company_name)
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
        features = features.with_columns([pl.lit(sentiment_score).alias('sentiment_score')])
        return features.drop_nulls()

class LightGBMModel(TradingModel):
    """LightGBMモデル (最適化パラメータ対応)"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model = None
        self.params = self._load_best_params()

        if model_path:
            try:
                self.model = lgb.Booster(model_file=model_path)
                print("Loaded pre-trained LightGBM model.")
            except lgb.basic.LightGBMError:
                print(f"Warning: Could not load model from {model_path}. Using dummy model.")
                self.model = self._create_dummy_model()
        else:
            self.model = self._create_dummy_model()

    def _load_best_params(self) -> Dict[str, Any]:
        """最適化済みのハイパーパラメータをロード"""
        params_path = Path(__file__).parent.parent.parent / 'hyperparameter_optimization/best_params.json'
        if params_path.exists():
            print(f"Loading best hyperparameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f)
        else:
            print("No best_params.json found. Using default parameters.")
            # デフォルトパラメータ
            return {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt'
            }

    def _create_dummy_model(self):
        print("Creating a dummy LightGBM model.")
        return None

    def predict(self, data: DataFrame, symbol: str, company_name: str) -> Dict[str, Any]:
        if self.model is None:
            # ダミーモデルはパラメータ情報を理由に含める
            reason = f"Dummy LightGBM Model Prediction (using params: {self.params.get('boosting_type')})"
            return {
                'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': np.random.uniform(0.5, 0.9),
                'reason': reason
            }

        features = self._create_features(data, symbol, company_name)
        if features.is_empty():
            return {'signal': 'HOLD', 'confidence': 0.1, 'reason': 'Not enough data for prediction'}
        
        feature_cols = [col for col in features.columns if col not in ['Date', 'Symbol']]
        prediction = self.model.predict(features[feature_cols])
        
        predicted_class = np.argmax(prediction, axis=1)[-1]
        confidence = prediction[0][predicted_class]
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        return {
            'signal': signal_map.get(predicted_class, 'HOLD'),
            'confidence': confidence,
            'reason': f'LightGBM prediction (class: {predicted_class})'
        }
