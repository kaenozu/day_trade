import logging
from typing import Dict, Any

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.day_trade.ml.core_types import BaseModelTrainer, ModelType

# 機械学習ライブラリの可用性チェック (ml_prediction_models_improved.pyから移動)
SKLEARN_AVAILABLE = True # sklearnは常に利用可能と仮定

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """Random Forestモデル作成"""
        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return RandomForestClassifier(**final_params)
        else:
            return RandomForestRegressor(**final_params)


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """XGBoostモデル作成"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return xgb.XGBClassifier(**final_params)
        else:
            return xgb.XGBRegressor(**final_params)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """LightGBMモデル作成"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return lgb.LGBMClassifier(**final_params)
        else:
            return lgb.LGBMRegressor(**final_params)
