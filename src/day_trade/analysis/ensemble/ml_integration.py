"""
機械学習統合モジュール
分割されたML機能を統合するメインクラス
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...utils.logging_config import get_context_logger
from .ml_models import MLModelManager
from .ml_prediction import MLPredictionGenerator
from .ml_training import MLTrainingManager

logger = get_context_logger(__name__, component="ensemble_ml_integration")


class MLIntegrationManager:
    """機械学習統合管理クラス"""

    def __init__(self, enable_ml_models: bool = True, models_dir: Optional[str] = None):
        """
        Args:
            enable_ml_models: 機械学習モデルを有効にするか
            models_dir: 機械学習モデル保存ディレクトリ
        """
        self.enable_ml_models = enable_ml_models
        self.models_dir = models_dir

        # 各コンポーネントの初期化
        self.model_manager = MLModelManager(enable_ml_models, models_dir)
        self.prediction_generator = MLPredictionGenerator(self.model_manager)
        self.training_manager = MLTrainingManager(self.model_manager)

        # 履歴管理
        self.ml_predictions_history = []

    def generate_ml_predictions(
        self, df: pd.DataFrame, indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """機械学習予測を生成"""
        try:
            predictions, feature_importance = self.prediction_generator.generate_ml_predictions(
                df, indicators
            )
            
            # 履歴に追加
            if predictions:
                self.ml_predictions_history.extend(
                    self.prediction_generator.get_predictions_history(1)
                )
                
                # 履歴サイズを制限
                if len(self.ml_predictions_history) > 100:
                    self.ml_predictions_history = self.ml_predictions_history[-100:]
            
            return predictions, feature_importance

        except Exception as e:
            logger.error(f"ML予測生成エラー: {e}")
            return None, None

    def train_ml_models(
        self, historical_data: pd.DataFrame, retrain: bool = False
    ) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        return self.training_manager.train_ml_models(historical_data, retrain)

    def validate_ml_models(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """機械学習モデルを検証"""
        return self.training_manager.validate_models(validation_data)

    def get_ml_model_info(self) -> Dict[str, Any]:
        """機械学習モデルの情報を取得"""
        base_info = self.model_manager.get_ml_model_info()
        
        # 追加情報を付加
        if base_info.get("ml_enabled", False):
            base_info.update({
                "prediction_history_count": len(self.ml_predictions_history),
                "fitted_models": self.model_manager.get_fitted_models(),
                "available_models": self.model_manager.get_model_list(),
            })
            
            # 最近の予測パフォーマンス情報を追加
            performance_summary = self.prediction_generator.get_model_performance_summary()
            if performance_summary:
                base_info["recent_performance"] = performance_summary
        
        return base_info

    def is_available(self) -> bool:
        """機械学習機能が利用可能かチェック"""
        return self.model_manager.is_available()

    def get_fitted_models(self) -> List[str]:
        """訓練済みモデルのリストを取得"""
        return self.model_manager.get_fitted_models()

    def get_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """予測結果の信頼度を計算"""
        return self.prediction_generator.get_prediction_confidence(predictions)

    def get_ensemble_prediction(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """アンサンブル予測を計算"""
        return self.prediction_generator.get_ensemble_prediction(predictions)

    def clear_predictions_history(self) -> None:
        """予測履歴をクリア"""
        self.ml_predictions_history.clear()
        self.prediction_generator.clear_predictions_history()

    def get_predictions_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """予測履歴を取得"""
        return self.ml_predictions_history[-limit:] if self.ml_predictions_history else []

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリーを取得"""
        try:
            if not self.is_available():
                return {"available": False}

            fitted_models = self.get_fitted_models()
            model_info = self.get_ml_model_info()
            
            summary = {
                "available": True,
                "total_models": len(model_info.get("models", {})),
                "fitted_models_count": len(fitted_models),
                "fitted_models": fitted_models,
                "prediction_history_length": len(self.ml_predictions_history),
                "feature_engineer_available": model_info.get("feature_engineer_available", False)
            }
            
            # 各モデルの基本情報を追加
            models_detail = {}
            for model_name, info in model_info.get("models", {}).items():
                models_detail[model_name] = {
                    "fitted": info.get("is_fitted", False),
                    "type": info.get("model_type", "unknown"),
                    "task": info.get("task_type", "unknown"),
                    "features": info.get("feature_count", 0)
                }
            
            summary["models_detail"] = models_detail
            
            return summary

        except Exception as e:
            logger.error(f"モデルサマリー取得エラー: {e}")
            return {"available": False, "error": str(e)}

    # 後方互換性のためのメソッド
    def _initialize_ml_components(self):
        """ML コンポーネントを初期化（後方互換性）"""
        # すでに __init__ で初期化済み
        pass

    def _initialize_ml_models(self):
        """ML モデルを初期化（後方互換性）"""
        # すでに MLModelManager で初期化済み
        pass

    def _generate_ml_predictions(self, df, indicators=None):
        """ML 予測を生成（後方互換性）"""
        return self.generate_ml_predictions(df, indicators)

    def _train_ml_models(self, historical_data, retrain=False):
        """ML モデルを訓練（後方互換性）"""
        return self.train_ml_models(historical_data, retrain)

    def _get_ml_model_info(self):
        """ML モデル情報を取得（後方互換性）"""
        return self.get_ml_model_info()