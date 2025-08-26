"""
機械学習モデル管理モジュール
基本的なモデル初期化と管理機能を提供
"""

from typing import Any, Dict, List, Optional

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_ml_models")


class MLModelManager:
    """機械学習モデル管理クラス"""

    def __init__(self, enable_ml_models: bool = True, models_dir: Optional[str] = None):
        """
        Args:
            enable_ml_models: 機械学習モデルを有効にするか
            models_dir: 機械学習モデル保存ディレクトリ
        """
        self.enable_ml_models = enable_ml_models
        self.models_dir = models_dir
        
        self.ml_manager = None
        self.feature_engineer = None

        if self.enable_ml_models:
            self._initialize_ml_components()

    def _initialize_ml_components(self) -> None:
        """機械学習コンポーネントを初期化"""
        try:
            from ..feature_engineering import AdvancedFeatureEngineer
            from ..ml_models import MLModelManager as CoreMLManager, ModelConfig

            self.ml_manager = CoreMLManager()
            self.feature_engineer = AdvancedFeatureEngineer()
            self._initialize_ml_models()
            
        except ImportError as e:
            logger.warning(f"機械学習モジュールが利用できません: {e}")
            self.enable_ml_models = False

    def _initialize_ml_models(self) -> None:
        """機械学習モデルを初期化"""
        if not self.ml_manager:
            return

        try:
            from ..ml_models import ModelConfig

            # 1. 回帰モデル（リターン予測）
            return_model_config = ModelConfig(
                model_type="random_forest",
                task_type="regression",
                cv_folds=5,
                model_params={"n_estimators": 100, "max_depth": 10},
            )
            self.ml_manager.create_model("return_predictor", return_model_config)

            # 2. 分類モデル（方向性予測）
            direction_model_config = ModelConfig(
                model_type="gradient_boosting",
                task_type="classification",
                cv_folds=5,
                model_params={"n_estimators": 100, "learning_rate": 0.1},
            )
            self.ml_manager.create_model("direction_predictor", direction_model_config)

            # 3. ボラティリティ予測モデル
            volatility_model_config = ModelConfig(
                model_type="xgboost",
                task_type="regression",
                cv_folds=3,
                model_params={"n_estimators": 50, "max_depth": 6},
            )
            self.ml_manager.create_model(
                "volatility_predictor", volatility_model_config
            )

            # 4. メタラーナー（アンサンブル最適化）
            meta_model_config = ModelConfig(
                model_type="linear", task_type="regression", cv_folds=3
            )
            self.ml_manager.create_model("meta_learner", meta_model_config)

            logger.info("機械学習モデルを初期化しました")

        except Exception as e:
            logger.error(f"機械学習モデル初期化エラー: {e}")
            self.enable_ml_models = False

    def get_ml_model_info(self) -> Dict[str, Any]:
        """機械学習モデルの情報を取得"""
        if not self.enable_ml_models or not self.ml_manager:
            return {"ml_enabled": False}

        info = {
            "ml_enabled": True,
            "models": {},
            "feature_engineer_available": self.feature_engineer is not None,
        }

        for model_name in self.ml_manager.list_models():
            try:
                model_info = self.ml_manager.get_model_info(model_name)
                info["models"][model_name] = {
                    "model_type": model_info["model_type"],
                    "task_type": model_info["task_type"],
                    "is_fitted": model_info["is_fitted"],
                    "feature_count": model_info["feature_count"],
                }
                if model_info["is_fitted"] and "feature_importance" in model_info:
                    # 上位5個の重要特徴量のみ表示
                    importance = model_info["feature_importance"]
                    if importance:
                        top_features = dict(list(importance.items())[:5])
                        info["models"][model_name]["top_features"] = top_features
            except Exception as e:
                info["models"][model_name] = {"error": str(e)}

        return info

    def is_available(self) -> bool:
        """機械学習機能が利用可能かチェック"""
        return (
            self.enable_ml_models 
            and self.ml_manager is not None 
            and self.feature_engineer is not None
        )

    def get_fitted_models(self) -> List[str]:
        """訓練済みモデルのリストを取得"""
        if not self.is_available():
            return []

        try:
            fitted_models = []
            for model_name in self.ml_manager.list_models():
                if self.ml_manager.models[model_name].is_fitted:
                    fitted_models.append(model_name)
            return fitted_models
        except Exception as e:
            logger.error(f"訓練済みモデル取得エラー: {e}")
            return []

    def get_model_list(self) -> List[str]:
        """利用可能なモデルのリストを取得"""
        if not self.is_available():
            return []
        return self.ml_manager.list_models()

    def get_core_ml_manager(self):
        """コアMLマネージャーを取得（他のモジュールから使用）"""
        return self.ml_manager

    def get_feature_engineer(self):
        """特徴量エンジニアを取得（他のモジュールから使用）"""
        return self.feature_engineer