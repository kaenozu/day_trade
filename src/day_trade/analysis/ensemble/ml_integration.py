"""
機械学習統合モジュール
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_ml")


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
        
        self.ml_manager = None
        self.feature_engineer = None
        self.ml_predictions_history = []

        if self.enable_ml_models:
            self._initialize_ml_components()

    def _initialize_ml_components(self) -> None:
        """機械学習コンポーネントを初期化"""
        try:
            from ..feature_engineering import AdvancedFeatureEngineer
            from ..ml_models import MLModelManager, ModelConfig

            self.ml_manager = MLModelManager()
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

    def generate_ml_predictions(
        self, df: pd.DataFrame, indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """機械学習予測を生成"""
        try:
            if not self.enable_ml_models or not self.ml_manager or not self.feature_engineer:
                return None, None

            if len(df) < 50:
                return None, None

            # 高度な特徴量を生成
            volume_data = df["Volume"] if "Volume" in df.columns else None
            features = self.feature_engineer.generate_all_features(
                price_data=df, volume_data=volume_data
            )

            if features.empty:
                return None, None

            # 最新の特徴量を取得
            latest_features = features.tail(1)

            predictions = {}
            feature_importance = {}

            # 各モデルで予測を実行
            for model_name in self.ml_manager.list_models():
                try:
                    model = self.ml_manager.models[model_name]
                    if model.is_fitted:
                        pred = self.ml_manager.predict(model_name, latest_features)
                        predictions[model_name] = (
                            float(pred[0]) if len(pred) > 0 else 0.0
                        )

                        # 特徴量重要度を取得
                        importance = model._get_feature_importance()
                        if importance:
                            feature_importance[model_name] = importance

                except Exception as e:
                    logger.warning(f"モデル {model_name} の予測でエラー: {e}")

            return predictions, feature_importance

        except Exception as e:
            logger.error(f"機械学習予測エラー: {e}")
            return None, None

    def train_ml_models(
        self, historical_data: pd.DataFrame, retrain: bool = False
    ) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        try:
            if (
                not self.enable_ml_models
                or not self.ml_manager
                or not self.feature_engineer
            ):
                return {"error": "機械学習機能が無効です"}

            if len(historical_data) < 200:
                return {"error": "訓練に十分なデータがありません（最低200日必要）"}

            logger.info("機械学習モデルの訓練を開始")

            # 特徴量生成
            volume_data = (
                historical_data["Volume"]
                if "Volume" in historical_data.columns
                else None
            )
            features = self.feature_engineer.generate_all_features(
                price_data=historical_data, volume_data=volume_data
            )

            if features.empty:
                return {"error": "特徴量生成に失敗しました"}

            # ターゲット変数生成
            from ..feature_engineering import create_target_variables
            targets = create_target_variables(historical_data, prediction_horizon=5)

            training_results = {}

            # 各モデルの訓練
            model_configs = [
                ("return_predictor", "future_returns"),
                ("direction_predictor", "future_direction"),
                ("volatility_predictor", "future_high_volatility"),
            ]

            for model_name, target_name in model_configs:
                try:
                    if target_name not in targets:
                        logger.warning(f"ターゲット変数 {target_name} が見つかりません")
                        continue

                    # データの整合性チェック
                    common_index = features.index.intersection(
                        targets[target_name].index
                    )
                    if len(common_index) < 100:
                        logger.warning(
                            f"モデル {model_name} に十分なデータがありません"
                        )
                        continue

                    X_train = features.loc[common_index]
                    y_train = targets[target_name].loc[common_index]

                    # モデルが存在しない場合は作成
                    if model_name not in self.ml_manager.list_models():
                        logger.warning(
                            f"モデル {model_name} が存在しません。スキップします。"
                        )
                        continue

                    # 既に訓練済みでretrainがFalseの場合はスキップ
                    if not retrain and self.ml_manager.models[model_name].is_fitted:
                        logger.info(f"モデル {model_name} は既に訓練済みです")
                        continue

                    # モデル訓練
                    logger.info(f"モデル {model_name} を訓練中...")
                    result = self.ml_manager.train_model(model_name, X_train, y_train)
                    training_results[model_name] = result

                    # モデル保存
                    try:
                        self.ml_manager.save_model(model_name)
                        logger.info(f"モデル {model_name} を保存しました")
                    except Exception as e:
                        logger.warning(f"モデル {model_name} の保存に失敗: {e}")

                except Exception as e:
                    logger.error(f"モデル {model_name} の訓練エラー: {e}")
                    training_results[model_name] = {"error": str(e)}

            # メタラーナーの訓練
            try:
                if len(training_results) >= 2:
                    meta_result = self._train_meta_learner(historical_data)
                    training_results["meta_learner"] = meta_result

            except Exception as e:
                logger.error(f"メタラーナー訓練エラー: {e}")
                training_results["meta_learner"] = {"error": str(e)}

            logger.info(f"機械学習モデル訓練完了: {len(training_results)}個のモデル")
            return {
                "success": True,
                "models_trained": len(training_results),
                "training_results": training_results,
                "feature_count": len(features.columns),
                "data_points": len(historical_data),
            }

        except Exception as e:
            logger.error(f"機械学習モデル訓練エラー: {e}")
            return {"error": str(e)}

    def _train_meta_learner(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """メタラーナーを訓練"""
        try:
            meta_features_list = []
            meta_targets_list = []
            volume_data = (
                historical_data["Volume"]
                if "Volume" in historical_data.columns
                else None
            )

            # 各データポイントでメタ特徴量を生成
            for i in range(
                50, len(historical_data) - 10
            ):  # 十分な履歴とフォワードルッキングを確保
                try:
                    subset_data = historical_data.iloc[: i + 1]
                    future_return = (
                        historical_data["Close"].iloc[i + 5]
                        / historical_data["Close"].iloc[i]
                        - 1
                    )

                    # 各基本モデルの予測を特徴量として使用
                    model_predictions = []
                    for model_name in [
                        "return_predictor",
                        "direction_predictor",
                        "volatility_predictor",
                    ]:
                        if model_name in self.ml_manager.list_models():
                            try:
                                subset_features = (
                                    self.feature_engineer.generate_all_features(
                                        price_data=subset_data,
                                        volume_data=(
                                            volume_data.iloc[: i + 1]
                                            if volume_data is not None
                                            else None
                                        ),
                                    )
                                )
                                if not subset_features.empty:
                                    pred = self.ml_manager.predict(
                                        model_name, subset_features.tail(1)
                                    )
                                    model_predictions.append(
                                        pred[0] if len(pred) > 0 else 0.0
                                    )
                                else:
                                    model_predictions.append(0.0)
                            except Exception:
                                model_predictions.append(0.0)

                    if len(model_predictions) >= 2:
                        meta_features_list.append(model_predictions)
                        meta_targets_list.append(future_return)

                except Exception:
                    continue

            if len(meta_features_list) >= 50:
                meta_X = pd.DataFrame(meta_features_list)
                meta_y = pd.Series(meta_targets_list)

                meta_result = self.ml_manager.train_model(
                    "meta_learner", meta_X, meta_y
                )

                try:
                    self.ml_manager.save_model("meta_learner")
                    logger.info("メタラーナーを保存しました")
                except Exception as e:
                    logger.warning(f"メタラーナーの保存に失敗: {e}")

                return meta_result
            else:
                return {"error": "メタラーナー用データが不足"}

        except Exception as e:
            logger.error(f"メタラーナー訓練エラー: {e}")
            return {"error": str(e)}

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