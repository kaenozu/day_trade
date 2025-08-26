"""
機械学習モデル訓練モジュール
"""

from typing import Any, Dict

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_ml_training")


class MLTrainingManager:
    """機械学習モデル訓練管理クラス"""

    def __init__(self, ml_model_manager):
        """
        Args:
            ml_model_manager: MLModelManagerインスタンス
        """
        self.ml_model_manager = ml_model_manager

    def train_ml_models(
        self, historical_data: pd.DataFrame, retrain: bool = False
    ) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        try:
            if not self.ml_model_manager.is_available():
                return {"error": "機械学習機能が無効です"}

            if len(historical_data) < 200:
                return {"error": "訓練に十分なデータがありません（最低200日必要）"}

            logger.info("機械学習モデルの訓練を開始")

            # 特徴量生成
            feature_engineer = self.ml_model_manager.get_feature_engineer()
            ml_manager = self.ml_model_manager.get_core_ml_manager()
            
            volume_data = (
                historical_data["Volume"]
                if "Volume" in historical_data.columns
                else None
            )
            features = feature_engineer.generate_all_features(
                price_data=historical_data, volume_data=volume_data
            )

            if features.empty:
                return {"error": "特徴量生成に失敗しました"}

            # ターゲット変数生成
            targets = self._create_target_variables(historical_data)

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
                    if model_name not in ml_manager.list_models():
                        logger.warning(
                            f"モデル {model_name} が存在しません。スキップします。"
                        )
                        continue

                    # 既に訓練済みでretrainがFalseの場合はスキップ
                    if not retrain and ml_manager.models[model_name].is_fitted:
                        logger.info(f"モデル {model_name} は既に訓練済みです")
                        continue

                    # モデル訓練
                    logger.info(f"モデル {model_name} を訓練中...")
                    result = ml_manager.train_model(model_name, X_train, y_train)
                    training_results[model_name] = result

                    # モデル保存
                    try:
                        ml_manager.save_model(model_name)
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

    def _create_target_variables(self, data: pd.DataFrame, prediction_horizon: int = 5) -> Dict[str, pd.Series]:
        """ターゲット変数を生成"""
        try:
            from ..feature_engineering import create_target_variables
            return create_target_variables(data, prediction_horizon)
        except ImportError:
            # フォールバック実装
            targets = {}
            
            # 将来リターン
            future_returns = data['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            targets['future_returns'] = future_returns.dropna()
            
            # 将来方向（上昇/下降）
            future_direction = (future_returns > 0).astype(int)
            targets['future_direction'] = future_direction.dropna()
            
            # 高ボラティリティ
            returns = data['Close'].pct_change()
            rolling_vol = returns.rolling(prediction_horizon).std()
            high_vol_threshold = rolling_vol.quantile(0.7)
            future_high_vol = (rolling_vol.shift(-prediction_horizon) > high_vol_threshold).astype(int)
            targets['future_high_volatility'] = future_high_vol.dropna()
            
            return targets

    def _train_meta_learner(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """メタラーナーを訓練"""
        try:
            feature_engineer = self.ml_model_manager.get_feature_engineer()
            ml_manager = self.ml_model_manager.get_core_ml_manager()
            
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
                        if model_name in ml_manager.list_models():
                            try:
                                subset_features = (
                                    feature_engineer.generate_all_features(
                                        price_data=subset_data,
                                        volume_data=(
                                            volume_data.iloc[: i + 1]
                                            if volume_data is not None
                                            else None
                                        ),
                                    )
                                )
                                if not subset_features.empty:
                                    pred = ml_manager.predict(
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

                meta_result = ml_manager.train_model(
                    "meta_learner", meta_X, meta_y
                )

                try:
                    ml_manager.save_model("meta_learner")
                    logger.info("メタラーナーを保存しました")
                except Exception as e:
                    logger.warning(f"メタラーナーの保存に失敗: {e}")

                return meta_result
            else:
                return {"error": "メタラーナー用データが不足"}

        except Exception as e:
            logger.error(f"メタラーナー訓練エラー: {e}")
            return {"error": str(e)}

    def validate_models(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """モデルを検証"""
        try:
            if not self.ml_model_manager.is_available():
                return {"error": "機械学習機能が無効です"}

            if len(validation_data) < 50:
                return {"error": "検証に十分なデータがありません"}

            feature_engineer = self.ml_model_manager.get_feature_engineer()
            ml_manager = self.ml_model_manager.get_core_ml_manager()
            
            # 特徴量生成
            volume_data = validation_data["Volume"] if "Volume" in validation_data.columns else None
            features = feature_engineer.generate_all_features(
                price_data=validation_data, volume_data=volume_data
            )

            if features.empty:
                return {"error": "検証用特徴量生成に失敗しました"}

            # ターゲット変数生成
            targets = self._create_target_variables(validation_data)

            validation_results = {}

            for model_name in ml_manager.list_models():
                try:
                    model = ml_manager.models[model_name]
                    if not model.is_fitted:
                        continue

                    # 対応するターゲット変数を見つける
                    target_mapping = {
                        "return_predictor": "future_returns",
                        "direction_predictor": "future_direction", 
                        "volatility_predictor": "future_high_volatility"
                    }
                    
                    target_name = target_mapping.get(model_name)
                    if not target_name or target_name not in targets:
                        continue

                    # 検証実行
                    common_index = features.index.intersection(targets[target_name].index)
                    if len(common_index) < 10:
                        continue

                    X_val = features.loc[common_index]
                    y_val = targets[target_name].loc[common_index]

                    # 予測と評価
                    predictions = ml_manager.predict(model_name, X_val)
                    
                    # 評価メトリクス計算
                    if model.config.task_type == "regression":
                        from sklearn.metrics import mean_squared_error, r2_score
                        mse = mean_squared_error(y_val, predictions)
                        r2 = r2_score(y_val, predictions)
                        validation_results[model_name] = {
                            "mse": mse,
                            "r2_score": r2,
                            "sample_count": len(predictions)
                        }
                    else:  # classification
                        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                        accuracy = accuracy_score(y_val, (predictions > 0.5).astype(int))
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_val, (predictions > 0.5).astype(int), average='weighted'
                        )
                        validation_results[model_name] = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "sample_count": len(predictions)
                        }

                except Exception as e:
                    logger.warning(f"モデル {model_name} の検証でエラー: {e}")
                    validation_results[model_name] = {"error": str(e)}

            return {
                "success": True,
                "validation_results": validation_results,
                "models_validated": len(validation_results)
            }

        except Exception as e:
            logger.error(f"モデル検証エラー: {e}")
            return {"error": str(e)}