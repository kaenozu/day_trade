"""
機械学習予測生成モジュール
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_ml_prediction")


class MLPredictionGenerator:
    """機械学習予測生成クラス"""

    def __init__(self, ml_model_manager):
        """
        Args:
            ml_model_manager: MLModelManagerインスタンス
        """
        self.ml_model_manager = ml_model_manager
        self.predictions_history = []

    def generate_ml_predictions(
        self, df: pd.DataFrame, indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """機械学習予測を生成"""
        try:
            if not self.ml_model_manager.is_available():
                return None, None

            if len(df) < 50:
                return None, None

            # 高度な特徴量を生成
            volume_data = df["Volume"] if "Volume" in df.columns else None
            feature_engineer = self.ml_model_manager.get_feature_engineer()
            ml_manager = self.ml_model_manager.get_core_ml_manager()
            
            features = feature_engineer.generate_all_features(
                price_data=df, volume_data=volume_data
            )

            if features.empty:
                return None, None

            # 最新の特徴量を取得
            latest_features = features.tail(1)

            predictions = {}
            feature_importance = {}

            # 各モデルで予測を実行
            for model_name in ml_manager.list_models():
                try:
                    model = ml_manager.models[model_name]
                    if model.is_fitted:
                        pred = ml_manager.predict(model_name, latest_features)
                        predictions[model_name] = (
                            float(pred[0]) if len(pred) > 0 else 0.0
                        )

                        # 特徴量重要度を取得
                        importance = model._get_feature_importance()
                        if importance:
                            feature_importance[model_name] = importance

                except Exception as e:
                    logger.warning(f"モデル {model_name} の予測でエラー: {e}")

            # 予測履歴に追加
            if predictions:
                prediction_record = {
                    'timestamp': pd.Timestamp.now(),
                    'predictions': predictions.copy(),
                    'features_shape': latest_features.shape
                }
                self.predictions_history.append(prediction_record)
                
                # 履歴サイズを制限（最新の100件のみ保持）
                if len(self.predictions_history) > 100:
                    self.predictions_history = self.predictions_history[-100:]

            return predictions, feature_importance

        except Exception as e:
            logger.error(f"機械学習予測エラー: {e}")
            return None, None

    def get_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """予測結果の信頼度を計算"""
        try:
            if not predictions:
                return 0.0

            # モデル間の予測一致度を計算
            values = list(predictions.values())
            if len(values) <= 1:
                return 50.0  # デフォルト信頼度

            # 標準偏差ベースの信頼度計算
            import numpy as np
            std_dev = np.std(values)
            mean_abs_value = np.mean(np.abs(values))
            
            if mean_abs_value == 0:
                return 50.0
            
            # 相対標準偏差を使用（低いほど信頼度が高い）
            relative_std = std_dev / (mean_abs_value + 1e-8)
            confidence = max(10.0, 90.0 - relative_std * 100.0)
            
            return min(90.0, confidence)

        except Exception as e:
            logger.warning(f"予測信頼度計算エラー: {e}")
            return 50.0

    def get_ensemble_prediction(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """アンサンブル予測を計算"""
        try:
            if not predictions:
                return {}

            # 簡単な平均アンサンブル
            import numpy as np
            values = list(predictions.values())
            
            ensemble_result = {
                'mean_prediction': np.mean(values),
                'median_prediction': np.median(values),
                'std_prediction': np.std(values),
                'min_prediction': np.min(values),
                'max_prediction': np.max(values),
                'model_count': len(values),
                'confidence': self.get_prediction_confidence(predictions)
            }
            
            return ensemble_result

        except Exception as e:
            logger.error(f"アンサンブル予測計算エラー: {e}")
            return {}

    def get_predictions_history(self, limit: int = 10) -> list:
        """予測履歴を取得"""
        return self.predictions_history[-limit:] if self.predictions_history else []

    def clear_predictions_history(self) -> None:
        """予測履歴をクリア"""
        self.predictions_history.clear()

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """モデルパフォーマンスサマリーを取得"""
        try:
            if not self.predictions_history:
                return {}

            # 最近の予測結果からパフォーマンス統計を計算
            recent_predictions = self.predictions_history[-20:]  # 最新20件
            
            model_stats = {}
            for record in recent_predictions:
                for model_name, pred_value in record['predictions'].items():
                    if model_name not in model_stats:
                        model_stats[model_name] = []
                    model_stats[model_name].append(pred_value)
            
            # 各モデルの統計を計算
            summary = {}
            for model_name, values in model_stats.items():
                import numpy as np
                summary[model_name] = {
                    'mean_prediction': np.mean(values),
                    'std_prediction': np.std(values),
                    'prediction_count': len(values),
                    'last_prediction': values[-1] if values else None
                }
                
            return summary

        except Exception as e:
            logger.error(f"モデルパフォーマンスサマリー取得エラー: {e}")
            return {}