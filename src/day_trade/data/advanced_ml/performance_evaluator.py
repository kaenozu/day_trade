#!/usr/bin/env python3
"""
Performance Evaluator Module

モデル性能評価と推論時間測定機能
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class PerformanceEvaluator:
    """モデル性能評価クラス"""

    def __init__(self):
        self.performance_metrics = {
            "accuracy_history": [],
            "inference_times": [],
            "mae_history": [],
            "rmse_history": [],
        }

    def evaluate_model_performance(
        self, hybrid_model, data: pd.DataFrame, target_column: str
    ) -> Dict[str, float]:
        """モデル性能評価"""
        try:
            # テストデータで予測実行
            test_data = data.tail(min(100, len(data) // 4))  # 最後の25%をテスト用

            prediction_result = hybrid_model.predict(test_data)
            predictions = prediction_result.predictions

            # 実際の値取得
            X, y_true = hybrid_model.prepare_data(test_data, target_column)

            if len(y_true) > 0 and len(predictions) > 0:
                # 長さ調整
                min_len = min(len(y_true), len(predictions))
                y_true = y_true[:min_len]
                predictions = predictions[:min_len]

                # メトリクス計算
                mae = np.mean(np.abs(y_true - predictions))
                rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
                mape = (
                    np.mean(
                        np.abs(
                            (y_true - predictions)
                            / np.where(y_true != 0, y_true, 1e-8)
                        )
                    )
                    * 100
                )
                accuracy = max(0, 100 - mape) / 100

                # 推論時間測定 - Issue #707対応最適化版
                avg_inference_time = self._measure_inference_time_optimized(
                    hybrid_model, test_data
                )
                if avg_inference_time is None:
                    # フォールバック: 従来方式
                    inference_times = []
                    for _ in range(10):  # 10回測定
                        start = time.time()
                        _ = hybrid_model.predict(test_data.tail(10))
                        inference_times.append((time.time() - start) * 1000)  # ms変換
                    avg_inference_time = np.mean(inference_times)

                # メトリクス履歴更新
                self.performance_metrics["accuracy_history"].append(accuracy)
                self.performance_metrics["mae_history"].append(mae)
                self.performance_metrics["rmse_history"].append(rmse)

                return {
                    "accuracy": accuracy,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "avg_inference_time": avg_inference_time,
                    "test_samples": min_len,
                }
            else:
                logger.warning("性能評価用データが不足")
                return {
                    "accuracy": 0.0,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mape": 100.0,
                    "avg_inference_time": 1000.0,
                    "test_samples": 0,
                }

        except Exception as e:
            logger.error(f"性能評価エラー: {e}")
            return {
                "accuracy": 0.0,
                "mae": 1.0,
                "rmse": 1.0,
                "mape": 100.0,
                "avg_inference_time": 1000.0,
                "test_samples": 0,
                "error": str(e),
            }

    def _measure_inference_time_optimized(
        self, hybrid_model, test_data: pd.DataFrame, n_iterations: int = 10
    ) -> Optional[float]:
        """
        最適化された推論時間測定 - Issue #707対応

        Args:
            hybrid_model: ハイブリッドモデル
            test_data: テストデータ
            n_iterations: 測定回数

        Returns:
            平均推論時間(ms)、エラー時はNone
        """
        try:
            test_sample = test_data.tail(10)

            def single_inference():
                """単一推論実行"""
                start = time.time()
                try:
                    _ = hybrid_model.predict(test_sample)
                    return (time.time() - start) * 1000  # ms変換
                except Exception as e:
                    logger.warning(f"推論時間測定エラー: {e}")
                    return None

            # 並列推論時間測定（I/O待機を活用）
            with ThreadPoolExecutor(max_workers=min(4, n_iterations)) as executor:
                futures = [
                    executor.submit(single_inference) for _ in range(n_iterations)
                ]

                inference_times = []
                for future in futures:
                    result = future.result()
                    if result is not None:
                        inference_times.append(result)

            if inference_times:
                avg_time = np.mean(inference_times)
                logger.info(
                    f"並列推論時間測定完了: {len(inference_times)}回測定、"
                    f"平均{avg_time:.2f}ms"
                )
                return avg_time
            else:
                return None

        except Exception as e:
            logger.warning(f"最適化推論時間測定失敗: {e}")
            return None

    def get_performance_summary(self) -> Dict[str, float]:
        """性能サマリー取得"""
        return {
            "total_predictions": len(self.performance_metrics["inference_times"]),
            "avg_accuracy": (
                np.mean(self.performance_metrics["accuracy_history"])
                if self.performance_metrics["accuracy_history"]
                else 0
            ),
            "avg_mae": (
                np.mean(self.performance_metrics["mae_history"])
                if self.performance_metrics["mae_history"]
                else 0
            ),
            "avg_rmse": (
                np.mean(self.performance_metrics["rmse_history"])
                if self.performance_metrics["rmse_history"]
                else 0
            ),
            "avg_inference_time_ms": (
                np.mean(self.performance_metrics["inference_times"])
                if self.performance_metrics["inference_times"]
                else 0
            ),
        }

    def record_inference_time(self, inference_time_ms: float):
        """推論時間を記録"""
        self.performance_metrics["inference_times"].append(inference_time_ms)