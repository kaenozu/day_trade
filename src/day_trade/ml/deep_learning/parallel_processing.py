#!/usr/bin/env python3
"""
深層学習統合システム - 並列処理
Phase F: 次世代機能拡張フェーズ

Monte Carlo DropoutとPermutation Importanceの並列化処理
"""

import warnings
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Issue #695対応: 並列処理ライブラリ
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib未インストール - 並列処理が制限されます", ImportWarning)

from .model_types import PredictionResult, UncertaintyEstimate

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ParallelProcessingMixin:
    """並列処理機能のMixinクラス"""

    def _optimize_mc_dropout_parallel_jobs(self, num_samples: int, n_jobs: Optional[int] = None) -> int:
        """
        Issue #695対応: Monte Carlo Dropout並列化の最適化

        Args:
            num_samples: Monte Carloサンプル数
            n_jobs: ユーザー指定並列ジョブ数

        Returns:
            最適化された並列ジョブ数
        """
        try:
            cpu_count = mp.cpu_count()

            if n_jobs is not None:
                if n_jobs == -1:
                    return cpu_count
                elif n_jobs > 0:
                    return min(n_jobs, cpu_count)
                else:
                    return 1

            # 自動最適化
            if num_samples < 10:
                return 1  # 少数サンプルは直列が効率的
            elif num_samples < 50:
                return min(2, cpu_count)
            elif num_samples < 200:
                return min(4, cpu_count)
            else:
                # 大量サンプル：CPUの75%を使用
                return min(max(2, int(cpu_count * 0.75)), cpu_count)

        except Exception as e:
            logger.warning(f"Monte Carlo Dropout並列化設定エラー: {e}")
            return 1

    def _optimize_permutation_parallel_jobs(self, num_features: int, n_jobs: Optional[int] = None) -> int:
        """
        Issue #695対応: Permutation Importance並列化の最適化

        Args:
            num_features: 特徴量数
            n_jobs: ユーザー指定並列ジョブ数

        Returns:
            最適化された並列ジョブ数
        """
        try:
            cpu_count = mp.cpu_count()

            if n_jobs is not None:
                if n_jobs == -1:
                    return min(num_features, cpu_count)
                elif n_jobs > 0:
                    return min(n_jobs, cpu_count, num_features)
                else:
                    return 1

            # 自動最適化
            if num_features < 3:
                return 1  # 少数特徴量は直列が効率的
            elif num_features <= cpu_count:
                return num_features  # 特徴量数がCPU数以下なら全並列
            else:
                # 特徴量数が多い場合は適度な並列化
                return min(cpu_count, max(2, cpu_count // 2))

        except Exception as e:
            logger.warning(f"Permutation Importance並列化設定エラー: {e}")
            return 1

    def _parallel_monte_carlo_dropout(self, data: pd.DataFrame, num_samples: int, n_jobs: int) -> List[np.ndarray]:
        """
        Issue #695対応: 並列化Monte Carlo Dropout実行

        Args:
            data: 予測対象データ
            num_samples: Monte Carloサンプル数
            n_jobs: 並列ジョブ数

        Returns:
            予測結果リスト
        """
        def _single_prediction(sample_idx: int) -> np.ndarray:
            """単一予測実行（並列処理用）"""
            np.random.seed(sample_idx)  # 再現性のためのシード設定
            pred_result = self.predict(data)
            return pred_result.predictions

        # joblib並列実行
        predictions_list = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_single_prediction)(i) for i in range(num_samples)
        )

        return predictions_list

    def _parallel_permutation_importance(
        self, X: np.ndarray, baseline_pred: np.ndarray, baseline_error: float,
        feature_names: List[str], n_jobs: int
    ) -> Dict[str, float]:
        """
        Issue #695対応: 並列化Permutation Importance実行

        Args:
            X: 入力データ
            baseline_pred: ベースライン予測
            baseline_error: ベースラインエラー
            feature_names: 特徴量名リスト
            n_jobs: 並列ジョブ数

        Returns:
            特徴量重要度辞書
        """
        def _calculate_feature_importance(feature_idx_and_name: Tuple[int, str]) -> Tuple[str, float]:
            """単一特徴量重要度計算（並列処理用）"""
            feature_idx, feature_name = feature_idx_and_name

            # 特徴量をシャッフル
            X_shuffled = X.copy()
            if len(X_shuffled.shape) == 3:  # (samples, sequence, features)
                np.random.seed(feature_idx)  # 再現性のため
                X_shuffled[:, :, feature_idx] = np.random.permutation(X_shuffled[:, :, feature_idx])

            shuffled_pred = self._predict_internal(X_shuffled)
            shuffled_error = np.mean((shuffled_pred - baseline_pred) ** 2)

            importance = max(0, shuffled_error - baseline_error) / (baseline_error + 1e-8)
            return feature_name, float(importance)

        # joblib並列実行
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_calculate_feature_importance)((i, name))
            for i, name in enumerate(feature_names)
        )

        return dict(results)

    def predict_with_uncertainty(
        self, data: pd.DataFrame, num_samples: int = 100, n_jobs: Optional[int] = None
    ) -> PredictionResult:
        """
        不確実性付き予測

        Issue #695対応: Monte Carlo Dropout並列化

        Args:
            data: 予測対象データ
            num_samples: Monte Carloサンプル数
            n_jobs: 並列ジョブ数（None=自動、1=直列、-1=全CPU使用）
        """
        import time
        start_time = time.time()

        # 並列化設定の最適化
        optimal_n_jobs = self._optimize_mc_dropout_parallel_jobs(num_samples, n_jobs)

        if optimal_n_jobs > 1 and JOBLIB_AVAILABLE:
            # Issue #695対応: joblib並列化によるMonte Carlo Dropout
            logger.info(f"Monte Carlo Dropout並列化実行: {num_samples}サンプル x {optimal_n_jobs}並列")
            predictions_list = self._parallel_monte_carlo_dropout(data, num_samples, optimal_n_jobs)
        else:
            # 直列実行（フォールバック）
            if optimal_n_jobs > 1:
                logger.warning("joblib未使用のため直列実行にフォールバック")

            predictions_list = []
            for i in range(num_samples):
                if i % 20 == 0:
                    logger.debug(f"Monte Carlo Dropout進捗: {i}/{num_samples}")
                pred_result = self.predict(data)
                predictions_list.append(pred_result.predictions)

        # 統計計算
        all_predictions = np.array(predictions_list)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

        # 不確実性推定
        uncertainty = UncertaintyEstimate(
            mean=float(np.mean(std_pred)),
            std=float(np.std(std_pred)),
            lower_bound=mean_pred - 1.96 * std_pred,
            upper_bound=mean_pred + 1.96 * std_pred,
            epistemic=float(np.mean(std_pred)),  # モデル不確実性
            aleatoric=float(np.std(std_pred)),  # データ不確実性
        )

        processing_time = time.time() - start_time
        logger.info(f"Monte Carlo Dropout完了: {processing_time:.2f}秒 ({num_samples}サンプル)")

        return PredictionResult(
            predictions=mean_pred,
            confidence=1.0 - std_pred / (np.abs(mean_pred) + 1e-8),
            uncertainty=uncertainty,
            model_used=self.__class__.__name__,
            prediction_time=processing_time
        )