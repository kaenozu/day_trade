#!/usr/bin/env python3
"""
深層学習統合システム - 基底モデル
Phase F: 次世代機能拡張フェーズ

深層学習モデルの抽象基底クラス
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .model_types import ModelConfig, ModelTrainingResult, PredictionResult
from .data_preparation import DataPreparationMixin
from .parallel_processing import ParallelProcessingMixin
from .utils import calculate_accuracy

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BaseDeepLearningModel(ABC, DataPreparationMixin, ParallelProcessingMixin):
    """深層学習モデル基底クラス"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """モデル構築"""
        pass

    def train(self, data: pd.DataFrame) -> ModelTrainingResult:
        """データフレームからモデル訓練"""
        X, y = self.prepare_training_data(data)

        # モデル構築
        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape)

        # 訓練実行
        start_time = time.time()
        training_result = self._train_internal(X, y)
        training_time = time.time() - start_time

        return ModelTrainingResult(
            final_loss=training_result.get("final_loss", 0.0),
            best_loss=training_result.get("best_loss", 0.0),
            epochs_run=training_result.get("epochs_run", self.config.epochs),
            training_time=training_time,
            validation_metrics=training_result.get("validation_metrics", {}),
            convergence_achieved=training_result.get("convergence_achieved", True),
        )

    @abstractmethod
    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """内部訓練メソッド"""
        pass

    def prepare_training_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データ準備"""
        return self.prepare_data(data)

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """予測実行"""
        X, _ = self.prepare_data(data)

        if not self.is_trained or self.model is None:
            raise ValueError("モデルが訓練されていません")

        start_time = time.time()
        predictions = self._predict_internal(X)
        prediction_time = time.time() - start_time

        # 信頼度スコア計算（簡易版）
        confidence = np.ones(len(predictions)) * 0.8

        return PredictionResult(
            predictions=predictions,
            confidence=confidence,
            prediction_time=prediction_time,
            model_used=self.__class__.__name__,
            metrics={"mae": 0.0},  # 実際の実装では適切な計算を行う
        )

    @abstractmethod
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """内部予測メソッド"""
        pass

    def get_feature_importance(self, data: Optional[pd.DataFrame] = None, n_jobs: Optional[int] = None) -> Dict[str, float]:
        """
        特徴量重要度分析

        Issue #695対応: Permutation Importance並列化

        Args:
            data: 分析対象データ
            n_jobs: 並列ジョブ数（None=自動、1=直列、-1=全CPU使用）
        """
        # Issue #495対応: データ提供チェック
        if data is None:
            if not self.is_trained:
                logger.debug(f"{self.__class__.__name__}: 未学習かつデータ未提供のため特徴量重要度を取得できません")
                return {}
            else:
                logger.warning(f"{self.__class__.__name__}: データ未提供のため特徴量重要度を取得できません（DeepLearningモデルではデータが必要）")
                return {}

        start_time = time.time()

        try:
            # Permutation Importanceによる特徴量重要度計算
            X, y = self.prepare_data(data)
            baseline_pred = self._predict_internal(X)
        except Exception as e:
            logger.warning(f"特徴量重要度計算用データ準備エラー: {e}")
            return {}
        baseline_error = np.mean((baseline_pred - y) ** 2) if y is not None else 0.0

        feature_names = ["Open", "High", "Low", "Close", "Volume"][: X.shape[-1]]

        # 並列化設定の最適化
        optimal_n_jobs = self._optimize_permutation_parallel_jobs(len(feature_names), n_jobs)

        # 並列処理ライブラリのチェック
        try:
            from joblib import Parallel, delayed
            JOBLIB_AVAILABLE = True
        except ImportError:
            JOBLIB_AVAILABLE = False

        if optimal_n_jobs > 1 and JOBLIB_AVAILABLE:
            # Issue #695対応: joblib並列化によるPermutation Importance
            logger.info(f"Permutation Importance並列化実行: {len(feature_names)}特徴量 x {optimal_n_jobs}並列")
            feature_importance = self._parallel_permutation_importance(
                X, baseline_pred, baseline_error, feature_names, optimal_n_jobs
            )
        else:
            # 直列実行（フォールバック）
            if optimal_n_jobs > 1:
                logger.warning("joblib未使用のため直列実行にフォールバック")

            feature_importance = {}
            for i, feature_name in enumerate(feature_names):
                # 特徴量をシャッフル
                X_shuffled = X.copy()
                if len(X_shuffled.shape) == 3:  # (samples, sequence, features)
                    X_shuffled[:, :, i] = np.random.permutation(X_shuffled[:, :, i])

                shuffled_pred = self._predict_internal(X_shuffled)
                shuffled_error = np.mean((shuffled_pred - baseline_pred) ** 2)

                importance = max(0, shuffled_error - baseline_error) / (
                    baseline_error + 1e-8
                )
                feature_importance[feature_name] = float(importance)

        # 正規化
        total_importance = sum(feature_importance.values()) or 1.0
        for key in feature_importance:
            feature_importance[key] /= total_importance

        processing_time = time.time() - start_time
        logger.info(f"Permutation Importance完了: {processing_time:.2f}秒 ({len(feature_names)}特徴量)")

        return feature_importance

    def has_feature_importance(self) -> bool:
        """
        Issue #495対応: DeepLearningモデルは特徴量重要度を提供可能（データが必要）

        Returns:
            学習済みの場合True（実際にはデータが必要だが、提供可能性としてTrue）
        """
        return self.is_trained