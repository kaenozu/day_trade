#!/usr/bin/env python3
"""
深層学習統合システム - データ準備
Phase F: 次世代機能拡張フェーズ

深層学習モデル用データ前処理とシーケンス生成
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DataPreparationMixin:
    """データ準備機能のMixinクラス"""

    def prepare_data(
        self, data: pd.DataFrame, target_column: str = "Close", use_optimized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列データ準備

        Issue #694対応: データ準備最適化

        Args:
            data: 入力データ
            target_column: ターゲット列名
            use_optimized: 最適化版を使用するかどうか
        """
        start_time = time.time()

        # OHLCV特徴量使用
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in feature_columns if col in data.columns]

        if not available_columns:
            raise ValueError("必要な価格データ列が見つかりません")

        # データ正規化
        features = data[available_columns].values.astype(np.float32)  # Issue #694: float32で最適化

        # Issue #694対応: ベクトル化された正規化
        features_mean = np.mean(features, axis=0, keepdims=True)
        features_std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - features_mean) / features_std

        if use_optimized:
            # Issue #694対応: stride_tricks最適化版
            X, y = self._prepare_data_optimized(features, data, target_column)
            optimization_type = "最適化版"
        else:
            # 従来版（デバッグ・比較用）
            X, y = self._prepare_data_legacy(features, data, target_column)
            optimization_type = "従来版"

        preparation_time = time.time() - start_time
        logger.info(f"データ準備完了 ({optimization_type}): {preparation_time:.3f}秒, 形状: X{X.shape}, y{y.shape}")

        return X, y

    def _prepare_data_optimized(
        self, features: np.ndarray, data: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Issue #694対応: stride_tricksによるメモリ効率化データ準備

        Args:
            features: 正規化済み特徴量データ
            data: 元データ
            target_column: ターゲット列名

        Returns:
            X, y のタプル
        """
        sequence_length = self.config.sequence_length
        prediction_horizon = self.config.prediction_horizon

        # 有効なシーケンス数を計算
        n_sequences = len(features) - sequence_length - prediction_horizon + 1

        if n_sequences <= 0:
            raise ValueError(f"データが不足: {len(features)}サンプル < {sequence_length + prediction_horizon}必要")

        try:
            # Issue #694対応: sliding_window_viewによるメモリ効率化
            # NumPy 1.20.0+ でsliding_window_viewが利用可能
            X = sliding_window_view(features, window_shape=sequence_length, axis=0)
            X = X[:n_sequences]  # 有効な範囲のみ切り出し

        except (ImportError, AttributeError):
            # sliding_window_view未利用環境でのフォールバック
            logger.warning("sliding_window_view未利用、手動stride実装を使用")
            X = self._manual_sliding_window(features, sequence_length, n_sequences)

        # ターゲットデータ準備
        if target_column in data.columns:
            target_data = data[target_column].values.astype(np.float32)
            # 正規化されたターゲットデータを使用
            target_mean = np.mean(target_data)
            target_std = np.std(target_data) + 1e-8
            target_data = (target_data - target_mean) / target_std

            # Issue #694対応: ベクトル化されたターゲット抽出
            if prediction_horizon == 1:
                y = target_data[sequence_length:sequence_length + n_sequences]
            else:
                # 複数ステップ予測の場合
                y = np.array([
                    target_data[i + sequence_length:i + sequence_length + prediction_horizon]
                    for i in range(n_sequences)
                ])
        else:
            # ターゲット列がない場合は終値を使用
            close_idx = -1  # 通常、終値は最後の列
            if prediction_horizon == 1:
                y = features[sequence_length:sequence_length + n_sequences, close_idx]
            else:
                y = np.array([
                    features[i + sequence_length:i + sequence_length + prediction_horizon, close_idx]
                    for i in range(n_sequences)
                ])

        return X, y

    def _prepare_data_legacy(
        self, features: np.ndarray, data: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Issue #694対応: 従来版データ準備（比較・デバッグ用）

        Args:
            features: 正規化済み特徴量データ
            data: 元データ
            target_column: ターゲット列名

        Returns:
            X, y のタプル
        """
        # 従来のシーケンスデータ作成
        X, y = [], []
        for i in range(
            len(features)
            - self.config.sequence_length
            - self.config.prediction_horizon
            + 1
        ):
            X.append(features[i : i + self.config.sequence_length])
            if target_column in data.columns:
                target_data = data[target_column].values.astype(np.float32)
                # 正規化
                target_mean = np.mean(target_data)
                target_std = np.std(target_data) + 1e-8
                target_data = (target_data - target_mean) / target_std

                y.append(
                    target_data[
                        i + self.config.sequence_length : i
                        + self.config.sequence_length
                        + self.config.prediction_horizon
                    ]
                )
            else:
                # ターゲット列がない場合は終値を使用
                y.append(
                    features[
                        i + self.config.sequence_length : i
                        + self.config.sequence_length
                        + self.config.prediction_horizon,
                        -1,
                    ]
                )

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _manual_sliding_window(
        self, features: np.ndarray, sequence_length: int, n_sequences: int
    ) -> np.ndarray:
        """
        Issue #694対応: 手動sliding window実装（フォールバック用）

        Args:
            features: 特徴量データ
            sequence_length: シーケンス長
            n_sequences: 生成するシーケンス数

        Returns:
            スライディングウィンドウデータ
        """
        n_features = features.shape[1]

        # 事前にサイズを決めてNumPy配列を確保
        X = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)

        for i in range(n_sequences):
            X[i] = features[i:i + sequence_length]

        return X