#!/usr/bin/env python3
"""
Performance evaluation and statistics module for Dynamic Weighting System

This module handles model performance evaluation and statistical analysis
with improved modularity and data validation separation.
"""

import time
from typing import Dict, List, Any, Union, Optional
import numpy as np
from collections import deque

from .core import WeightingState, DynamicWeightingConfig, PerformanceWindow
from .data_validator import DataValidator
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class PerformanceManager:
    """
    パフォーマンス管理クラス

    モデルの予測性能の追跡、評価、統計分析を担当します。
    データ検証機能は DataValidator に委譲されています。
    """

    def __init__(self, model_names: List[str], config: DynamicWeightingConfig):
        """
        初期化

        Args:
            model_names: モデル名のリスト
            config: システム設定
        """
        self.model_names = model_names
        self.config = config
        
        # WeightingStateを初期化
        self.state = WeightingState()
        self._initialize_data_structures()
        
        # データバリデータ
        self.data_validator = DataValidator()

    def _initialize_data_structures(self):
        """データ構造を初期化"""
        # パフォーマンス履歴の初期化
        self.state.performance_windows = {
            name: deque(maxlen=self.config.window_size) 
            for name in self.model_names
        }
        self.state.recent_predictions = {
            name: deque(maxlen=self.config.window_size)
            for name in self.model_names
        }
        self.state.recent_actuals = deque(maxlen=self.config.window_size)
        self.state.recent_timestamps = deque(maxlen=self.config.window_size)

    def update_performance(
        self,
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]],
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Issue #475対応: パフォーマンス更新（改善版）

        予測値・実際値の処理を統一し、冗長なチェックを排除

        Args:
            predictions: モデル別予測値（単一値または配列）
            actuals: 実際の値（単一値または配列）
            timestamp: タイムスタンプ（未指定時は現在時刻）

        Returns:
            更新成功可否

        Raises:
            ValueError: 予測値と実際値の次元が一致しない場合
            TypeError: サポートされていない型の場合
        """
        try:
            if timestamp is None:
                timestamp = int(time.time())

            # Issue #475対応: 一貫した配列変換処理（DataValidatorに委譲）
            normalized_actuals = self.data_validator.normalize_to_array(
                actuals, "actuals"
            )

            # 予測値の正規化と記録
            for model_name, pred in predictions.items():
                if model_name in self.state.recent_predictions:
                    try:
                        normalized_pred = self.data_validator.normalize_to_array(
                            pred, f"predictions[{model_name}]"
                        )

                        # 次元一致性チェック
                        if len(normalized_pred) != len(normalized_actuals):
                            if len(normalized_pred) == 1:
                                # 単一予測値を実際値の数だけ複製
                                normalized_pred = np.repeat(
                                    normalized_pred[0], len(normalized_actuals)
                                )
                            elif len(normalized_actuals) == 1:
                                # 実際値が単一の場合は予測値の最初の値を使用
                                normalized_pred = normalized_pred[:1]
                            else:
                                raise ValueError(
                                    f"{model_name}: 予測値の次元({len(normalized_pred)}) != "
                                    f"実際値の次元({len(normalized_actuals)})"
                                )

                        # 予測値をキューに追加
                        for pred_val in normalized_pred:
                            self.state.recent_predictions[model_name].append(
                                float(pred_val)
                            )

                    except Exception as e:
                        logger.warning(f"{model_name}の予測値処理でエラー: {e}")
                        continue

            # 実際値とタイムスタンプの記録
            for actual_val in normalized_actuals:
                self.state.recent_actuals.append(float(actual_val))
                self.state.recent_timestamps.append(timestamp)

            # 更新カウンタ増加
            self.state.update_counter += len(normalized_actuals)

            return True

        except Exception as e:
            logger.error(f"パフォーマンス更新エラー: {e}")
            return False

    # データ正規化は DataValidator に委譲済み

    def update_performance_batch(
        self,
        batch_predictions: List[Dict[str, Union[float, int, np.ndarray, List[float]]]],
        batch_actuals: List[Union[float, int, np.ndarray, List[float]]],
        batch_timestamps: Optional[List[int]] = None
    ):
        """
        Issue #475対応: バッチ形式でのパフォーマンス更新

        Args:
            batch_predictions: モデル別予測値のリスト
            batch_actuals: 実際の値のリスト
            batch_timestamps: タイムスタンプのリスト

        Raises:
            ValueError: バッチサイズが一致しない場合
        """
        if len(batch_predictions) != len(batch_actuals):
            raise ValueError(
                f"バッチサイズ不一致: predictions={len(batch_predictions)}, "
                f"actuals={len(batch_actuals)}"
            )

        if (batch_timestamps is not None and 
            len(batch_timestamps) != len(batch_predictions)):
            raise ValueError(
                f"タイムスタンプのサイズ不一致: {len(batch_timestamps)} != "
                f"{len(batch_predictions)}"
            )

        batch_size = len(batch_predictions)
        processed_count = 0
        error_count = 0

        for i in range(batch_size):
            try:
                timestamp = batch_timestamps[i] if batch_timestamps else None
                success = self.update_performance(
                    batch_predictions[i], batch_actuals[i], timestamp
                )
                if success:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"バッチ項目{i}の処理エラー: {e}")

        if self.config.verbose:
            logger.info(
                f"バッチ処理完了: 成功={processed_count}, エラー={error_count}"
            )

    def validate_input_data(
        self,
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """
        Issue #475対応: 入力データの検証（DataValidator に委譲）

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            検証結果レポート
        """
        return self.data_validator.validate_input_data(predictions, actuals)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Issue #475対応: データ統計の取得

        Returns:
            データ統計情報
        """
        stats = {
            'total_samples': len(self.state.recent_actuals),
            'update_counter': self.state.update_counter,
            'models': {},
            'actuals_stats': {},
            'data_health': {}
        }

        if len(self.state.recent_actuals) > 0:
            actuals_array = np.array(list(self.state.recent_actuals))
            stats['actuals_stats'] = {
                'mean': float(np.mean(actuals_array)),
                'std': float(np.std(actuals_array)),
                'min': float(np.min(actuals_array)),
                'max': float(np.max(actuals_array)),
                'trend': (float(np.mean(np.diff(actuals_array))) 
                         if len(actuals_array) > 1 else 0.0)
            }

        # モデル別統計
        for model_name, predictions in self.state.recent_predictions.items():
            if len(predictions) > 0:
                pred_array = np.array(list(predictions))
                stats['models'][model_name] = {
                    'count': len(predictions),
                    'mean': float(np.mean(pred_array)),
                    'std': float(np.std(pred_array)),
                    'min': float(np.min(pred_array)),
                    'max': float(np.max(pred_array))
                }

                # 実際値との相関（共通の期間）
                if len(self.state.recent_actuals) >= len(predictions):
                    common_actuals = np.array(
                        list(self.state.recent_actuals)[-len(predictions):]
                    )
                    correlation = np.corrcoef(pred_array, common_actuals)[0, 1]
                    stats['models'][model_name]['correlation'] = (
                        float(correlation) if not np.isnan(correlation) else 0.0
                    )

        # データ健全性チェック
        stats['data_health'] = {
            'sufficient_samples': (
                len(self.state.recent_actuals) >= 
                self.config.min_samples_for_update
            ),
            'all_models_active': all(
                len(preds) > 0 
                for preds in self.state.recent_predictions.values()
            ),
            'data_freshness': len(self.state.recent_actuals) > 0
        }

        return stats

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンス要約の取得

        Returns:
            パフォーマンス要約辞書
        """
        summary = {
            'total_updates': self.state.total_updates,
            'data_points': len(self.state.recent_actuals),
            'model_performance': {}
        }

        # 各モデルの直近パフォーマンス
        for model_name in self.model_names:
            if len(self.state.recent_predictions[model_name]) >= 10:
                pred_array = np.array(
                    list(self.state.recent_predictions[model_name])[-10:]
                )
                actual_array = np.array(list(self.state.recent_actuals)[-10:])

                window = PerformanceWindow(
                    pred_array, actual_array, [], self.state.current_regime
                )
                metrics = window.calculate_metrics()
                summary['model_performance'][model_name] = metrics

        return summary

    def has_sufficient_data(self) -> bool:
        """
        重み更新に十分なデータがあるかチェック

        Returns:
            十分なデータがあるかの判定結果
        """
        return (len(self.state.recent_actuals) >= 
                self.config.min_samples_for_update)

    def should_update_weights(self) -> bool:
        """
        重み更新タイミングの判定

        Returns:
            重み更新すべきかの判定結果
        """
        return (self.state.update_counter >= self.config.update_frequency and
                self.has_sufficient_data())

    def reset_update_counter(self):
        """更新カウンタをリセット"""
        self.state.update_counter = 0

    def get_recent_data_for_model(self, model_name: str) -> Optional[np.ndarray]:
        """
        指定モデルの直近データを取得

        Args:
            model_name: モデル名

        Returns:
            直近の予測データ（なければNone）
        """
        if (model_name in self.state.recent_predictions and
            len(self.state.recent_predictions[model_name]) >= 
            self.config.min_samples_for_update):
            return np.array(
                list(self.state.recent_predictions[model_name])
                [-self.config.min_samples_for_update:]
            )
        return None

    def get_recent_actuals(self) -> Optional[np.ndarray]:
        """
        直近の実際値を取得

        Returns:
            直近の実際値データ（なければNone）
        """
        if len(self.state.recent_actuals) >= self.config.min_samples_for_update:
            return np.array(
                list(self.state.recent_actuals)
                [-self.config.min_samples_for_update:]
            )
        return None