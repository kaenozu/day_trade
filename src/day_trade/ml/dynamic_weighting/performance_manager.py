#!/usr/bin/env python3
"""
Dynamic Weighting System - Performance Manager

パフォーマンス管理とデータ正規化処理
"""

import time
from typing import Dict, List, Union, Optional, Any
import numpy as np
from collections import deque

from .core import DynamicWeightingConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class PerformanceManager:
    """パフォーマンス管理クラス"""

    def __init__(self, config: DynamicWeightingConfig, model_names: List[str]):
        """
        初期化

        Args:
            config: 動的重み調整設定
            model_names: モデル名リスト
        """
        self.config = config
        self.model_names = model_names

        # パフォーマンス履歴
        self.recent_predictions = {
            name: deque(maxlen=self.config.window_size) for name in model_names
        }
        self.recent_actuals = deque(maxlen=self.config.window_size)
        self.recent_timestamps = deque(maxlen=self.config.window_size)

        # カウンタ
        self.update_counter = 0

    def update_performance(self,
                         predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                         actuals: Union[float, int, np.ndarray, List[float]],
                         timestamp: Optional[int] = None):
        """
        Issue #475対応: パフォーマンス更新（改善版）

        予測値・実際値の処理を統一し、冗長なチェックを排除

        Args:
            predictions: モデル別予測値（単一値または配列）
            actuals: 実際の値（単一値または配列）
            timestamp: タイムスタンプ（未指定時は現在時刻）

        Raises:
            ValueError: 予測値と実際値の次元が一致しない場合
            TypeError: サポートされていない型の場合
        """
        try:
            if timestamp is None:
                timestamp = int(time.time())

            # Issue #475対応: 一貫した配列変換処理
            normalized_actuals = self._normalize_to_array(actuals, "actuals")

            # 予測値の正規化と記録
            for model_name, pred in predictions.items():
                if model_name in self.recent_predictions:
                    try:
                        normalized_pred = self._normalize_to_array(pred, f"predictions[{model_name}]")

                        # 次元一致性チェック
                        if len(normalized_pred) != len(normalized_actuals):
                            if len(normalized_pred) == 1:
                                # 単一予測値を実際値の数だけ複製
                                normalized_pred = np.repeat(normalized_pred[0], len(normalized_actuals))
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
                            self.recent_predictions[model_name].append(float(pred_val))

                    except Exception as e:
                        logger.warning(f"{model_name}の予測値処理でエラー: {e}")
                        continue

            # 実際値とタイムスタンプの記録
            for actual_val in normalized_actuals:
                self.recent_actuals.append(float(actual_val))
                self.recent_timestamps.append(timestamp)

            # 更新カウンタ増加
            self.update_counter += len(normalized_actuals)

        except Exception as e:
            logger.error(f"パフォーマンス更新エラー: {e}")
            raise

    def _normalize_to_array(self, data: Union[float, int, np.ndarray, List[float]],
                          name: str) -> np.ndarray:
        """
        Issue #475対応: データの一貫した配列変換

        Args:
            data: 変換対象データ
            name: データ名（エラーメッセージ用）

        Returns:
            正規化されたnp.ndarray

        Raises:
            TypeError: サポートされていない型の場合
            ValueError: 無効なデータの場合
        """
        try:
            # None チェック
            if data is None:
                raise ValueError(f"{name}がNoneです")

            # 型別処理
            if isinstance(data, (int, float)):
                # 単一値の場合
                if np.isnan(data) or np.isinf(data):
                    raise ValueError(f"{name}に無効な値が含まれています: {data}")
                return np.array([float(data)])

            elif isinstance(data, (list, tuple)):
                # リスト/タプルの場合
                if len(data) == 0:
                    raise ValueError(f"{name}が空です")
                array_data = np.array(data, dtype=float)

            elif isinstance(data, np.ndarray):
                # NumPy配列の場合
                if data.size == 0:
                    raise ValueError(f"{name}が空の配列です")
                array_data = data.astype(float)

            else:
                # その他の型（pd.Series等も含む）
                try:
                    array_data = np.array(data, dtype=float)
                except Exception as e:
                    raise TypeError(f"{name}の型{type(data)}はサポートされていません: {e}")

            # 1次元に変換
            array_data = np.atleast_1d(array_data.flatten())

            # 有効値チェック
            if np.any(np.isnan(array_data)) or np.any(np.isinf(array_data)):
                raise ValueError(f"{name}に無効な値(NaN/Inf)が含まれています")

            return array_data

        except Exception as e:
            logger.error(f"データ正規化エラー ({name}): {e}")
            raise

    def update_performance_batch(self,
                               batch_predictions: List[Dict[str, Union[float, int, np.ndarray, List[float]]]],
                               batch_actuals: List[Union[float, int, np.ndarray, List[float]]],
                               batch_timestamps: Optional[List[int]] = None):
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
            raise ValueError(f"バッチサイズ不一致: predictions={len(batch_predictions)}, actuals={len(batch_actuals)}")

        if batch_timestamps is not None and len(batch_timestamps) != len(batch_predictions):
            raise ValueError(f"タイムスタンプのサイズ不一致: {len(batch_timestamps)} != {len(batch_predictions)}")

        batch_size = len(batch_predictions)
        processed_count = 0
        error_count = 0

        for i in range(batch_size):
            try:
                timestamp = batch_timestamps[i] if batch_timestamps else None
                self.update_performance(batch_predictions[i], batch_actuals[i], timestamp)
                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"バッチ項目{i}の処理エラー: {e}")

        if self.config.verbose:
            logger.info(f"バッチ処理完了: 成功={processed_count}, エラー={error_count}")

    def validate_input_data(self,
                           predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                           actuals: Union[float, int, np.ndarray, List[float]]) -> Dict[str, any]:
        """
        Issue #475対応: 入力データの検証

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            検証結果レポート
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'model_stats': {},
            'data_shape': {}
        }

        try:
            # 実際値の検証
            normalized_actuals = self._normalize_to_array(actuals, "actuals")
            report['data_shape']['actuals'] = len(normalized_actuals)

            # 予測値の検証
            for model_name, pred in predictions.items():
                try:
                    normalized_pred = self._normalize_to_array(pred, f"predictions[{model_name}]")
                    report['data_shape'][model_name] = len(normalized_pred)
                    report['model_stats'][model_name] = {
                        'mean': float(np.mean(normalized_pred)),
                        'std': float(np.std(normalized_pred)),
                        'min': float(np.min(normalized_pred)),
                        'max': float(np.max(normalized_pred))
                    }

                    # 次元チェック
                    if (len(normalized_pred) != len(normalized_actuals) and 
                        len(normalized_pred) > 1 and len(normalized_actuals) > 1):
                        report['warnings'].append(
                            f"{model_name}: 次元不一致 {len(normalized_pred)} vs {len(normalized_actuals)}"
                        )

                except Exception as e:
                    report['errors'].append(f"{model_name}: {str(e)}")
                    report['valid'] = False

        except Exception as e:
            report['errors'].append(f"実際値検証エラー: {str(e)}")
            report['valid'] = False

        return report

    def get_data_statistics(self) -> Dict[str, any]:
        """
        Issue #475対応: データ統計の取得

        Returns:
            データ統計情報
        """
        stats = {
            'total_samples': len(self.recent_actuals),
            'update_counter': self.update_counter,
            'models': {},
            'actuals_stats': {},
            'data_health': {}
        }

        if len(self.recent_actuals) > 0:
            actuals_array = np.array(list(self.recent_actuals))
            stats['actuals_stats'] = {
                'mean': float(np.mean(actuals_array)),
                'std': float(np.std(actuals_array)),
                'min': float(np.min(actuals_array)),
                'max': float(np.max(actuals_array)),
                'trend': float(np.mean(np.diff(actuals_array))) if len(actuals_array) > 1 else 0.0
            }

        # モデル別統計
        for model_name, predictions in self.recent_predictions.items():
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
                if len(self.recent_actuals) >= len(predictions):
                    common_actuals = np.array(list(self.recent_actuals)[-len(predictions):])
                    correlation = np.corrcoef(pred_array, common_actuals)[0, 1]
                    stats['models'][model_name]['correlation'] = (
                        float(correlation) if not np.isnan(correlation) else 0.0
                    )

        # データ健全性チェック
        stats['data_health'] = {
            'sufficient_samples': len(self.recent_actuals) >= self.config.min_samples_for_update,
            'all_models_active': all(len(preds) > 0 for preds in self.recent_predictions.values()),
            'data_freshness': len(self.recent_actuals) > 0
        }

        return stats

    def reset_counter(self):
        """更新カウンタのリセット"""
        self.update_counter = 0

    def has_sufficient_samples(self) -> bool:
        """十分なサンプル数があるかチェック"""
        return (self.update_counter >= self.config.update_frequency and
                len(self.recent_actuals) >= self.config.min_samples_for_update)