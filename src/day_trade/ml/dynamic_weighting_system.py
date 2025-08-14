#!/usr/bin/env python3
"""
Dynamic Weighting System for Ensemble Learning

Issue #462: 動的重み調整システム実装
市場状況に応じたリアルタイム重み最適化で最高精度を実現
"""

import time
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

from .base_models.base_model_interface import BaseModelInterface, ModelPrediction
from .concept_drift_detector import ConceptDriftDetector, ConceptDriftConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketRegime(Enum):
    """市場状態"""
    BULL_MARKET = "bull"      # 強気相場
    BEAR_MARKET = "bear"      # 弱気相場
    SIDEWAYS = "sideways"     # 横ばい
    HIGH_VOLATILITY = "high_vol"  # 高ボラティリティ
    LOW_VOLATILITY = "low_vol"    # 低ボラティリティ


@dataclass
class PerformanceWindow:
    """パフォーマンス評価ウィンドウ"""
    predictions: np.ndarray
    actuals: np.ndarray
    timestamps: List[int]
    market_regime: Optional[MarketRegime] = None

    def calculate_metrics(self) -> Dict[str, float]:
        """メトリクス計算"""
        if len(self.predictions) == 0:
            return {}

        # 基本メトリクス
        mse = np.mean((self.actuals - self.predictions) ** 2)
        mae = np.mean(np.abs(self.actuals - self.predictions))

        # 方向的中率
        if len(self.predictions) > 1:
            actual_diff = np.diff(self.actuals)
            pred_diff = np.diff(self.predictions)
            hit_rate = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
        else:
            hit_rate = 0.5

        return {
            'rmse': np.sqrt(mse),
            'mae': mae,
            'hit_rate': hit_rate,
            'sample_count': len(self.predictions)
        }


@dataclass
class DynamicWeightingConfig:
    """動的重み調整設定"""
    # パフォーマンス評価
    window_size: int = 100           # 評価ウィンドウサイズ
    min_samples_for_update: int = 50  # 重み更新最小サンプル数
    update_frequency: int = 20        # 更新頻度（サンプル数）

    # 重み調整アルゴリズム
    weighting_method: str = "performance_based"  # performance_based, sharpe_based, regime_aware
    decay_factor: float = 0.95       # 過去データの減衰率
    momentum_factor: float = 0.1     # モーメンタム要素

    # 市場状態適応
    enable_regime_detection: bool = True
    regime_sensitivity: float = 0.3   # 市場状態変化への感度
    volatility_threshold: float = 0.02  # ボラティリティ閾値

    # Issue #478対応: レジーム認識調整外部化
    regime_adjustments: Optional[Dict[MarketRegime, Dict[str, float]]] = None

    # Issue #477対応: スコアリング明確化・カスタマイズ
    sharpe_clip_min: float = 0.1      # シャープレシオ下限クリップ値
    accuracy_weight: float = 1.0      # 精度スコア重み係数
    direction_weight: float = 1.0     # 方向スコア重み係数
    enable_score_logging: bool = False # スコア詳細ログ出力

    # コンセプトドリフト検出
    enable_concept_drift_detection: bool = False
    drift_detection_metric: str = "rmse"
    drift_detection_threshold: float = 0.1
    drift_detection_window_size: int = 50

    # リスク管理
    max_weight_change: float = 0.1    # 1回の最大重み変更
    min_weight: float = 0.05          # 最小重み
    max_weight: float = 0.6           # 最大重み

    # パフォーマンス設定
    verbose: bool = True


class DynamicWeightingSystem:
    """
    動的重み調整システム

    市場状況とモデルパフォーマンスに基づいて
    アンサンブル重みをリアルタイムで最適化
    """

    def __init__(self, model_names: List[str],
                 config: Optional[DynamicWeightingConfig] = None):
        """
        初期化

        Args:
            model_names: モデル名リスト
            config: 動的重み調整設定
        """
        self.model_names = model_names
        self.config = config or DynamicWeightingConfig()

        # Issue #478対応: レジーム調整設定の外部化
        self._setup_regime_adjustments()

        # 現在の重み
        n_models = len(model_names)
        self.current_weights = {name: 1.0 / n_models for name in model_names}
        self.weight_history = []

        # パフォーマンス履歴
        self.performance_windows = {name: deque(maxlen=self.config.window_size) for name in model_names}
        self.recent_predictions = {name: deque(maxlen=self.config.window_size)
                                 for name in model_names}
        self.recent_actuals = deque(maxlen=self.config.window_size)
        self.recent_timestamps = deque(maxlen=self.config.window_size)

        # 市場状態
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.market_indicators = deque(maxlen=50)

        # 更新カウンタ
        self.update_counter = 0
        self.total_updates = 0

        # コンセプトドリフト検出器
        self.concept_drift_detector = None
        self.re_evaluation_needed = False # New flag for model re-evaluation
        self.drift_detection_updates_count = 0 # Counter for updates since last drift detection

        if self.config.enable_concept_drift_detection:
            self.concept_drift_detector = ConceptDriftDetector(
                metric_threshold=self.config.drift_detection_threshold,
                window_size=self.config.drift_detection_window_size
            )

        logger.info(f"Dynamic Weighting System初期化: {n_models}モデル")

    def _setup_regime_adjustments(self):
        """
        Issue #478対応: レジーム調整設定のセットアップ

        外部設定があればそれを使用し、なければデフォルト設定を作成
        """
        if self.config.regime_adjustments is None:
            # デフォルトレジーム調整設定
            self.config.regime_adjustments = self._get_default_regime_adjustments()

        if self.config.verbose:
            logger.info("レジーム調整設定をセットアップしました")

    def _get_default_regime_adjustments(self) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Issue #478対応: デフォルトレジーム調整設定取得

        Returns:
            市場状態別の調整係数辞書
        """
        # 一般的なモデル名のデフォルト設定
        default_adjustments = {
            MarketRegime.BULL_MARKET: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9},
            MarketRegime.BEAR_MARKET: {'svr': 1.2, 'gradient_boosting': 1.1, 'random_forest': 0.9},
            MarketRegime.SIDEWAYS: {'gradient_boosting': 1.1, 'random_forest': 1.0, 'svr': 1.0},
            MarketRegime.HIGH_VOLATILITY: {'svr': 1.3, 'gradient_boosting': 0.9, 'random_forest': 0.8},
            MarketRegime.LOW_VOLATILITY: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9}
        }

        # 実際のモデル名に基づいて調整
        model_adjusted = {}
        for regime, adjustments in default_adjustments.items():
            model_adjusted[regime] = {}
            for model_name in self.model_names:
                # モデル名のマッピング（部分一致で判定）
                adjustment = 1.0  # デフォルト係数
                for pattern, value in adjustments.items():
                    if pattern in model_name.lower():
                        adjustment = value
                        break
                model_adjusted[regime][model_name] = adjustment

        return model_adjusted

    def update_regime_adjustments(self, new_adjustments: Dict[MarketRegime, Dict[str, float]]):
        """
        Issue #478対応: レジーム調整設定の動的更新

        Args:
            new_adjustments: 新しい調整係数辞書
        """
        try:
            # 設定の妥当性チェック
            for regime, adjustments in new_adjustments.items():
                if not isinstance(regime, MarketRegime):
                    raise ValueError(f"無効な市場状態: {regime}")

                for model_name, adjustment in adjustments.items():
                    if not isinstance(adjustment, (int, float)) or adjustment <= 0:
                        raise ValueError(f"無効な調整係数: {model_name}={adjustment}")

            # 設定更新
            self.config.regime_adjustments = new_adjustments

            if self.config.verbose:
                logger.info("レジーム調整設定を更新しました")

        except Exception as e:
            logger.error(f"レジーム調整設定更新エラー: {e}")
            raise

    def load_regime_adjustments_from_dict(self, adjustments_dict: Dict[str, Dict[str, float]]):
        """
        Issue #478対応: 辞書からレジーム調整設定を読み込み

        Args:
            adjustments_dict: 市場状態名をキーとする調整係数辞書
                例: {"bull_market": {"model_a": 1.2, "model_b": 0.9}, ...}
        """
        try:
            regime_adjustments = {}

            for regime_name, adjustments in adjustments_dict.items():
                # 市場状態名を MarketRegime に変換
                regime = None
                for market_regime in MarketRegime:
                    if market_regime.value == regime_name:
                        regime = market_regime
                        break

                if regime is None:
                    logger.warning(f"未知の市場状態: {regime_name}")
                    continue

                regime_adjustments[regime] = adjustments

            self.update_regime_adjustments(regime_adjustments)

        except Exception as e:
            logger.error(f"辞書からのレジーム調整設定読み込みエラー: {e}")
            raise

    def get_regime_adjustments(self) -> Optional[Dict[MarketRegime, Dict[str, float]]]:
        """
        Issue #478対応: 現在のレジーム調整設定取得

        Returns:
            現在の調整係数辞書
        """
        return self.config.regime_adjustments

    def update_scoring_config(self,
                            sharpe_clip_min: Optional[float] = None,
                            accuracy_weight: Optional[float] = None,
                            direction_weight: Optional[float] = None,
                            enable_score_logging: Optional[bool] = None):
        """
        Issue #477対応: スコアリング設定の動的更新

        Args:
            sharpe_clip_min: シャープレシオ下限クリップ値
            accuracy_weight: 精度スコア重み係数
            direction_weight: 方向スコア重み係数
            enable_score_logging: スコア詳細ログ出力フラグ
        """
        try:
            if sharpe_clip_min is not None:
                if sharpe_clip_min < 0:
                    raise ValueError("sharpe_clip_minは0以上である必要があります")
                self.config.sharpe_clip_min = sharpe_clip_min

            if accuracy_weight is not None:
                if accuracy_weight < 0:
                    raise ValueError("accuracy_weightは0以上である必要があります")
                self.config.accuracy_weight = accuracy_weight

            if direction_weight is not None:
                if direction_weight < 0:
                    raise ValueError("direction_weightは0以上である必要があります")
                self.config.direction_weight = direction_weight

            if enable_score_logging is not None:
                self.config.enable_score_logging = enable_score_logging

            if self.config.verbose:
                logger.info("スコアリング設定を更新しました")

        except Exception as e:
            logger.error(f"スコアリング設定更新エラー: {e}")
            raise

    def get_scoring_config(self) -> Dict[str, Any]:
        """
        Issue #477対応: 現在のスコアリング設定取得

        Returns:
            現在のスコアリング設定辞書
        """
        return {
            'sharpe_clip_min': self.config.sharpe_clip_min,
            'accuracy_weight': self.config.accuracy_weight,
            'direction_weight': self.config.direction_weight,
            'enable_score_logging': self.config.enable_score_logging,
            'weighting_method': self.config.weighting_method
        }

    def get_scoring_explanation(self) -> Dict[str, str]:
        """
        Issue #477対応: スコアリング手法の説明取得

        Returns:
            各スコアリング手法の詳細説明
        """
        explanations = {
            'performance_based': {
                'description': 'RMSE逆数と方向的中率の重み付き合計',
                'formula': f'{self.config.accuracy_weight} × (1/(1+RMSE)) + {self.config.direction_weight} × 方向的中率',
                'range': f'0 - {self.config.accuracy_weight + self.config.direction_weight}',
                'components': {
                    'accuracy_score': '1/(1+RMSE) - 予測誤差の逆数（範囲: 0-1）',
                    'direction_score': '方向的中率 - 価格変動方向の予測精度（範囲: 0-1）'
                }
            },
            'sharpe_based': {
                'description': 'リスク調整後の予測精度評価（シャープレシオ）',
                'formula': 'max(mean(accuracy_returns) / std(accuracy_returns), clip_min)',
                'range': f'{self.config.sharpe_clip_min} - ∞',
                'components': {
                    'accuracy_returns': 'pred_returns × actual_returns - 方向一致度',
                    'sharpe_ratio': 'accuracy_returnsの平均/標準偏差',
                    'clipping': f'下限値{self.config.sharpe_clip_min}でクリップ'
                }
            },
            'regime_aware': {
                'description': 'performance_basedに市場状態別調整係数を適用',
                'formula': 'performance_score × regime_adjustment_factor',
                'range': '動的（レジーム調整係数に依存）',
                'components': {
                    'base_score': 'performance_basedスコア',
                    'regime_factor': '現在の市場状態に応じた調整係数'
                }
            }
        }

        return explanations

    def update_performance(self, predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
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

            # 市場状態検出
            if self.config.enable_regime_detection:
                self._detect_market_regime()

            # コンセプトドリフト検出
            if self.concept_drift_detector:
                # 全体のアンサンブル予測と実際値を使用してドリフト検出器を更新
                # より複雑なシナリオでは、個々のモデルのパフォーマンスをフィードすることも検討
                if len(self.recent_actuals) > 0 and len(self.recent_predictions[self.model_names[0]]) > 0:
                    # 最新の単一の予測と実際値を取得
                    latest_actual = self.recent_actuals[-1]
                    # ここでは、最もパフォーマンスの良いモデルの予測を代表として使用するか、
                    # あるいは単純に最初のモデルの予測を使用する。
                    # 実際のシステムでは、アンサンブルの最終予測を使用するのが適切。
                    # 今回はテストのため、単純に最初のモデルの予測を使用する。
                    ensemble_prediction = self.recent_predictions[self.model_names[0]][-1] # 仮のアンサンブル予測

                    self.concept_drift_detector.add_performance_data(
                        predictions=np.array([ensemble_prediction]),
                        actuals=np.array([latest_actual])
                    )
                    drift_result = self.concept_drift_detector.detect_drift()
                    drift_detected = drift_result.get("drift_detected", False)
                    drift_reason = drift_result.get("reason", "不明")

                    if drift_detected:
                        logger.warning(f"コンセプトドリフト検出！理由: {drift_reason}")
                        # 適応戦略: 加速された重み調整
                        self.config.update_frequency = max(1, int(self.config.update_frequency * 0.5)) # 半減
                        self.config.momentum_factor = min(0.9, self.config.momentum_factor + 0.1) # モーメンタム増加
                        logger.info(f"適応戦略適用: update_frequency={self.config.update_frequency}, momentum_factor={self.config.momentum_factor}")

                        # モデル再評価/再トレーニングフラグ
                        self.re_evaluation_needed = True
                        logger.warning("モデルの再評価/再トレーニングが必要です。")

                        # フォールバックメカニズム: 一時的に均等重みに戻す (今回は、ドリフト検出時は常にリセットとする)
                        logger.critical("コンセプトドリフトを検出しました！重みを均等にリセットし、モデルの緊急再評価を推奨します。")
                        n_models = len(self.model_names)
                        self.current_weights = {name: 1.0 / n_models for name in self.model_names}

                        # アラートのトリガー (例: ログ、外部システムへの通知)
                        logger.critical(f"重大なコンセプトドリフトを検出しました！モデルの再評価を強く推奨します。理由: {drift_reason}")
                        # 本番アラートシステム統合
                        self._send_critical_alert(
                            title="重大なコンセプトドリフト検出",
                            message=f"理由: {drift_reason}",
                            urgency="critical"
                        )

            # 重み更新判定
            if (self.update_counter >= self.config.update_frequency and
                len(self.recent_actuals) >= self.config.min_samples_for_update):
                self._update_weights()
                self.update_counter = 0

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

    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self.current_weights.copy()

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

    def validate_input_data(self, predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
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
                    if len(normalized_pred) != len(normalized_actuals) and len(normalized_pred) > 1 and len(normalized_actuals) > 1:
                        report['warnings'].append(f"{model_name}: 次元不一致 {len(normalized_pred)} vs {len(normalized_actuals)}")

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
                    stats['models'][model_name]['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0

        # データ健全性チェック
        stats['data_health'] = {
            'sufficient_samples': len(self.recent_actuals) >= self.config.min_samples_for_update,
            'all_models_active': all(len(preds) > 0 for preds in self.recent_predictions.values()),
            'data_freshness': len(self.recent_actuals) > 0
        }

        return stats

    def update_external_weights(self, external_weights: Dict[str, float]) -> bool:
        """
        Issue #472対応: 外部重みシステムへの直接更新

        外部システム（EnsembleSystemなど）の重みを直接更新する

        Args:
            external_weights: 更新対象の外部重み辞書への参照

        Returns:
            更新成功可否
        """
        try:
            # 現在の重みを取得
            current_weights = self.get_current_weights()

            # 外部重み辞書を更新
            updated_count = 0
            for model_name, weight in current_weights.items():
                if model_name in external_weights:
                    external_weights[model_name] = weight
                    updated_count += 1

            if self.config.verbose:
                logger.info(f"外部重み更新完了: {updated_count}モデルの重みを更新")

            return updated_count > 0

        except Exception as e:
            logger.error(f"外部重み更新エラー: {e}")
            return False

    def sync_and_update_performance(self,
                                  predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                                  actuals: Union[float, int, np.ndarray, List[float]],
                                  external_weights: Optional[Dict[str, float]] = None,
                                  timestamp: Optional[int] = None) -> Dict[str, float]:
        """
        Issue #472対応: 性能更新と外部重み同期を一括処理

        パフォーマンス更新、重み計算、外部重み同期を一つのメソッドで実行

        Args:
            predictions: モデル別予測値
            actuals: 実際の値
            external_weights: 同期対象の外部重み辞書（オプション）
            timestamp: タイムスタンプ（オプション）

        Returns:
            更新後の重み辞書
        """
        try:
            # パフォーマンス更新
            self.update_performance(predictions, actuals, timestamp)

            # 重み更新判定（通常の更新フローに従う）
            if (self.update_counter >= self.config.update_frequency and
                len(self.recent_actuals) >= self.config.min_samples_for_update):

                # 重み更新実行
                self._update_weights()
                self.update_counter = 0

                # 外部重み同期
                if external_weights is not None:
                    self.update_external_weights(external_weights)

                if self.config.verbose:
                    logger.info("統合重み更新完了: パフォーマンス更新 → 重み計算 → 外部同期")

            return self.get_current_weights()

        except Exception as e:
            logger.error(f"統合重み更新エラー: {e}")
            return self.get_current_weights()  # 現在の重みを返す

    def create_weight_updater(self) -> callable:
        """
        Issue #472対応: 重み更新関数の生成

        外部システム用の簡潔な重み更新関数を生成

        Returns:
            重み更新用の関数オブジェクト
        """
        def weight_updater(predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                          actuals: Union[float, int, np.ndarray, List[float]],
                          external_weights: Dict[str, float],
                          timestamp: Optional[int] = None) -> bool:
            """生成された重み更新関数"""
            try:
                updated_weights = self.sync_and_update_performance(
                    predictions, actuals, external_weights, timestamp
                )
                return len(updated_weights) > 0
            except Exception as e:
                logger.warning(f"重み更新関数エラー: {e}")
                return False

        return weight_updater

    def _detect_market_regime(self):
        """市場状態検出"""
        if len(self.recent_actuals) < 20:
            return

        try:
            # 直近のリターンを計算
            recent_values = np.array(list(self.recent_actuals)[-20:])
            returns = np.diff(recent_values) / recent_values[:-1]

            # 統計量計算
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            self.market_indicators.append({
                'mean_return': mean_return,
                'volatility': volatility,
                'timestamp': int(time.time())
            })

            # 市場状態判定
            if volatility > self.config.volatility_threshold:
                new_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < self.config.volatility_threshold * 0.5:
                new_regime = MarketRegime.LOW_VOLATILITY
            elif mean_return > 0.001:
                new_regime = MarketRegime.BULL_MARKET
            elif mean_return < -0.001:
                new_regime = MarketRegime.BEAR_MARKET
            else:
                new_regime = MarketRegime.SIDEWAYS

            # 市場状態変更
            if new_regime != self.current_regime:
                self.regime_history.append({
                    'old_regime': self.current_regime,
                    'new_regime': new_regime,
                    'timestamp': int(time.time())
                })
                self.current_regime = new_regime

                if self.config.verbose:
                    logger.info(f"市場状態変更: {self.current_regime.value}")

        except Exception as e:
            logger.warning(f"市場状態検出エラー: {e}")

    def _update_weights(self):
        """
        重み更新

        Issue #479対応: 重み制約とモメンタム適用順序の最適化
        1. 基本重み計算
        2. モーメンタム適用（制約前）
        3. 包括的制約適用（min/max/sum/change制限）
        4. 最終正規化保証
        """
        try:
            # Step 1: 基本重み計算
            if self.config.weighting_method == "performance_based":
                new_weights = self._performance_based_weighting()
            elif self.config.weighting_method == "sharpe_based":
                new_weights = self._sharpe_based_weighting()
            elif self.config.weighting_method == "regime_aware":
                new_weights = self._regime_aware_weighting()
            else:
                logger.warning(f"未知の重み調整手法: {self.config.weighting_method}")
                return

            # Step 2: モーメンタム適用（制約前に実行）
            # モーメンタムにより滑らかな重み変化を実現
            if self.config.momentum_factor > 0:
                momentum_weights = self._apply_momentum(new_weights)
            else:
                momentum_weights = new_weights

            # Step 3: 包括的制約適用（モーメンタム後に実行）
            # 全制約を同時に考慮して最適な重みを計算
            final_weights = self._apply_comprehensive_constraints(momentum_weights)

            # Step 4: 最終検証と更新
            self._validate_and_update_weights(final_weights)

        except Exception as e:
            logger.error(f"重み更新エラー: {e}")
            # エラー時は現在の重みを維持（安全な動作）

    def _performance_based_weighting(self) -> Dict[str, float]:
        """
        Issue #477対応: パフォーマンスベース重み調整（明確化版）

        スコアリング手法:
        1. 精度スコア: 1/(1+RMSE) - RMSE逆数で精度を評価（範囲: 0-1）
        2. 方向スコア: 方向的中率 - 価格変動方向の予測精度（範囲: 0-1）
        3. 総合スコア: 精度スコア + 方向スコア（範囲: 0-2）

        理論的根拠:
        - RMSE逆数: 低い予測誤差により高いスコアを付与
        - 方向的中率: 金融予測では方向性が重要
        - 加重平均: 両要素を等しく重視した総合評価

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_scores = {}

        for model_name in self.model_names:
            if (len(self.recent_predictions[model_name]) >= self.config.min_samples_for_update and
                len(self.recent_actuals) >= self.config.min_samples_for_update):

                # パフォーマンスウィンドウ作成
                pred_array = np.array(list(self.recent_predictions[model_name])[-self.config.min_samples_for_update:])
                actual_array = np.array(list(self.recent_actuals)[-self.config.min_samples_for_update:])

                # 1. 精度スコア計算（RMSE based）
                rmse = np.sqrt(np.mean((actual_array - pred_array) ** 2))
                # RMSE逆数スコア: 1/(1+RMSE)
                # 理由: RMSEが0の時に1、RMSEが大きくなるにつれて0に近づく
                accuracy_score = 1.0 / (1.0 + rmse)

                # 2. 方向スコア計算（Direction Hit Rate）
                if len(pred_array) > 1:
                    # 実際の価格変化方向
                    actual_diff = np.diff(actual_array)
                    # 予測の価格変化方向
                    pred_diff = np.diff(pred_array)
                    # 方向一致率: sign関数で方向を判定し、一致率を計算
                    direction_score = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
                else:
                    # データ不足時はニュートラル（0.5）
                    direction_score = 0.5

                # 3. 総合スコア算出
                # Issue #477対応: カスタマイズ可能な重み係数を使用
                # 重み付き合計: accuracy_weight * accuracy + direction_weight * direction
                composite_score = (self.config.accuracy_weight * accuracy_score +
                                 self.config.direction_weight * direction_score)

                model_scores[model_name] = composite_score

                # 詳細ログ（カスタマイズ可能）
                if self.config.enable_score_logging or self.config.verbose:
                    logger.debug(f"{model_name} スコア詳細: RMSE={rmse:.4f}, "
                               f"精度={accuracy_score:.3f}(×{self.config.accuracy_weight}), "
                               f"方向={direction_score:.3f}(×{self.config.direction_weight}), "
                               f"総合={composite_score:.3f}")
            else:
                # データ不足の場合は中立スコア（1.0）
                # 理由: 精度0.5 + 方向0.5 = 1.0 の中立的評価
                model_scores[model_name] = 1.0

        # 重みの正規化
        total_score = sum(model_scores.values())
        if total_score > 0:
            normalized_weights = {name: score / total_score for name, score in model_scores.items()}

            if self.config.verbose:
                logger.info(f"パフォーマンスベース重み: {normalized_weights}")

            return normalized_weights
        else:
            # フォールバック: 均等重み
            logger.warning("全モデルスコアが0です。均等重みを適用します。")
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _sharpe_based_weighting(self) -> Dict[str, float]:
        """
        Issue #477対応: シャープレシオベース重み調整（明確化版）

        スコアリング手法:
        1. リターン計算: 予測・実際両方の価格変化率を算出
        2. 精度リターン: pred_returns × actual_returns で方向一致度を評価
        3. シャープレシオ: 精度リターンの平均/標準偏差
        4. 下限クリップ: 負のシャープレシオを0.1に制限

        理論的根拠:
        - 精度リターン: 方向が一致する時は正値、不一致時は負値
        - シャープレシオ: リスク調整後リターンの標準的指標
        - 下限クリップ: 極端に悪いモデルでも最小限の重みを保持
        - クリップ値0.1: 経験的に安定した重み分散を実現

        数学的定義:
        accuracy_returns[i] = pred_return[i] × actual_return[i]
        sharpe_ratio = mean(accuracy_returns) / std(accuracy_returns)
        final_sharpe = max(sharpe_ratio, 0.1)

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_sharpe = {}

        for model_name in self.model_names:
            if len(self.recent_predictions[model_name]) >= self.config.min_samples_for_update:
                pred_array = np.array(list(self.recent_predictions[model_name])[-self.config.min_samples_for_update:])
                actual_array = np.array(list(self.recent_actuals)[-self.config.min_samples_for_update:])

                # 1. リターン計算
                # 予測リターン: (価格t+1 - 価格t) / 価格t
                pred_returns = np.diff(pred_array) / pred_array[:-1]
                # 実際リターン: 同様の計算
                actual_returns = np.diff(actual_array) / actual_array[:-1]

                # 2. 予測精度リターン計算
                # 理論: 方向が一致する時は正値、不一致時は負値
                # 例: pred_return=0.1, actual_return=0.05 => accuracy_return=0.005 (正値)
                # 例: pred_return=0.1, actual_return=-0.05 => accuracy_return=-0.005 (負値)
                accuracy_returns = pred_returns * actual_returns

                # 3. シャープレシオ計算
                accuracy_std = np.std(accuracy_returns)
                if accuracy_std > 0:
                    # シャープレシオ = 期待リターン / リターンのボラティリティ
                    sharpe_ratio = np.mean(accuracy_returns) / accuracy_std
                else:
                    # ボラティリティが0の場合（予測が一定）
                    sharpe_ratio = 0.0

                # 4. 下限クリップ適用
                # Issue #477対応: 設定可能なクリップ値を使用
                # - 負のシャープレシオは予測性能が非常に悪いことを示す
                # - しかし完全に排除せず、最小限の重み（設定値）を維持
                # - これにより重み分散の極端な偏りを防ぎ、安定性を向上
                clipped_sharpe = max(sharpe_ratio, self.config.sharpe_clip_min)
                model_sharpe[model_name] = clipped_sharpe

                # 詳細ログ（カスタマイズ可能）
                if self.config.enable_score_logging or self.config.verbose:
                    logger.debug(f"{model_name} シャープ詳細: 生シャープ={sharpe_ratio:.4f}, "
                               f"クリップ後={clipped_sharpe:.3f} (下限={self.config.sharpe_clip_min}), "
                               f"精度リターン平均={np.mean(accuracy_returns):.4f}")
            else:
                # データ不足時の中立値
                # 理由: 0.5はクリップ値0.1より大きく、均等重みに近い扱い
                model_sharpe[model_name] = 0.5

        # 重みの正規化
        total_sharpe = sum(model_sharpe.values())
        if total_sharpe > 0:
            normalized_weights = {name: sharpe / total_sharpe for name, sharpe in model_sharpe.items()}

            if self.config.verbose:
                logger.info(f"シャープベース重み: {normalized_weights}")

            return normalized_weights
        else:
            logger.warning("全モデルシャープレシオが0です。均等重みを適用します。")
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _regime_aware_weighting(self) -> Dict[str, float]:
        """
        Issue #478対応: 市場状態適応重み調整（外部設定対応）

        Returns:
            市場状態に応じて調整された重み辞書
        """
        try:
            # 基本パフォーマンスベース重み
            base_weights = self._performance_based_weighting()

            # Issue #478対応: 外部設定されたレジーム調整係数を使用
            regime_adjustments = self.config.regime_adjustments
            if not regime_adjustments:
                logger.warning("レジーム調整設定が見つかりません。基本重みを返します。")
                return base_weights

            # 調整係数適用
            adjusted_weights = {}
            adjustments = regime_adjustments.get(self.current_regime, {})

            if not adjustments:
                logger.warning(f"現在のレジーム '{self.current_regime.value}' の調整設定が見つかりません。")
                return base_weights

            for model_name in self.model_names:
                base_weight = base_weights.get(model_name, 1.0 / len(self.model_names))
                adjustment = adjustments.get(model_name, 1.0)
                adjusted_weights[model_name] = base_weight * adjustment

            # 正規化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {name: weight / total_weight for name, weight in adjusted_weights.items()}

                if self.config.verbose:
                    changes = []
                    for model_name in self.model_names:
                        adj = adjustments.get(model_name, 1.0)
                        if adj != 1.0:
                            changes.append(f"{model_name}x{adj:.1f}")
                    if changes:
                        logger.info(f"レジーム '{self.current_regime.value}' 調整: {', '.join(changes)}")

                return normalized_weights
            else:
                logger.warning("調整後の重み合計が0です。基本重みを返します。")
                return base_weights

        except Exception as e:
            logger.error(f"レジーム認識重み調整エラー: {e}")
            return self._performance_based_weighting()

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重み制約適用"""
        constrained = {}

        for model_name, new_weight in weights.items():
            current_weight = self.current_weights.get(model_name, 1.0 / len(self.model_names))

            # 最大変更量制限
            max_change = self.config.max_weight_change
            if new_weight > current_weight + max_change:
                constrained_weight = current_weight + max_change
            elif new_weight < current_weight - max_change:
                constrained_weight = current_weight - max_change
            else:
                constrained_weight = new_weight

            # 最小・最大重み制限
            constrained_weight = max(self.config.min_weight, constrained_weight)
            constrained_weight = min(self.config.max_weight, constrained_weight)

            constrained[model_name] = constrained_weight

        # 正規化（制約により合計が1でなくなる場合）
        total_weight = sum(constrained.values())
        if total_weight > 0:
            return {name: weight / total_weight for name, weight in constrained.items()}
        else:
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _apply_momentum(self, weights: Dict[str, float]) -> Dict[str, float]:
        """モーメンタム適用"""
        momentum_weights = {}
        momentum = self.config.momentum_factor

        for model_name in self.model_names:
            current = self.current_weights.get(model_name, 1.0 / len(self.model_names))
            new = weights.get(model_name, 1.0 / len(self.model_names))

            # モーメンタム適用: 新重み = (1-momentum) * 新重み + momentum * 現在重み
            momentum_weight = (1 - momentum) * new + momentum * current
            momentum_weights[model_name] = momentum_weight

        return momentum_weights

    def _apply_comprehensive_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        包括的制約適用

        Issue #479対応: モーメンタム後に全制約を同時適用
        1. 最大変更量制限
        2. 最小・最大重み制限
        3. 合計1.0正規化
        4. 制約競合時の最適解計算
        """
        try:
            constrained = {}

            # Step 1: 各モデルの制約適用
            for model_name in self.model_names:
                new_weight = weights.get(model_name, 1.0 / len(self.model_names))
                current_weight = self.current_weights.get(model_name, 1.0 / len(self.model_names))

                # 最大変更量制限
                max_change = self.config.max_weight_change
                if new_weight > current_weight + max_change:
                    constrained_weight = current_weight + max_change
                elif new_weight < current_weight - max_change:
                    constrained_weight = current_weight - max_change
                else:
                    constrained_weight = new_weight

                # 最小・最大重み制限
                constrained_weight = max(self.config.min_weight, constrained_weight)
                constrained_weight = min(self.config.max_weight, constrained_weight)

                constrained[model_name] = constrained_weight

            # Step 2: 合計正規化（制約により合計が1でない場合の対処）
            total_weight = sum(constrained.values())

            if total_weight > 0:
                # 比例配分による正規化
                normalized = {name: weight / total_weight for name, weight in constrained.items()}

                # Step 3: 正規化後の制約再チェック
                final_weights = {}
                needs_rebalancing = False

                for model_name, normalized_weight in normalized.items():
                    # 正規化により制約を逸脱していないかチェック
                    if (normalized_weight < self.config.min_weight or
                        normalized_weight > self.config.max_weight):
                        needs_rebalancing = True
                        # 制約内にクリップ
                        final_weights[model_name] = max(self.config.min_weight,
                                                      min(self.config.max_weight, normalized_weight))
                    else:
                        final_weights[model_name] = normalized_weight

                # Step 4: リバランシング必要時の最終調整
                if needs_rebalancing:
                    final_total = sum(final_weights.values())
                    if final_total > 0:
                        # 最終正規化
                        final_weights = {name: weight / final_total for name, weight in final_weights.items()}

                return final_weights
            else:
                # フォールバック: 均等分散
                logger.warning("制約適用後に重み合計が0になりました。均等分散を適用します。")
                return {name: 1.0 / len(self.model_names) for name in self.model_names}

        except Exception as e:
            logger.error(f"包括的制約適用エラー: {e}")
            # エラー時は現在の重みを維持
            return self.current_weights.copy()

    def _validate_and_update_weights(self, weights: Dict[str, float]):
        """
        最終検証と重み更新

        Issue #479対応: 制約チェックと安全な重み更新
        """
        try:
            # 基本検証
            if not weights or len(weights) != len(self.model_names):
                logger.warning("無効な重み辞書です。現在の重みを維持します。")
                return

            # 制約検証
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"重み合計が1.0でありません: {total_weight:.6f}")
                return

            # 個別重み制約検証
            for model_name, weight in weights.items():
                if weight < 0 or weight > 1:
                    logger.warning(f"重み範囲外: {model_name}={weight:.6f}")
                    return

                if weight < self.config.min_weight or weight > self.config.max_weight:
                    logger.warning(f"設定制約外: {model_name}={weight:.6f} "
                                 f"(範囲: {self.config.min_weight}-{self.config.max_weight})")
                    return

            # 重み更新実行
            old_weights = self.current_weights.copy()
            self.current_weights = weights.copy()

            # 履歴記録
            self.weight_history.append({
                'weights': weights.copy(),
                'timestamp': int(time.time()),
                'regime': self.current_regime,
                'total_updates': self.total_updates
            })

            self.total_updates += 1

            # ログ出力（設定に応じて）
            if self.config.verbose:
                changes = []
                for model_name in self.model_names:
                    old_w = old_weights.get(model_name, 0)
                    new_w = weights[model_name]
                    if abs(new_w - old_w) > 0.01:  # 1%以上の変化
                        changes.append(f"{model_name}: {old_w:.3f}→{new_w:.3f}")

                if changes:
                    logger.info(f"重み更新: {', '.join(changes)}")

        except Exception as e:
            logger.error(f"重み検証・更新エラー: {e}")
            # エラー時は重みを変更しない

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """重み履歴取得"""
        return self.weight_history.copy()

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """市場状態履歴取得"""
        return self.regime_history.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        summary = {
            'current_weights': self.current_weights,
            'current_regime': self.current_regime.value,
            'total_updates': self.total_updates,
            'data_points': len(self.recent_actuals)
        }

        # 各モデルの直近パフォーマンス
        model_performance = {}
        for model_name in self.model_names:
            if len(self.recent_predictions[model_name]) >= 10:
                pred_array = np.array(list(self.recent_predictions[model_name])[-10:])
                actual_array = np.array(list(self.recent_actuals)[-10:])

                window = PerformanceWindow(pred_array, actual_array, [], self.current_regime)
                metrics = window.calculate_metrics()
                model_performance[model_name] = metrics

        summary['model_performance'] = model_performance
        return summary

    def plot_weight_evolution(self, save_path: Optional[str] = None):
        """重み変化の可視化"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime

            if not self.weight_history:
                logger.warning("重み履歴データが存在しません")
                return

            # データ準備
            timestamps = [datetime.fromtimestamp(h['timestamp']) for h in self.weight_history]

            plt.figure(figsize=(12, 8))

            # 各モデルの重み変化をプロット
            for model_name in self.model_names:
                weights = [h['weights'].get(model_name, 0) for h in self.weight_history]
                plt.plot(timestamps, weights, label=model_name, marker='o', markersize=3)

            plt.xlabel('時間')
            plt.ylabel('重み')
            plt.title('動的重み調整の変化')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # 市場状態の背景色
            regime_colors = {
                MarketRegime.BULL_MARKET: 'lightgreen',
                MarketRegime.BEAR_MARKET: 'lightcoral',
                MarketRegime.HIGH_VOLATILITY: 'lightyellow',
                MarketRegime.LOW_VOLATILITY: 'lightblue',
                MarketRegime.SIDEWAYS: 'lightgray'
            }

            for i, h in enumerate(self.weight_history[:-1]):
                plt.axvspan(timestamps[i], timestamps[i+1],
                           alpha=0.2, color=regime_colors.get(h.get('regime', MarketRegime.SIDEWAYS), 'white'))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"重み変化グラフ保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未インストール")
        except Exception as e:
            logger.error(f"重み変化可視化エラー: {e}")

    def _send_critical_alert(self, title: str, message: str, urgency: str = "high"):
        """重要アラート送信（本番実装）"""
        try:
            # ログベースアラート（基本実装）
            logger.critical(f"[ALERT-{urgency.upper()}] {title}: {message}")
            
            # アラート履歴記録
            if not hasattr(self, '_alert_history'):
                self._alert_history = []
            
            import time
            alert_record = {
                'timestamp': time.time(),
                'title': title,
                'message': message,
                'urgency': urgency,
                'acknowledged': False
            }
            self._alert_history.append(alert_record)
            
            # 外部システム統合（例：Slack、メール等）
            # 実際の本番環境では以下を有効化：
            # self._send_to_external_systems(alert_record)
            
        except Exception as e:
            logger.error(f"アラート送信失敗: {e}")

    def _send_to_external_systems(self, alert_record: dict):
        """外部アラートシステムへの送信（拡張可能）"""
        # 実装例：
        # - Slack webhook
        # - メール送信
        # - PagerDuty統合
        # - Teams通知
        pass

    def get_alert_history(self) -> list:
        """アラート履歴取得"""
        return getattr(self, '_alert_history', [])


if __name__ == "__main__":
    # テスト実行
    print("=== Dynamic Weighting System テスト ===")

    # テストデータ生成
    np.random.seed(42)
    model_names = ["random_forest", "gradient_boosting", "svr"]

    # システム初期化
    config = DynamicWeightingConfig(
        window_size=50,
        update_frequency=10,
        weighting_method="regime_aware"
    )
    dws = DynamicWeightingSystem(model_names, config)

    print(f"初期重み: {dws.get_current_weights()}")

    # シミュレーションデータ
    n_steps = 200
    for step in range(n_steps):
        # 異なるトレンドを持つテストデータ
        if step < 50:
            # 上昇トレンド（Bull Market）
            true_value = 100 + step * 0.5 + np.random.normal(0, 1)
        elif step < 100:
            # 下降トレンド（Bear Market）
            true_value = 125 - (step - 50) * 0.3 + np.random.normal(0, 2)
        elif step < 150:
            # 横ばい（Sideways）
            true_value = 110 + np.random.normal(0, 0.5)
        else:
            # 高ボラティリティ
            true_value = 110 + np.random.normal(0, 5)

        # モデル予測値（異なる特性を持つ）
        predictions = {
            "random_forest": true_value + np.random.normal(0, 1.5),
            "gradient_boosting": true_value + np.random.normal(0, 1.0),
            "svr": true_value + np.random.normal(0, 2.0)
        }

        # パフォーマンス更新
        dws.update_performance(predictions, true_value, step)

        if step % 50 == 0:
            weights = dws.get_current_weights()
            print(f"Step {step}: 重み={weights}, 市場状態={dws.current_regime.value}")

    # 最終結果
    print("\n=== 最終結果 ===")
    summary = dws.get_performance_summary()
    print(f"最終重み: {summary['current_weights']}")
    print(f"総更新回数: {summary['total_updates']}")
    print(f"現在の市場状態: {summary['current_regime']}")

    # パフォーマンス履歴
    regime_history = dws.get_regime_history()
    print(f"\n市場状態変更回数: {len(regime_history)}")
    for change in regime_history[-3:]:
        print(f"  {change['old_regime'].value} -> {change['new_regime'].value}")