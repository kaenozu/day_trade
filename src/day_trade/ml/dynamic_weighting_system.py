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

    def update_performance(self, predictions: Dict[str, np.ndarray],
                         actuals: np.ndarray, timestamp: int = None):
        """
        パフォーマンス更新

        Args:
            predictions: モデル別予測値
            actuals: 実際の値
            timestamp: タイムスタンプ
        """
        if timestamp is None:
            timestamp = int(time.time())

        # 予測値と実際値を記録
        for model_name, pred in predictions.items():
            if model_name in self.recent_predictions:
                if len(pred) == len(actuals):
                    for p in pred:
                        self.recent_predictions[model_name].append(p)
                else:
                    # 単一予測値の場合
                    self.recent_predictions[model_name].append(pred[0] if hasattr(pred, '__len__') else pred)

        # 実際値とタイムスタンプを記録
        for actual in (actuals if hasattr(actuals, '__len__') else [actuals]):
            self.recent_actuals.append(actual)
            self.recent_timestamps.append(timestamp)

        self.update_counter += len(actuals) if hasattr(actuals, '__len__') else 1

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
                    # TODO: Integrate with actual alert system (e.g., send email, Slack notification)

        # 重み更新判定
        if (self.update_counter >= self.config.update_frequency and
            len(self.recent_actuals) >= self.config.min_samples_for_update):
            self._update_weights()
            self.update_counter = 0

    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self.current_weights.copy()

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
        """重み更新"""
        try:
            if self.config.weighting_method == "performance_based":
                new_weights = self._performance_based_weighting()
            elif self.config.weighting_method == "sharpe_based":
                new_weights = self._sharpe_based_weighting()
            elif self.config.weighting_method == "regime_aware":
                new_weights = self._regime_aware_weighting()
            else:
                logger.warning(f"未知の重み調整手法: {self.config.weighting_method}")
                return

            # 重み変更制限
            constrained_weights = self._apply_weight_constraints(new_weights)

            # モーメンタム適用
            if self.config.momentum_factor > 0:
                constrained_weights = self._apply_momentum(constrained_weights)

            # 重み更新
            self.current_weights = constrained_weights
            self.weight_history.append({
                'weights': constrained_weights.copy(),
                'timestamp': int(time.time()),
                'regime': self.current_regime,
                'update_method': self.config.weighting_method
            })

            self.total_updates += 1

            if self.config.verbose:
                logger.info(f"重み更新 #{self.total_updates}: {constrained_weights}")

        except Exception as e:
            logger.error(f"重み更新エラー: {e}")

    def _performance_based_weighting(self) -> Dict[str, float]:
        """パフォーマンスベース重み調整"""
        model_scores = {}

        for model_name in self.model_names:
            if (len(self.recent_predictions[model_name]) >= self.config.min_samples_for_update and
                len(self.recent_actuals) >= self.config.min_samples_for_update):

                # パフォーマンスウィンドウ作成
                pred_array = np.array(list(self.recent_predictions[model_name])[-self.config.min_samples_for_update:])
                actual_array = np.array(list(self.recent_actuals)[-self.config.min_samples_for_update:])

                # メトリクス計算
                rmse = np.sqrt(np.mean((actual_array - pred_array) ** 2))

                # 方向的中率
                if len(pred_array) > 1:
                    actual_diff = np.diff(actual_array)
                    pred_diff = np.diff(pred_array)
                    hit_rate = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
                else:
                    hit_rate = 0.5

                # 総合スコア（RMSEの逆数 + 方向的中率）
                score = (1.0 / (1.0 + rmse)) + hit_rate
                model_scores[model_name] = score
            else:
                # データ不足の場合は現在の重みを保持
                model_scores[model_name] = 1.0

        # 重みの正規化
        total_score = sum(model_scores.values())
        if total_score > 0:
            return {name: score / total_score for name, score in model_scores.items()}
        else:
            # フォールバック: 均等重み
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _sharpe_based_weighting(self) -> Dict[str, float]:
        """シャープレシオベース重み調整"""
        model_sharpe = {}

        for model_name in self.model_names:
            if len(self.recent_predictions[model_name]) >= self.config.min_samples_for_update:
                pred_array = np.array(list(self.recent_predictions[model_name])[-self.config.min_samples_for_update:])
                actual_array = np.array(list(self.recent_actuals)[-self.config.min_samples_for_update:])

                # 予測リターン
                pred_returns = np.diff(pred_array) / pred_array[:-1]
                actual_returns = np.diff(actual_array) / actual_array[:-1]

                # 予測精度リターン
                accuracy_returns = pred_returns * actual_returns  # 方向的中時は正、外れた時は負

                # シャープレシオ計算
                if np.std(accuracy_returns) > 0:
                    sharpe = np.mean(accuracy_returns) / np.std(accuracy_returns)
                else:
                    sharpe = 0.0

                model_sharpe[model_name] = max(sharpe, 0.1)  # 最小値設定
            else:
                model_sharpe[model_name] = 0.5

        # 重みの正規化
        total_sharpe = sum(model_sharpe.values())
        if total_sharpe > 0:
            return {name: sharpe / total_sharpe for name, sharpe in model_sharpe.items()}
        else:
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _regime_aware_weighting(self) -> Dict[str, float]:
        """市場状態適応重み調整"""
        # 基本パフォーマンスベース重み
        base_weights = self._performance_based_weighting()

        # 市場状態別の調整係数
        regime_adjustments = {
            MarketRegime.BULL_MARKET: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9},
            MarketRegime.BEAR_MARKET: {'svr': 1.2, 'gradient_boosting': 1.1, 'random_forest': 0.9},
            MarketRegime.SIDEWAYS: {'gradient_boosting': 1.1, 'random_forest': 1.0, 'svr': 1.0},
            MarketRegime.HIGH_VOLATILITY: {'svr': 1.3, 'gradient_boosting': 0.9, 'random_forest': 0.8},
            MarketRegime.LOW_VOLATILITY: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9}
        }

        # 調整係数適用
        adjusted_weights = {}
        adjustments = regime_adjustments.get(self.current_regime, {})

        for model_name in self.model_names:
            base_weight = base_weights.get(model_name, 1.0 / len(self.model_names))
            adjustment = adjustments.get(model_name, 1.0)
            adjusted_weights[model_name] = base_weight * adjustment

        # 正規化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            return {name: weight / total_weight for name, weight in adjusted_weights.items()}
        else:
            return base_weights

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