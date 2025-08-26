#!/usr/bin/env python3
"""
Weight calculation algorithms for Dynamic Weighting System

このモジュールは各種重み計算アルゴリズムを実装します。
パフォーマンスベース、シャープレシオベース、市場状態適応型など
複数の手法を提供し、モデルアンサンブルの最適重みを算出します。
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

from .core import MarketRegime, DynamicWeightingConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightCalculator:
    """
    重み計算クラス

    各種重み計算アルゴリズムを実装し、
    モデル性能に基づく最適重みを算出します。
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

    def calculate_performance_based_weights(
        self,
        recent_predictions: Dict[str, deque],
        recent_actuals: deque
    ) -> Dict[str, float]:
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

        Args:
            recent_predictions: モデル別の直近予測値
            recent_actuals: 直近の実際値

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_scores = {}

        for model_name in self.model_names:
            if (len(recent_predictions[model_name]) >= 
                self.config.min_samples_for_update and
                len(recent_actuals) >= self.config.min_samples_for_update):

                # パフォーマンスウィンドウ作成
                pred_array = np.array(
                    list(recent_predictions[model_name])
                    [-self.config.min_samples_for_update:]
                )
                actual_array = np.array(
                    list(recent_actuals)[-self.config.min_samples_for_update:]
                )

                # 1. 精度スコア計算（RMSE based）
                rmse = np.sqrt(np.mean((actual_array - pred_array) ** 2))
                # RMSE逆数スコア: 1/(1+RMSE)
                # 理由: RMSEが0の時に1、RMSEが大きくなるにつれて0に近づく
                accuracy_score = 1.0 / (1.0 + rmse)

                # 2. 方向スコア計算（Direction Hit Rate）
                direction_score = self._calculate_direction_score(
                    pred_array, actual_array
                )

                # 3. 総合スコア算出
                # Issue #477対応: カスタマイズ可能な重み係数を使用
                composite_score = (
                    self.config.accuracy_weight * accuracy_score +
                    self.config.direction_weight * direction_score
                )

                model_scores[model_name] = composite_score

                # 詳細ログ（カスタマイズ可能）
                if self.config.enable_score_logging or self.config.verbose:
                    logger.debug(
                        f"{model_name} スコア詳細: RMSE={rmse:.4f}, "
                        f"精度={accuracy_score:.3f}(×{self.config.accuracy_weight}), "
                        f"方向={direction_score:.3f}(×{self.config.direction_weight}), "
                        f"総合={composite_score:.3f}"
                    )
            else:
                # データ不足の場合は中立スコア（1.0）
                model_scores[model_name] = 1.0

        return self._normalize_weights(model_scores, "パフォーマンスベース")

    def calculate_sharpe_based_weights(
        self,
        recent_predictions: Dict[str, deque],
        recent_actuals: deque
    ) -> Dict[str, float]:
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

        数学的定義:
        accuracy_returns[i] = pred_return[i] × actual_return[i]
        sharpe_ratio = mean(accuracy_returns) / std(accuracy_returns)
        final_sharpe = max(sharpe_ratio, 0.1)

        Args:
            recent_predictions: モデル別の直近予測値
            recent_actuals: 直近の実際値

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_sharpe = {}

        for model_name in self.model_names:
            if len(recent_predictions[model_name]) >= self.config.min_samples_for_update:
                pred_array = np.array(
                    list(recent_predictions[model_name])
                    [-self.config.min_samples_for_update:]
                )
                actual_array = np.array(
                    list(recent_actuals)[-self.config.min_samples_for_update:]
                )

                # シャープレシオ計算
                sharpe_ratio = self._calculate_sharpe_ratio(pred_array, actual_array)
                
                # 下限クリップ適用
                clipped_sharpe = max(sharpe_ratio, self.config.sharpe_clip_min)
                model_sharpe[model_name] = clipped_sharpe

                # 詳細ログ（カスタマイズ可能）
                if self.config.enable_score_logging or self.config.verbose:
                    logger.debug(
                        f"{model_name} シャープ詳細: 生シャープ={sharpe_ratio:.4f}, "
                        f"クリップ後={clipped_sharpe:.3f} "
                        f"(下限={self.config.sharpe_clip_min})"
                    )
            else:
                # データ不足時の中立値
                model_sharpe[model_name] = 0.5

        return self._normalize_weights(model_sharpe, "シャープベース")

    def calculate_regime_aware_weights(
        self,
        recent_predictions: Dict[str, deque],
        recent_actuals: deque,
        current_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Issue #478対応: 市場状態適応重み調整（外部設定対応）

        Args:
            recent_predictions: モデル別の直近予測値
            recent_actuals: 直近の実際値
            current_regime: 現在の市場状態

        Returns:
            市場状態に応じて調整された重み辞書
        """
        try:
            # 基本パフォーマンスベース重み
            base_weights = self.calculate_performance_based_weights(
                recent_predictions, recent_actuals
            )

            # Issue #478対応: 外部設定されたレジーム調整係数を使用
            regime_adjustments = self.config.regime_adjustments
            if not regime_adjustments:
                logger.warning("レジーム調整設定が見つかりません。基本重みを返します。")
                return base_weights

            # 調整係数適用
            adjusted_weights = {}
            adjustments = regime_adjustments.get(current_regime, {})

            if not adjustments:
                logger.warning(
                    f"現在のレジーム '{current_regime.value}' の"
                    "調整設定が見つかりません。"
                )
                return base_weights

            for model_name in self.model_names:
                base_weight = base_weights.get(model_name, 1.0 / len(self.model_names))
                adjustment = adjustments.get(model_name, 1.0)
                adjusted_weights[model_name] = base_weight * adjustment

            # 正規化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    name: weight / total_weight 
                    for name, weight in adjusted_weights.items()
                }

                if self.config.verbose:
                    changes = []
                    for model_name in self.model_names:
                        adj = adjustments.get(model_name, 1.0)
                        if adj != 1.0:
                            changes.append(f"{model_name}x{adj:.1f}")
                    if changes:
                        logger.info(
                            f"レジーム '{current_regime.value}' 調整: "
                            f"{', '.join(changes)}"
                        )

                return normalized_weights
            else:
                logger.warning("調整後の重み合計が0です。基本重みを返します。")
                return base_weights

        except Exception as e:
            logger.error(f"レジーム認識重み調整エラー: {e}")
            return self.calculate_performance_based_weights(
                recent_predictions, recent_actuals
            )

    def _calculate_direction_score(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        方向的中率を計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            方向的中率（0.0-1.0）
        """
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

        return float(direction_score)

    def _calculate_sharpe_ratio(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        シャープレシオを計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            シャープレシオ
        """
        # 1. リターン計算
        pred_returns = np.diff(pred_array) / pred_array[:-1]
        actual_returns = np.diff(actual_array) / actual_array[:-1]

        # 2. 予測精度リターン計算
        accuracy_returns = pred_returns * actual_returns

        # 3. シャープレシオ計算
        accuracy_std = np.std(accuracy_returns)
        if accuracy_std > 0:
            sharpe_ratio = np.mean(accuracy_returns) / accuracy_std
        else:
            sharpe_ratio = 0.0

        return float(sharpe_ratio)

    def _normalize_weights(
        self,
        model_scores: Dict[str, float],
        method_name: str
    ) -> Dict[str, float]:
        """
        重みの正規化

        Args:
            model_scores: モデル別スコア辞書
            method_name: 手法名（ログ用）

        Returns:
            正規化された重み辞書
        """
        total_score = sum(model_scores.values())
        if total_score > 0:
            normalized_weights = {
                name: score / total_score 
                for name, score in model_scores.items()
            }

            if self.config.verbose:
                logger.info(f"{method_name}重み: {normalized_weights}")

            return normalized_weights
        else:
            # フォールバック: 均等重み
            logger.warning(f"全モデルスコアが0です。均等重みを適用します。")
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def get_available_methods(self) -> List[str]:
        """
        利用可能な重み計算手法の一覧を取得

        Returns:
            重み計算手法名のリスト
        """
        return ["performance_based", "sharpe_based", "regime_aware"]

    def calculate_weights(
        self,
        method: str,
        recent_predictions: Dict[str, deque],
        recent_actuals: deque,
        current_regime: Optional[MarketRegime] = None
    ) -> Dict[str, float]:
        """
        指定された手法で重みを計算

        Args:
            method: 計算手法名
            recent_predictions: モデル別の直近予測値
            recent_actuals: 直近の実際値
            current_regime: 現在の市場状態（regime_aware時のみ必要）

        Returns:
            計算された重み辞書

        Raises:
            ValueError: 無効な手法名が指定された場合
        """
        if method == "performance_based":
            return self.calculate_performance_based_weights(
                recent_predictions, recent_actuals
            )
        elif method == "sharpe_based":
            return self.calculate_sharpe_based_weights(
                recent_predictions, recent_actuals
            )
        elif method == "regime_aware":
            if current_regime is None:
                raise ValueError("regime_aware手法には市場状態の指定が必要です")
            return self.calculate_regime_aware_weights(
                recent_predictions, recent_actuals, current_regime
            )
        else:
            raise ValueError(f"未知の重み計算手法: {method}")

    def get_scoring_explanation(self) -> Dict[str, Dict[str, str]]:
        """
        Issue #477対応: スコアリング手法の説明取得

        Returns:
            各スコアリング手法の詳細説明
        """
        explanations = {
            'performance_based': {
                'description': 'RMSE逆数と方向的中率の重み付き合計',
                'formula': (f'{self.config.accuracy_weight} × (1/(1+RMSE)) + '
                           f'{self.config.direction_weight} × 方向的中率'),
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