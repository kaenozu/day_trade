#!/usr/bin/env python3
"""
Dynamic Weighting System - Weighting Algorithms

各種重み計算アルゴリズムの実装
"""

from typing import Dict, List
import numpy as np
from collections import deque

from .core import DynamicWeightingConfig, MarketRegime
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightingAlgorithms:
    """重み計算アルゴリズム集合クラス"""

    def __init__(self, config: DynamicWeightingConfig, model_names: List[str]):
        """
        初期化

        Args:
            config: 動的重み調整設定
            model_names: モデル名リスト
        """
        self.config = config
        self.model_names = model_names

    def performance_based_weighting(self,
                                  recent_predictions: Dict[str, deque],
                                  recent_actuals: deque) -> Dict[str, float]:
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
            recent_predictions: 各モデルの最近の予測値のdeque
            recent_actuals: 最近の実際値のdeque

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_scores = {}

        for model_name in self.model_names:
            if (len(recent_predictions[model_name]) >= self.config.min_samples_for_update and
                len(recent_actuals) >= self.config.min_samples_for_update):

                # パフォーマンスウィンドウ作成
                pred_array = np.array(
                    list(recent_predictions[model_name])[-self.config.min_samples_for_update:]
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

    def sharpe_based_weighting(self,
                             recent_predictions: Dict[str, deque],
                             recent_actuals: deque) -> Dict[str, float]:
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

        Args:
            recent_predictions: 各モデルの最近の予測値のdeque
            recent_actuals: 最近の実際値のdeque

        Returns:
            モデル別重み辞書（正規化済み）
        """
        model_sharpe = {}

        for model_name in self.model_names:
            if len(recent_predictions[model_name]) >= self.config.min_samples_for_update:
                pred_array = np.array(
                    list(recent_predictions[model_name])[-self.config.min_samples_for_update:]
                )
                actual_array = np.array(
                    list(recent_actuals)[-self.config.min_samples_for_update:]
                )

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

    def regime_aware_weighting(self,
                             recent_predictions: Dict[str, deque],
                             recent_actuals: deque,
                             current_regime: MarketRegime) -> Dict[str, float]:
        """
        Issue #478対応: 市場状態適応重み調整（外部設定対応）

        Args:
            recent_predictions: 各モデルの最近の予測値のdeque
            recent_actuals: 最近の実際値のdeque
            current_regime: 現在の市場状態

        Returns:
            市場状態に応じて調整された重み辞書
        """
        try:
            # 基本パフォーマンスベース重み
            base_weights = self.performance_based_weighting(recent_predictions, recent_actuals)

            # Issue #478対応: 外部設定されたレジーム調整係数を使用
            regime_adjustments = self.config.regime_adjustments
            if not regime_adjustments:
                logger.warning("レジーム調整設定が見つかりません。基本重みを返します。")
                return base_weights

            # 調整係数適用
            adjusted_weights = {}
            adjustments = regime_adjustments.get(current_regime, {})

            if not adjustments:
                logger.warning(f"現在のレジーム '{current_regime.value}' の調整設定が見つかりません。")
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
                        logger.info(f"レジーム '{current_regime.value}' 調整: {', '.join(changes)}")

                return normalized_weights
            else:
                logger.warning("調整後の重み合計が0です。基本重みを返します。")
                return base_weights

        except Exception as e:
            logger.error(f"レジーム認識重み調整エラー: {e}")
            return self.performance_based_weighting(recent_predictions, recent_actuals)