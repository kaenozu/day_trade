#!/usr/bin/env python3
"""
Dynamic Weighting System - Weight Constraints

重み制約とモーメンタム処理
"""

import time
from typing import Dict, List, Any
from .core import DynamicWeightingConfig, MarketRegime
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightConstraintManager:
    """重み制約管理クラス"""

    def __init__(self, config: DynamicWeightingConfig, model_names: List[str]):
        """
        初期化

        Args:
            config: 動的重み調整設定
            model_names: モデル名リスト
        """
        self.config = config
        self.model_names = model_names

        # 履歴データ
        self.weight_history = []
        self.total_updates = 0

    def apply_momentum(self, 
                      new_weights: Dict[str, float],
                      current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        モーメンタム適用

        Args:
            new_weights: 新しい重み
            current_weights: 現在の重み

        Returns:
            モーメンタムを適用した重み
        """
        momentum_weights = {}
        momentum = self.config.momentum_factor

        for model_name in self.model_names:
            current = current_weights.get(model_name, 1.0 / len(self.model_names))
            new = new_weights.get(model_name, 1.0 / len(self.model_names))

            # モーメンタム適用: 新重み = (1-momentum) * 新重み + momentum * 現在重み
            momentum_weight = (1 - momentum) * new + momentum * current
            momentum_weights[model_name] = momentum_weight

        return momentum_weights

    def apply_comprehensive_constraints(self,
                                      weights: Dict[str, float],
                                      current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        包括的制約適用

        Issue #479対応: モーメンタム後に全制約を同時適用
        1. 最大変更量制限
        2. 最小・最大重み制限
        3. 合計1.0正規化
        4. 制約競合時の最適解計算

        Args:
            weights: 適用対象の重み
            current_weights: 現在の重み

        Returns:
            制約を適用した重み
        """
        try:
            constrained = {}

            # Step 1: 各モデルの制約適用
            for model_name in self.model_names:
                new_weight = weights.get(model_name, 1.0 / len(self.model_names))
                current_weight = current_weights.get(model_name, 1.0 / len(self.model_names))

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
                        final_weights[model_name] = max(
                            self.config.min_weight,
                            min(self.config.max_weight, normalized_weight)
                        )
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
            return current_weights.copy()

    def validate_and_update_weights(self,
                                  weights: Dict[str, float],
                                  current_weights: Dict[str, float],
                                  current_regime: MarketRegime) -> Dict[str, float]:
        """
        最終検証と重み更新

        Issue #479対応: 制約チェックと安全な重み更新

        Args:
            weights: 新しい重み
            current_weights: 現在の重み
            current_regime: 現在の市場状態

        Returns:
            検証済みの重み（更新が成功した場合は新しい重み、失敗時は現在の重み）
        """
        try:
            # 基本検証
            if not weights or len(weights) != len(self.model_names):
                logger.warning("無効な重み辞書です。現在の重みを維持します。")
                return current_weights

            # 制約検証
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"重み合計が1.0でありません: {total_weight:.6f}")
                return current_weights

            # 個別重み制約検証
            for model_name, weight in weights.items():
                if weight < 0 or weight > 1:
                    logger.warning(f"重み範囲外: {model_name}={weight:.6f}")
                    return current_weights

                if weight < self.config.min_weight or weight > self.config.max_weight:
                    logger.warning(f"設定制約外: {model_name}={weight:.6f} "
                                 f"(範囲: {self.config.min_weight}-{self.config.max_weight})")
                    return current_weights

            # 重み更新実行（履歴記録）
            old_weights = current_weights.copy()

            # 履歴記録
            self.weight_history.append({
                'weights': weights.copy(),
                'timestamp': int(time.time()),
                'regime': current_regime,
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

            return weights

        except Exception as e:
            logger.error(f"重み検証・更新エラー: {e}")
            # エラー時は重みを変更しない
            return current_weights

    def apply_legacy_weight_constraints(self,
                                      weights: Dict[str, float],
                                      current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        従来の重み制約適用（後方互換性のため）

        Args:
            weights: 新しい重み
            current_weights: 現在の重み

        Returns:
            制約を適用した重み
        """
        constrained = {}

        for model_name, new_weight in weights.items():
            current_weight = current_weights.get(model_name, 1.0 / len(self.model_names))

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

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """重み履歴取得"""
        return self.weight_history.copy()

    def get_total_updates(self) -> int:
        """総更新回数取得"""
        return self.total_updates