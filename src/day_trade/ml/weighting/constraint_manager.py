#!/usr/bin/env python3
"""
Weight constraint management utilities

このモジュールは重み制約の適用とバリデーション機能を提供します。
リスク管理のための最小/最大重み制限、変更量制限等を管理します。
"""

from typing import Dict, List, Any
import numpy as np

from .core import DynamicWeightingConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightConstraintManager:
    """
    重み制約管理クラス

    重みの各種制約（最小/最大重み、変更量制限等）の
    適用と検証を行います。
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

    def apply_comprehensive_constraints(
        self,
        weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        包括的制約適用

        Issue #479対応: 全制約を同時に適用し最適解を計算
        1. 最大変更量制限
        2. 最小・最大重み制限
        3. 合計1.0正規化
        4. 制約競合時の最適解計算

        Args:
            weights: 制約適用対象の重み
            current_weights: 現在の重み

        Returns:
            制約適用後の重み
        """
        try:
            constrained = {}

            # Step 1: 各モデルの制約適用
            for model_name in self.model_names:
                new_weight = weights.get(model_name, 1.0 / len(self.model_names))
                current_weight = current_weights.get(
                    model_name, 1.0 / len(self.model_names)
                )

                # 最大変更量制限
                constrained_weight = self._apply_change_limit(
                    new_weight, current_weight
                )
                
                # 最小・最大重み制限
                constrained_weight = self._apply_weight_bounds(constrained_weight)

                constrained[model_name] = constrained_weight

            # Step 2: 合計正規化
            normalized_weights = self._normalize_weights(constrained)

            # Step 3: 正規化後の制約再チェック
            final_weights = self._recheck_constraints_after_normalization(
                normalized_weights
            )

            return final_weights

        except Exception as e:
            logger.error(f"包括的制約適用エラー: {e}")
            # エラー時は現在の重みを維持
            return current_weights.copy()

    def _apply_change_limit(self, new_weight: float, current_weight: float) -> float:
        """
        最大変更量制限の適用

        Args:
            new_weight: 新しい重み
            current_weight: 現在の重み

        Returns:
            変更量制限適用後の重み
        """
        max_change = self.config.max_weight_change
        
        if new_weight > current_weight + max_change:
            return current_weight + max_change
        elif new_weight < current_weight - max_change:
            return current_weight - max_change
        else:
            return new_weight

    def _apply_weight_bounds(self, weight: float) -> float:
        """
        重み境界制限の適用

        Args:
            weight: 制限対象の重み

        Returns:
            境界制限適用後の重み
        """
        return max(
            self.config.min_weight,
            min(self.config.max_weight, weight)
        )

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        重みの正規化

        Args:
            weights: 正規化対象の重み

        Returns:
            正規化後の重み（合計1.0）
        """
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            return {
                name: weight / total_weight 
                for name, weight in weights.items()
            }
        else:
            # フォールバック: 均等分散
            logger.warning("制約適用後に重み合計が0になりました。均等分散を適用します。")
            return {name: 1.0 / len(self.model_names) for name in self.model_names}

    def _recheck_constraints_after_normalization(
        self, 
        normalized_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        正規化後の制約再チェック

        Args:
            normalized_weights: 正規化後の重み

        Returns:
            制約再チェック後の最終重み
        """
        final_weights = {}
        needs_rebalancing = False

        for model_name, normalized_weight in normalized_weights.items():
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

        # リバランシング必要時の最終調整
        if needs_rebalancing:
            final_total = sum(final_weights.values())
            if final_total > 0:
                # 最終正規化
                final_weights = {
                    name: weight / final_total 
                    for name, weight in final_weights.items()
                }

        return final_weights

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        重みの妥当性検証

        Args:
            weights: 検証対象の重み辞書

        Returns:
            検証結果辞書
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }

        try:
            # 基本検証
            if not weights or len(weights) != len(self.model_names):
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"無効な重み辞書: 期待{len(self.model_names)}モデル, "
                    f"実際{len(weights)}モデル"
                )
                return validation_result

            # 制約検証
            total_weight = sum(weights.values())
            validation_result['summary']['total_weight'] = total_weight

            if abs(total_weight - 1.0) > 1e-6:
                validation_result['warnings'].append(
                    f"重み合計が1.0でありません: {total_weight:.6f}"
                )

            # 個別重み制約検証
            constraint_violations = []
            for model_name, weight in weights.items():
                if weight < 0 or weight > 1:
                    constraint_violations.append(
                        f"{model_name}={weight:.6f} (範囲外: 0-1)"
                    )
                elif weight < self.config.min_weight:
                    constraint_violations.append(
                        f"{model_name}={weight:.6f} "
                        f"(最小制約違反: {self.config.min_weight})"
                    )
                elif weight > self.config.max_weight:
                    constraint_violations.append(
                        f"{model_name}={weight:.6f} "
                        f"(最大制約違反: {self.config.max_weight})"
                    )

            if constraint_violations:
                validation_result['valid'] = False
                validation_result['errors'].extend(constraint_violations)

            # 統計情報
            weight_values = list(weights.values())
            validation_result['summary'].update({
                'min_weight': min(weight_values),
                'max_weight': max(weight_values),
                'weight_std': float(np.std(weight_values)),
                'weight_entropy': self._calculate_weight_entropy(weight_values)
            })

        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"検証処理エラー: {e}")

        return validation_result

    def _calculate_weight_entropy(self, weights: List[float]) -> float:
        """
        重み分散のエントロピーを計算

        Args:
            weights: 重みのリスト

        Returns:
            エントロピー値（高いほど均等分散）
        """
        try:
            # 0除算回避
            safe_weights = [max(w, 1e-10) for w in weights]
            entropy = -sum(w * np.log(w) for w in safe_weights)
            return float(entropy)
        except Exception:
            return 0.0

    def check_constraint_conflicts(
        self,
        new_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        制約競合の事前チェック

        Args:
            new_weights: 新しい重み
            current_weights: 現在の重み

        Returns:
            制約競合分析結果
        """
        conflicts = {
            'has_conflicts': False,
            'conflict_details': [],
            'recommended_adjustments': []
        }

        try:
            for model_name in self.model_names:
                new_weight = new_weights.get(model_name, 1.0 / len(self.model_names))
                current_weight = current_weights.get(
                    model_name, 1.0 / len(self.model_names)
                )

                # 変更量制限チェック
                change = abs(new_weight - current_weight)
                if change > self.config.max_weight_change:
                    conflicts['has_conflicts'] = True
                    conflicts['conflict_details'].append(
                        f"{model_name}: 変更量超過 "
                        f"({change:.3f} > {self.config.max_weight_change})"
                    )

                # 境界制限チェック
                if new_weight < self.config.min_weight:
                    conflicts['has_conflicts'] = True
                    conflicts['conflict_details'].append(
                        f"{model_name}: 最小重み未満 "
                        f"({new_weight:.3f} < {self.config.min_weight})"
                    )
                elif new_weight > self.config.max_weight:
                    conflicts['has_conflicts'] = True
                    conflicts['conflict_details'].append(
                        f"{model_name}: 最大重み超過 "
                        f"({new_weight:.3f} > {self.config.max_weight})"
                    )

            # 合計制約チェック
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                conflicts['has_conflicts'] = True
                conflicts['conflict_details'].append(
                    f"重み合計が1.0ではありません ({total_weight:.6f})"
                )

            # 推奨調整の生成
            if conflicts['has_conflicts']:
                conflicts['recommended_adjustments'] = self._generate_adjustment_recommendations(
                    new_weights, current_weights
                )

        except Exception as e:
            conflicts['error'] = str(e)

        return conflicts

    def _generate_adjustment_recommendations(
        self,
        new_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> List[str]:
        """
        制約競合解決のための調整推奨事項を生成

        Args:
            new_weights: 新しい重み
            current_weights: 現在の重み

        Returns:
            調整推奨事項のリスト
        """
        recommendations = []

        try:
            # 大きな変更があるモデルを特定
            large_changes = []
            for model_name in self.model_names:
                new_w = new_weights.get(model_name, 1.0 / len(self.model_names))
                current_w = current_weights.get(model_name, 1.0 / len(self.model_names))
                change = abs(new_w - current_w)
                
                if change > self.config.max_weight_change:
                    large_changes.append((model_name, change))

            if large_changes:
                # 変更量が大きい順にソート
                large_changes.sort(key=lambda x: x[1], reverse=True)
                top_model = large_changes[0][0]
                recommendations.append(
                    f"最大変更量制限の緩和を検討 (現在: {self.config.max_weight_change})"
                )
                recommendations.append(
                    f"{top_model}の重み変更を段階的に実行することを推奨"
                )

            # 境界制約違反の対処
            boundary_violations = []
            for model_name, weight in new_weights.items():
                if weight < self.config.min_weight or weight > self.config.max_weight:
                    boundary_violations.append(model_name)

            if boundary_violations:
                recommendations.append(
                    f"重み境界制限の調整を検討: {', '.join(boundary_violations)}"
                )

        except Exception as e:
            recommendations.append(f"推奨事項生成エラー: {e}")

        return recommendations