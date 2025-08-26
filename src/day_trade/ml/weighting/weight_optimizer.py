#!/usr/bin/env python3
"""
Weight optimization and constraint management module

このモジュールは重み制約の適用、モーメンタム処理、
重み最適化を担当します。リスク管理のための重み制限や
滑らかな重み変化を実現するモーメンタム機能を提供します。
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np

from .core import DynamicWeightingConfig, MarketRegime
from .constraint_manager import WeightConstraintManager
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightOptimizer:
    """
    重み最適化・制約管理クラス

    重みの制約適用、モーメンタム処理、最適化を行います。
    リスク管理と安定性の両立を図ります。
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
        
        # 制約マネージャ
        self.constraint_manager = WeightConstraintManager(model_names, config)

    def optimize_weights(
        self,
        new_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_updates: int
    ) -> Dict[str, float]:
        """
        Issue #479対応: 重み制約とモメンタム適用順序の最適化
        
        1. 基本重み計算（外部で実行済み）
        2. モーメンタム適用（制約前）
        3. 包括的制約適用（min/max/sum/change制限）
        4. 最終正規化保証

        Args:
            new_weights: 新しく計算された重み
            current_weights: 現在の重み
            total_updates: 総更新回数

        Returns:
            最適化された重み辞書
        """
        try:
            # Step 1: モーメンタム適用（制約前に実行）
            if self.config.momentum_factor > 0:
                momentum_weights = self._apply_momentum(new_weights, current_weights)
            else:
                momentum_weights = new_weights

            # Step 2: 包括的制約適用（モーメンタム後に実行）
            final_weights = self.constraint_manager.apply_comprehensive_constraints(
                momentum_weights, current_weights
            )

            return final_weights

        except Exception as e:
            logger.error(f"重み最適化エラー: {e}")
            # エラー時は現在の重みを維持（安全な動作）
            return current_weights.copy()

    def _apply_momentum(
        self,
        new_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        モーメンタム適用

        Args:
            new_weights: 新しい重み
            current_weights: 現在の重み

        Returns:
            モーメンタム適用後の重み
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


    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        重みの妥当性検証（WeightConstraintManagerに委譲）

        Args:
            weights: 検証対象の重み辞書

        Returns:
            検証結果辞書
        """
        return self.constraint_manager.validate_weights(weights)

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

    def get_optimization_summary(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        最適化結果の要約を取得

        Args:
            old_weights: 最適化前の重み
            new_weights: 最適化後の重み

        Returns:
            最適化要約辞書
        """
        summary = {
            'weight_changes': {},
            'total_change': 0.0,
            'max_change': 0.0,
            'applied_constraints': [],
            'optimization_stats': {}
        }

        try:
            # 重み変更の分析
            total_change = 0.0
            max_change = 0.0
            significant_changes = []

            for model_name in self.model_names:
                old_w = old_weights.get(model_name, 0)
                new_w = new_weights.get(model_name, 0)
                change = new_w - old_w
                abs_change = abs(change)

                summary['weight_changes'][model_name] = {
                    'old': old_w,
                    'new': new_w,
                    'change': change,
                    'change_percent': (change / old_w * 100) if old_w > 0 else 0
                }

                total_change += abs_change
                max_change = max(max_change, abs_change)

                if abs_change > 0.01:  # 1%以上の変化
                    significant_changes.append(
                        f"{model_name}: {old_w:.3f}→{new_w:.3f} ({change:+.3f})"
                    )

            summary['total_change'] = total_change
            summary['max_change'] = max_change
            summary['significant_changes'] = significant_changes

            # 制約適用状況の分析
            applied_constraints = []
            for model_name, weight in new_weights.items():
                if weight == self.config.min_weight:
                    applied_constraints.append(f"{model_name}: 最小制約適用")
                elif weight == self.config.max_weight:
                    applied_constraints.append(f"{model_name}: 最大制約適用")

            summary['applied_constraints'] = applied_constraints

            # 最適化統計
            weight_values = list(new_weights.values())
            summary['optimization_stats'] = {
                'weight_diversity': float(np.std(weight_values)),
                'entropy': self._calculate_weight_entropy(weight_values),
                'concentration_ratio': max(weight_values) / min(weight_values) if min(weight_values) > 0 else float('inf')
            }

        except Exception as e:
            logger.error(f"最適化要約作成エラー: {e}")
            summary['error'] = str(e)

        return summary

    def suggest_config_adjustments(
        self,
        optimization_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        最適化履歴に基づく設定調整提案

        Args:
            optimization_history: 最適化履歴のリスト

        Returns:
            設定調整提案辞書
        """
        suggestions = {
            'adjustments': [],
            'analysis': {},
            'confidence': 'low'
        }

        if len(optimization_history) < 10:
            suggestions['analysis']['insufficient_data'] = True
            return suggestions

        try:
            # 制約違反の頻度分析
            constraint_violations = sum(
                1 for hist in optimization_history 
                if hist.get('applied_constraints')
            )
            
            violation_rate = constraint_violations / len(optimization_history)
            
            # 重み変更の振幅分析
            changes = [hist.get('max_change', 0) for hist in optimization_history]
            avg_change = np.mean(changes)
            std_change = np.std(changes)

            suggestions['analysis'] = {
                'constraint_violation_rate': violation_rate,
                'avg_weight_change': avg_change,
                'change_volatility': std_change,
                'total_optimizations': len(optimization_history)
            }

            # 提案生成
            if violation_rate > 0.5:
                suggestions['adjustments'].append({
                    'parameter': 'max_weight_change',
                    'current': self.config.max_weight_change,
                    'suggested': self.config.max_weight_change * 1.2,
                    'reason': '制約違反が多いため変更上限を緩和'
                })

            if avg_change < self.config.max_weight_change * 0.1:
                suggestions['adjustments'].append({
                    'parameter': 'momentum_factor',
                    'current': self.config.momentum_factor,
                    'suggested': max(0.01, self.config.momentum_factor - 0.05),
                    'reason': '重み変更が小さいためモーメンタムを削減'
                })

            if std_change > avg_change:
                suggestions['adjustments'].append({
                    'parameter': 'momentum_factor',
                    'current': self.config.momentum_factor,
                    'suggested': min(0.9, self.config.momentum_factor + 0.05),
                    'reason': '重み変更の変動が大きいためモーメンタムを増加'
                })

            suggestions['confidence'] = (
                'high' if len(optimization_history) > 50 else
                'medium' if len(optimization_history) > 25 else
                'low'
            )

        except Exception as e:
            logger.error(f"設定調整提案作成エラー: {e}")
            suggestions['error'] = str(e)

        return suggestions