#!/usr/bin/env python3
"""
Issue #640対応テスト: フォールバックロジックの簡潔化（簡単版）

基本的なフォールバック機能の動作確認
"""

import unittest
import sys
import os
from unittest.mock import patch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.day_trade.core.optimization_strategy import (
    OptimizationStrategyFactory,
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy,
)


class SimpleTestStrategy(OptimizationStrategy):
    """シンプルなテスト戦略"""

    def __init__(self, config, name="SimpleTestStrategy"):
        super().__init__(name, {})
        self.config = config
        self._name = name

    def execute(self, *args, **kwargs):
        return f"Executed: {self._name}"

    def get_strategy_name(self):
        return self._name


class StandardStrategy(SimpleTestStrategy):
    def __init__(self, config):
        super().__init__(config, "StandardStrategy")


class OptimizedStrategy(SimpleTestStrategy):
    def __init__(self, config):
        super().__init__(config, "OptimizedStrategy")


class DebugStrategy(SimpleTestStrategy):
    def __init__(self, config):
        super().__init__(config, "DebugStrategy")


class TestIssue640SimpleFallbackLogic(unittest.TestCase):
    """Issue #640: フォールバックロジック簡潔化のシンプルテスト"""

    def setUp(self):
        """テスト前準備"""
        # グローバル状態をクリア
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None

    def tearDown(self):
        """テスト後クリーンアップ"""
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None


class TestBasicFallback(TestIssue640SimpleFallbackLogic):
    """基本的なフォールバック機能テスト"""

    def test_optimized_fallback_priority(self):
        """OPTIMIZED戦略が最優先でフォールバックされることを確認"""
        # STANDARD, OPTIMIZED, DEBUG戦略を登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)

        # 存在しないGPU_ACCELERATEDレベルを要求（フォールバックが実際に動作する）
        config = OptimizationConfig(level=OptimizationLevel.GPU_ACCELERATED, auto_fallback=True)

        # 戦略を取得
        strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

        # OPTIMIZED戦略が選択されることを確認
        self.assertEqual(strategy.get_strategy_name(), "OptimizedStrategy")

    def test_standard_fallback_when_optimized_unavailable(self):
        """OPTIMIZED戦略がない場合にSTANDARDにフォールバックすることを確認"""
        # STANDARDとDEBUG戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)

        # 存在しないGPU_ACCELERATEDレベルを要求
        config = OptimizationConfig(level=OptimizationLevel.GPU_ACCELERATED, auto_fallback=True)

        # 戦略を取得
        strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

        # STANDARD戦略が選択されることを確認
        self.assertEqual(strategy.get_strategy_name(), "StandardStrategy")

    def test_debug_fallback_when_primary_levels_unavailable(self):
        """主要レベルがない場合にDEBUGにフォールバックすることを確認"""
        # DEBUGとADAPTIVE戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)

        # 存在しないSTANDARDレベルを要求
        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=True)

        # 戦略を取得
        strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

        # DEBUG戦略が選択されることを確認
        self.assertEqual(strategy.get_strategy_name(), "DebugStrategy")

    def test_no_fallback_when_disabled(self):
        """フォールバック無効時に例外が発生することを確認"""
        # OPTIMIZED戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)

        # フォールバック無効でSTANDARD戦略を要求
        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=False)

        # 例外が発生することを確認
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.get_strategy("test_comp", config)

        # エラーメッセージにコンポーネント名とレベルが含まれることを確認
        error_message = str(cm.exception)
        self.assertIn("test_comp", error_message)
        self.assertIn("standard", error_message)

    def test_fallback_deterministic_behavior(self):
        """フォールバック動作が決定論的であることを確認"""
        # 複数の戦略を登録
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.DEBUG, DebugStrategy)

        config = OptimizationConfig(level=OptimizationLevel.GPU_ACCELERATED, auto_fallback=True)

        # 複数回実行して同じ結果が得られることを確認
        results = []
        for _ in range(5):
            strategy = OptimizationStrategyFactory.get_strategy("det_comp", config)
            results.append(strategy.get_strategy_name())

        # 全ての結果が同じであることを確認
        self.assertTrue(all(result == results[0] for result in results))
        # OPTIMIZED が最優先選択されることを確認
        self.assertEqual(results[0], "OptimizedStrategy")


if __name__ == "__main__":
    unittest.main(verbosity=2)