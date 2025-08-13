#!/usr/bin/env python3
"""
Issue #640対応テスト: フォールバックロジックの簡潔化

OptimizationStrategyFactory.get_strategyメソッドのフォールバックロジックを
簡潔化し、透明性と予測可能性を向上させたことをテストする
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.day_trade.core.optimization_strategy import (
    OptimizationStrategyFactory,
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy,
)


class MockStrategy(OptimizationStrategy):
    """テスト用モック戦略"""

    def __init__(self, config, name="MockStrategy"):
        super().__init__(name, config.level.value if hasattr(config, 'level') else {})
        self.config = config

    def execute(self, *args, **kwargs):
        return f"Mock execution: {self.get_strategy_name()}"

    def get_strategy_name(self):
        return self.model_name


class StandardStrategy(MockStrategy):
    def __init__(self, config):
        super().__init__(config, "StandardStrategy")


class OptimizedStrategy(MockStrategy):
    def __init__(self, config):
        super().__init__(config, "OptimizedStrategy")


class DebugStrategy(MockStrategy):
    def __init__(self, config):
        super().__init__(config, "DebugStrategy")


class AdaptiveStrategy(MockStrategy):
    def __init__(self, config):
        super().__init__(config, "AdaptiveStrategy")


class GpuStrategy(MockStrategy):
    def __init__(self, config):
        super().__init__(config, "GpuStrategy")


class TestIssue640SimplifyFallbackLogic(unittest.TestCase):
    """Issue #640: フォールバックロジック簡潔化のテストスイート"""

    def setUp(self):
        """テスト前準備"""
        # クリーンな状態でテスト開始
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None

    def tearDown(self):
        """テスト後クリーンアップ"""
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None


class TestFallbackHierarchy(TestIssue640SimplifyFallbackLogic):
    """フォールバック階層のテスト"""

    def test_optimized_fallback_priority(self):
        """OPTIMIZED戦略が最優先フォールバックされることをテスト"""
        # STANDARD, OPTIMIZED, DEBUG戦略を登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)

        # 存在しないADAPTIVEレベルを要求 → OPTIMIZEDに優先的にフォールバック
        config = OptimizationConfig(level=OptimizationLevel.ADAPTIVE, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

            # OPTIMIZED戦略が選択されることを確認
            self.assertEqual(strategy.get_strategy_name(), "OptimizedStrategy")

            # 適切なログが出力されることを確認
            mock_logger.warning.assert_called_once()
            log_args = mock_logger.warning.call_args[0][0]
            self.assertIn("adaptive -> optimized", log_args)
            self.assertIn("優先順位に基づく選択", log_args)

    def test_standard_fallback_when_optimized_unavailable(self):
        """OPTIMIZED戦略が利用できない場合にSTANDARDにフォールバックすることをテスト"""
        # STANDARDとDEBUG戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)

        # 存在しないADAPTIVEレベルを要求 → STANDARDにフォールバック
        config = OptimizationConfig(level=OptimizationLevel.ADAPTIVE, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

            # STANDARD戦略が選択されることを確認
            self.assertEqual(strategy.get_strategy_name(), "StandardStrategy")

            # 適切なログが出力されることを確認
            mock_logger.warning.assert_called_once()
            log_args = mock_logger.warning.call_args[0][0]
            self.assertIn("adaptive -> standard", log_args)

    def test_debug_fallback_when_primary_levels_unavailable(self):
        """主要レベルが利用できない場合にDEBUGにフォールバックすることをテスト"""
        # DEBUGとGPU戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.DEBUG, DebugStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.GPU_ACCELERATED, GpuStrategy)

        # 存在しないADAPTIVEレベルを要求 → DEBUGにフォールバック
        config = OptimizationConfig(level=OptimizationLevel.ADAPTIVE, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

            # DEBUG戦略が選択されることを確認
            self.assertEqual(strategy.get_strategy_name(), "DebugStrategy")

            # 適切なログが出力されることを確認
            mock_logger.warning.assert_called_once()
            log_args = mock_logger.warning.call_args[0][0]
            self.assertIn("adaptive -> debug", log_args)

    def test_complete_fallback_hierarchy_order(self):
        """完全なフォールバック階層順序のテスト"""
        # 全レベルの戦略を登録
        strategies = {
            OptimizationLevel.STANDARD: StandardStrategy,
            OptimizationLevel.OPTIMIZED: OptimizedStrategy,
            OptimizationLevel.DEBUG: DebugStrategy,
            OptimizationLevel.ADAPTIVE: AdaptiveStrategy,
            OptimizationLevel.GPU_ACCELERATED: GpuStrategy,
        }

        for level, strategy_cls in strategies.items():
            OptimizationStrategyFactory.register_strategy("test_comp", level, strategy_cls)

        # 各レベルから除外して階層を確認
        target_levels = [
            OptimizationLevel.STANDARD,
            OptimizationLevel.OPTIMIZED,
            OptimizationLevel.DEBUG,
            OptimizationLevel.ADAPTIVE,
            OptimizationLevel.GPU_ACCELERATED,
        ]

        expected_fallbacks = {
            OptimizationLevel.STANDARD: "OptimizedStrategy",      # OPTIMIZED -> ...
            OptimizationLevel.OPTIMIZED: "StandardStrategy",     # STANDARD -> ...
            OptimizationLevel.DEBUG: "OptimizedStrategy",        # OPTIMIZED -> ...
            OptimizationLevel.ADAPTIVE: "OptimizedStrategy",     # OPTIMIZED -> ...
            OptimizationLevel.GPU_ACCELERATED: "OptimizedStrategy", # OPTIMIZED -> ...
        }

        for target_level in target_levels:
            with self.subTest(target_level=target_level):
                # 当該レベルを一時的に削除
                original_strategy = strategies[target_level]
                del OptimizationStrategyFactory._strategies["test_comp"][target_level]

                config = OptimizationConfig(level=target_level, auto_fallback=True)

                try:
                    strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)
                    expected_name = expected_fallbacks[target_level]
                    self.assertEqual(strategy.get_strategy_name(), expected_name)
                finally:
                    # 戦略を復元
                    OptimizationStrategyFactory._strategies["test_comp"][target_level] = original_strategy


class TestNonStandardLevelFallback(TestIssue640SimplifyFallbackLogic):
    """非標準レベルフォールバックのテスト"""

    def test_custom_level_alphabetical_fallback(self):
        """カスタムレベルのアルファベット順フォールバックテスト"""
        # 仮想的なカスタムレベル（EnumにはないがDictキーとして存在）

        # テスト用にカスタム戦略クラスを動的作成
        class CustomAStrategy(MockStrategy):
            def __init__(self, config):
                super().__init__(config, "CustomAStrategy")

        class CustomBStrategy(MockStrategy):
            def __init__(self, config):
                super().__init__(config, "CustomBStrategy")

        # 標準レベルを持たず、カスタムレベルのみ持つコンポーネント
        # 注意: 実際のOptimizationLevelのEnumを直接変更することはできないため、
        # このテストは概念的な検証として実装

        # 代わりに、標準レベルが全て除外された状況をシミュレート
        OptimizationStrategyFactory.register_strategy("custom_comp", OptimizationLevel.DEBUG, CustomAStrategy)
        OptimizationStrategyFactory.register_strategy("custom_comp", OptimizationLevel.GPU_ACCELERATED, CustomBStrategy)

        # STANDARD要求 → DEBUG (アルファベット順で先)にフォールバック
        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            strategy = OptimizationStrategyFactory.get_strategy("custom_comp", config)

            # DEBUG戦略が選択されることを確認（階層順優先）
            self.assertEqual(strategy.get_strategy_name(), "CustomAStrategy")


class TestFallbackDisabled(TestIssue640SimplifyFallbackLogic):
    """フォールバック無効化のテスト"""

    def test_no_fallback_when_disabled(self):
        """フォールバック無効時に例外が発生することをテスト"""
        # OPTIMIZED戦略のみ登録
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)

        # フォールバック無効でSTANDARD戦略を要求
        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=False)

        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.get_strategy("test_comp", config)

        # 適切なエラーメッセージが含まれることを確認
        error_message = str(cm.exception)
        self.assertIn("利用可能な戦略なし", error_message)
        self.assertIn("test_comp", error_message)
        self.assertIn("standard", error_message)


class TestFallbackLogging(TestIssue640SimplifyFallbackLogic):
    """フォールバックログのテスト"""

    def test_fallback_logging_contains_rationale(self):
        """フォールバックログに選択理由が含まれることをテスト"""
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("test_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)

        config = OptimizationConfig(level=OptimizationLevel.DEBUG, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            strategy = OptimizationStrategyFactory.get_strategy("test_comp", config)

            # ログが呼ばれたことを確認
            mock_logger.warning.assert_called_once()

            # ログメッセージの内容確認
            log_message = mock_logger.warning.call_args[0][0]
            self.assertIn("戦略フォールバック", log_message)
            self.assertIn("test_comp", log_message)
            self.assertIn("debug -> optimized", log_message)
            self.assertIn("優先順位に基づく選択", log_message)

    def test_no_fallback_strategy_error_logging(self):
        """フォールバック戦略が見つからない場合のエラーログテスト"""
        # 戦略を登録しない（空の状態）
        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=True)

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            with self.assertRaises(ValueError):
                OptimizationStrategyFactory.get_strategy("empty_comp", config)

            # エラーログは呼ばれない（戦略が未登録のため別のパスでエラー）


class TestTransparencyAndPredictability(TestIssue640SimplifyFallbackLogic):
    """透明性と予測可能性のテスト"""

    def test_fallback_behavior_is_deterministic(self):
        """フォールバック動作が決定論的であることをテスト"""
        # 同じ条件で複数回実行し、同じ結果が得られることを確認
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.STANDARD, StandardStrategy)
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.OPTIMIZED, OptimizedStrategy)
        OptimizationStrategyFactory.register_strategy("det_comp", OptimizationLevel.DEBUG, DebugStrategy)

        config = OptimizationConfig(level=OptimizationLevel.ADAPTIVE, auto_fallback=True)

        # 10回実行して同じ結果が得られることを確認
        results = []
        for _ in range(10):
            strategy = OptimizationStrategyFactory.get_strategy("det_comp", config)
            results.append(strategy.get_strategy_name())

        # 全ての結果が同じであることを確認
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(results[0], "OptimizedStrategy")  # OPTIMIZED が最優先

    def test_fallback_order_documented_behavior(self):
        """文書化されたフォールバック順序通りの動作テスト"""
        # 文書化された順序: OPTIMIZED -> STANDARD -> DEBUG -> ADAPTIVE -> GPU_ACCELERATED

        strategies_to_register = [
            (OptimizationLevel.DEBUG, DebugStrategy),
            (OptimizationLevel.GPU_ACCELERATED, GpuStrategy),
            (OptimizationLevel.ADAPTIVE, AdaptiveStrategy),
        ]

        for level, strategy_cls in strategies_to_register:
            OptimizationStrategyFactory.register_strategy("order_test", level, strategy_cls)

        config = OptimizationConfig(level=OptimizationLevel.STANDARD, auto_fallback=True)

        # DEBUG が選択されることを確認（階層3位）
        strategy = OptimizationStrategyFactory.get_strategy("order_test", config)
        self.assertEqual(strategy.get_strategy_name(), "DebugStrategy")


if __name__ == "__main__":
    unittest.main(verbosity=2)