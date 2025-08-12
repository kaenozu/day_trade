#!/usr/bin/env python3
"""
Issue #639対応テスト: 戦略クラス登録の型検証強化

OptimizationStrategyFactory.register_strategyメソッドに追加された
包括的な型検証機能をテストする
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


class ValidStrategy(OptimizationStrategy):
    """有効な戦略クラス（テスト用）"""

    def __init__(self, config):
        super().__init__(config)

    def execute(self, *args, **kwargs):
        return "valid execution"

    def get_strategy_name(self):
        return "ValidStrategy"


class IncompleteStrategy(OptimizationStrategy):
    """不完全な戦略クラス（executeメソッドなし）"""

    def __init__(self, config):
        super().__init__(config)

    def get_strategy_name(self):
        return "IncompleteStrategy"

    # executeメソッドが未実装


class AbstractStrategyExample(OptimizationStrategy):
    """抽象戦略クラス（抽象メソッドを持つ）"""

    def __init__(self, config):
        super().__init__(config)

    def get_strategy_name(self):
        return "AbstractStrategy"

    # executeメソッドが抽象（未実装）


class InvalidConstructorStrategy(OptimizationStrategy):
    """無効なコンストラクタを持つ戦略クラス"""

    def __init__(self):  # configパラメータなし
        pass  # super().__init__を呼ばない

    def execute(self, *args, **kwargs):
        return "invalid constructor execution"

    def get_strategy_name(self):
        return "InvalidConstructorStrategy"


class InvalidGetStrategyNameStrategy(OptimizationStrategy):
    """get_strategy_nameが無効な戦略クラス"""

    def __init__(self, config):
        super().__init__(config)

    def execute(self, *args, **kwargs):
        return "execution"

    def get_strategy_name(self):
        return None  # 無効な戻り値


class NonCallableMethodStrategy(OptimizationStrategy):
    """呼び出し可能でないメソッドを持つ戦略クラス"""

    def __init__(self, config):
        super().__init__(config)

    execute = "not a method"  # 関数でない

    def get_strategy_name(self):
        return "NonCallableMethodStrategy"


class TestIssue639StrengthentTypeValidation(unittest.TestCase):
    """Issue #639: 戦略クラス登録型検証強化のテストスイート"""

    def setUp(self):
        """テスト前準備"""
        # グローバル状態をクリア
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None

    def tearDown(self):
        """テスト後クリーンアップ"""
        OptimizationStrategyFactory._strategies.clear()
        OptimizationStrategyFactory._config = None


class TestValidStrategyRegistration(TestIssue639StrengthentTypeValidation):
    """有効な戦略登録のテスト"""

    def test_valid_strategy_registration_success(self):
        """有効な戦略クラスの登録が成功することをテスト"""
        # 有効な戦略クラスの登録
        OptimizationStrategyFactory.register_strategy(
            "valid_test", OptimizationLevel.STANDARD, ValidStrategy
        )

        # 正常に登録されたことを確認
        registered = OptimizationStrategyFactory.get_registered_components()
        self.assertIn("valid_test", registered)
        self.assertIn("standard", registered["valid_test"])

    def test_valid_strategy_can_be_retrieved(self):
        """登録された有効な戦略を取得できることをテスト"""
        OptimizationStrategyFactory.register_strategy(
            "retrieve_test", OptimizationLevel.OPTIMIZED, ValidStrategy
        )

        config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
        strategy = OptimizationStrategyFactory.get_strategy("retrieve_test", config)

        self.assertIsInstance(strategy, ValidStrategy)
        self.assertEqual(strategy.get_strategy_name(), "ValidStrategy")


class TestBasicTypeValidation(TestIssue639StrengthentTypeValidation):
    """基本的な型検証のテスト"""

    def test_non_class_object_rejection(self):
        """クラスでないオブジェクトの登録が拒否されることをテスト"""
        non_class_objects = [
            "string",
            123,
            [],
            {},
            lambda x: x,
            ValidStrategy(OptimizationConfig()),  # インスタンス
        ]

        for obj in non_class_objects:
            with self.subTest(obj=obj):
                with self.assertRaises(TypeError) as cm:
                    OptimizationStrategyFactory.register_strategy(
                        "invalid", OptimizationLevel.STANDARD, obj
                    )

                error_msg = str(cm.exception)
                self.assertIn("戦略登録エラー", error_msg)
                self.assertIn("クラスである必要があります", error_msg)

    def test_non_optimization_strategy_subclass_rejection(self):
        """OptimizationStrategyのサブクラスでないクラスの拒否をテスト"""

        class NotAStrategy:
            def __init__(self, config):
                pass

            def execute(self):
                pass

        with self.assertRaises(TypeError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "not_strategy", OptimizationLevel.STANDARD, NotAStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("OptimizationStrategyのサブクラスである必要があります", error_msg)


class TestAbstractClassValidation(TestIssue639StrengthentTypeValidation):
    """抽象クラス検証のテスト"""

    def test_abstract_strategy_rejection(self):
        """抽象戦略クラスの登録が拒否されることをテスト"""
        # OptimizationStrategy自体は抽象クラス
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "abstract", OptimizationLevel.STANDARD, OptimizationStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("抽象クラスです", error_msg)
        self.assertIn("未実装の抽象メソッド", error_msg)


class TestRequiredMethodsValidation(TestIssue639StrengthentTypeValidation):
    """必須メソッド検証のテスト"""

    def test_missing_execute_method_rejection(self):
        """executeメソッドが欠けている戦略の拒否をテスト"""
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "incomplete", OptimizationLevel.STANDARD, IncompleteStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("必須メソッドが不足", error_msg)
        self.assertIn("execute", error_msg)

    def test_non_callable_method_rejection(self):
        """呼び出し可能でないメソッドを持つ戦略の拒否をテスト"""
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "non_callable", OptimizationLevel.STANDARD, NonCallableMethodStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("必須メソッドが不足", error_msg)
        self.assertIn("execute (not callable)", error_msg)


class TestConstructorValidation(TestIssue639StrengthentTypeValidation):
    """コンストラクタ検証のテスト"""

    def test_invalid_constructor_warning(self):
        """無効なコンストラクタに対する警告をテスト"""
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            with self.assertRaises(ValueError):  # インスタンス化テストで失敗
                OptimizationStrategyFactory.register_strategy(
                    "invalid_constructor", OptimizationLevel.STANDARD, InvalidConstructorStrategy
                )

            # コンストラクタ検証の警告がログに出力されることを確認
            mock_logger.warning.assert_called()
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            constructor_warning = any("コンストラクタ検証をスキップ" in msg for msg in warning_calls)
            self.assertTrue(constructor_warning)


class TestInstanceCreationValidation(TestIssue639StrengthentTypeValidation):
    """インスタンス化検証のテスト"""

    def test_instantiation_failure_rejection(self):
        """インスタンス化に失敗する戦略の拒否をテスト"""
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "invalid_constructor", OptimizationLevel.STANDARD, InvalidConstructorStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("インスタンス化テストに失敗", error_msg)

    def test_invalid_get_strategy_name_rejection(self):
        """無効なget_strategy_name戻り値を持つ戦略の拒否をテスト"""
        with self.assertRaises(ValueError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "invalid_name", OptimizationLevel.STANDARD, InvalidGetStrategyNameStrategy
            )

        error_msg = str(cm.exception)
        self.assertIn("空でない文字列を返す必要があります", error_msg)


class TestValidationLogging(TestIssue639StrengthentTypeValidation):
    """検証ログのテスト"""

    def test_successful_validation_debug_log(self):
        """成功した検証のデバッグログをテスト"""
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            OptimizationStrategyFactory.register_strategy(
                "log_test", OptimizationLevel.DEBUG, ValidStrategy
            )

            # デバッグログが呼ばれたことを確認
            mock_logger.debug.assert_called()
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            validation_success = any("戦略クラス検証成功" in msg for msg in debug_calls)
            self.assertTrue(validation_success)

    def test_error_messages_contain_context(self):
        """エラーメッセージにコンテキスト情報が含まれることをテスト"""
        with self.assertRaises(TypeError) as cm:
            OptimizationStrategyFactory.register_strategy(
                "context_test", OptimizationLevel.GPU_ACCELERATED, "not_a_class"
            )

        error_msg = str(cm.exception)
        self.assertIn("context_test", error_msg)
        self.assertIn("gpu_accelerated", error_msg)
        self.assertIn("戦略登録エラー", error_msg)


class TestValidationBypass(TestIssue639StrengthentTypeValidation):
    """検証バイパスのテスト"""

    def test_validation_robustness_under_edge_cases(self):
        """エッジケースでの検証の堅牢性をテスト"""

        class EdgeCaseStrategy(OptimizationStrategy):
            def __init__(self, config, *args, **kwargs):  # 追加パラメータ
                super().__init__(config)

            def execute(self, *args, **kwargs):
                return "edge case execution"

            def get_strategy_name(self):
                return "EdgeCaseStrategy"

        # エッジケース戦略が正常に登録されることを確認
        OptimizationStrategyFactory.register_strategy(
            "edge_case", OptimizationLevel.STANDARD, EdgeCaseStrategy
        )

        # 登録確認
        registered = OptimizationStrategyFactory.get_registered_components()
        self.assertIn("edge_case", registered)


class TestValidationPerformance(TestIssue639StrengthentTypeValidation):
    """検証パフォーマンスのテスト"""

    def test_validation_does_not_significantly_slow_registration(self):
        """検証が登録を大幅に遅くしないことをテスト"""
        import time

        # 複数の戦略を登録して時間を測定
        start_time = time.time()

        for i in range(10):

            class TestStrategy(OptimizationStrategy):
                def __init__(self, config):
                    super().__init__(config)

                def execute(self, *args, **kwargs):
                    return f"execution {i}"

                def get_strategy_name(self):
                    return f"TestStrategy{i}"

            OptimizationStrategyFactory.register_strategy(
                f"perf_test_{i}", OptimizationLevel.STANDARD, TestStrategy
            )

        end_time = time.time()
        registration_time = end_time - start_time

        # 10個の戦略登録が1秒以内に完了することを確認
        self.assertLess(registration_time, 1.0,
                       f"戦略登録が遅すぎます: {registration_time:.3f}秒")


if __name__ == "__main__":
    unittest.main(verbosity=2)