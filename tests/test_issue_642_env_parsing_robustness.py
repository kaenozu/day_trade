#!/usr/bin/env python3
"""
Issue #642対応テスト: 環境変数パース機能の堅牢性テスト

OptimizationConfig._parse_env_bool()と_parse_env_int()メソッドの
堅牢な環境変数パース機能をテストする
"""

import os
import pytest
import unittest
from unittest.mock import patch

from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel


class TestIssue642EnvironmentParsingRobustness(unittest.TestCase):
    """Issue #642: 環境変数パース堅牢性テストスイート"""

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用環境変数を削除
        test_env_vars = [
            "DAYTRADE_OPTIMIZATION_LEVEL",
            "DAYTRADE_AUTO_FALLBACK",
            "DAYTRADE_PERF_MONITORING",
            "DAYTRADE_CACHE_ENABLED",
            "DAYTRADE_PARALLEL",
            "DAYTRADE_BATCH_SIZE",
            "DAYTRADE_TIMEOUT",
            "DAYTRADE_MEMORY_LIMIT",
            "CI",
        ]

        for var in test_env_vars:
            if var in os.environ:
                del os.environ[var]


class TestParseEnvBool(TestIssue642EnvironmentParsingRobustness):
    """_parse_env_bool静的メソッドのテスト"""

    def test_true_values(self):
        """True値のパーステスト"""
        true_values = ["true", "TRUE", "True", "yes", "YES", "Yes",
                      "1", "on", "ON", "On", "enable", "ENABLE",
                      "enabled", "ENABLED", "y", "Y", "t", "T"]

        for value in true_values:
            with self.subTest(value=value):
                os.environ["TEST_BOOL"] = value
                result = OptimizationConfig._parse_env_bool("TEST_BOOL", False)
                self.assertTrue(result, f"'{value}' should parse as True")
                del os.environ["TEST_BOOL"]

    def test_false_values(self):
        """False値のパーステスト"""
        false_values = ["false", "FALSE", "False", "no", "NO", "No",
                       "0", "off", "OFF", "Off", "disable", "DISABLE",
                       "disabled", "DISABLED", "n", "N", "f", "F"]

        for value in false_values:
            with self.subTest(value=value):
                os.environ["TEST_BOOL"] = value
                result = OptimizationConfig._parse_env_bool("TEST_BOOL", True)
                self.assertFalse(result, f"'{value}' should parse as False")
                del os.environ["TEST_BOOL"]

    def test_whitespace_handling(self):
        """空白文字処理のテスト"""
        test_cases = [
            ("  true  ", True),
            ("\ttrue\t", True),
            ("\nfalse\n", False),
            ("  YES  ", True),
            ("  no  ", False),
        ]

        for value, expected in test_cases:
            with self.subTest(value=repr(value)):
                os.environ["TEST_BOOL"] = value
                result = OptimizationConfig._parse_env_bool("TEST_BOOL", False)
                self.assertEqual(result, expected)
                del os.environ["TEST_BOOL"]

    def test_empty_string(self):
        """空文字列処理のテスト"""
        os.environ["TEST_BOOL"] = ""
        result = OptimizationConfig._parse_env_bool("TEST_BOOL", True)
        self.assertTrue(result)  # デフォルト値を使用
        del os.environ["TEST_BOOL"]

        os.environ["TEST_BOOL"] = "   "  # 空白のみ
        result = OptimizationConfig._parse_env_bool("TEST_BOOL", False)
        self.assertFalse(result)  # デフォルト値を使用
        del os.environ["TEST_BOOL"]

    def test_undefined_variable(self):
        """未定義変数処理のテスト"""
        # 環境変数が存在しない場合
        result = OptimizationConfig._parse_env_bool("NONEXISTENT_VAR", True)
        self.assertTrue(result)

        result = OptimizationConfig._parse_env_bool("NONEXISTENT_VAR", False)
        self.assertFalse(result)

    def test_invalid_values(self):
        """不正値処理のテスト（デフォルト値に戻る）"""
        invalid_values = ["maybe", "perhaps", "2", "invalid", "tru", "fals"]

        for value in invalid_values:
            with self.subTest(value=value):
                os.environ["TEST_BOOL"] = value
                # デフォルト値がTrueの場合
                result = OptimizationConfig._parse_env_bool("TEST_BOOL", True)
                self.assertTrue(result, f"'{value}' should fallback to default True")

                # デフォルト値がFalseの場合
                result = OptimizationConfig._parse_env_bool("TEST_BOOL", False)
                self.assertFalse(result, f"'{value}' should fallback to default False")
                del os.environ["TEST_BOOL"]


class TestParseEnvInt(TestIssue642EnvironmentParsingRobustness):
    """_parse_env_int静的メソッドのテスト"""

    def test_valid_integers(self):
        """有効な整数値のテスト"""
        test_cases = [
            ("100", 100),
            ("0", 0),
            ("-50", -50),
            ("999999", 999999),
        ]

        for value, expected in test_cases:
            with self.subTest(value=value):
                os.environ["TEST_INT"] = value
                result = OptimizationConfig._parse_env_int("TEST_INT", 42)
                self.assertEqual(result, expected)
                del os.environ["TEST_INT"]

    def test_whitespace_handling(self):
        """空白文字処理のテスト"""
        test_cases = [
            ("  100  ", 100),
            ("\t200\t", 200),
            ("\n300\n", 300),
        ]

        for value, expected in test_cases:
            with self.subTest(value=repr(value)):
                os.environ["TEST_INT"] = value
                result = OptimizationConfig._parse_env_int("TEST_INT", 42)
                self.assertEqual(result, expected)
                del os.environ["TEST_INT"]

    def test_empty_string(self):
        """空文字列処理のテスト"""
        os.environ["TEST_INT"] = ""
        result = OptimizationConfig._parse_env_int("TEST_INT", 42)
        self.assertEqual(result, 42)  # デフォルト値を使用
        del os.environ["TEST_INT"]

        os.environ["TEST_INT"] = "   "  # 空白のみ
        result = OptimizationConfig._parse_env_int("TEST_INT", 123)
        self.assertEqual(result, 123)  # デフォルト値を使用
        del os.environ["TEST_INT"]

    def test_undefined_variable(self):
        """未定義変数処理のテスト"""
        result = OptimizationConfig._parse_env_int("NONEXISTENT_VAR", 99)
        self.assertEqual(result, 99)

    def test_invalid_values(self):
        """不正値処理のテスト（デフォルト値に戻る）"""
        invalid_values = ["not_a_number", "12.34", "1e5", "100px", "true", ""]

        for value in invalid_values:
            with self.subTest(value=value):
                os.environ["TEST_INT"] = value
                result = OptimizationConfig._parse_env_int("TEST_INT", 999)
                self.assertEqual(result, 999, f"'{value}' should fallback to default 999")
                del os.environ["TEST_INT"]


class TestFromEnvIntegration(TestIssue642EnvironmentParsingRobustness):
    """from_envメソッドの統合テスト"""

    def test_all_boolean_environments(self):
        """全てのブール環境変数の統合テスト"""
        # 全てTrueに設定
        env_vars = {
            "DAYTRADE_AUTO_FALLBACK": "yes",
            "DAYTRADE_PERF_MONITORING": "1",
            "DAYTRADE_CACHE_ENABLED": "enable",
            "DAYTRADE_PARALLEL": "true",
            "CI": "on",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        config = OptimizationConfig.from_env()

        self.assertTrue(config.auto_fallback)
        self.assertTrue(config.performance_monitoring)
        self.assertTrue(config.cache_enabled)
        self.assertTrue(config.parallel_processing)
        self.assertTrue(config.ci_test_mode)

    def test_all_integer_environments(self):
        """全ての整数環境変数の統合テスト"""
        env_vars = {
            "DAYTRADE_BATCH_SIZE": "200",
            "DAYTRADE_TIMEOUT": "60",
            "DAYTRADE_MEMORY_LIMIT": "1024",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        config = OptimizationConfig.from_env()

        self.assertEqual(config.batch_size, 200)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.memory_limit_mb, 1024)

    def test_mixed_valid_invalid_values(self):
        """有効・無効値混在テスト"""
        env_vars = {
            "DAYTRADE_AUTO_FALLBACK": "invalid_bool",  # 無効 -> デフォルトTrue
            "DAYTRADE_PERF_MONITORING": "false",       # 有効
            "DAYTRADE_BATCH_SIZE": "not_a_number",     # 無効 -> デフォルト100
            "DAYTRADE_TIMEOUT": "45",                  # 有効
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        config = OptimizationConfig.from_env()

        # 無効値はデフォルトに戻る
        self.assertTrue(config.auto_fallback)  # デフォルト値
        self.assertFalse(config.performance_monitoring)  # パース成功
        self.assertEqual(config.batch_size, 100)  # デフォルト値
        self.assertEqual(config.timeout_seconds, 45)  # パース成功

    def test_optimization_level_parsing(self):
        """最適化レベルパースのテスト（既存機能）"""
        valid_levels = ["standard", "optimized", "adaptive", "debug"]

        for level in valid_levels:
            with self.subTest(level=level):
                os.environ["DAYTRADE_OPTIMIZATION_LEVEL"] = level
                config = OptimizationConfig.from_env()
                self.assertEqual(config.level.value, level)
                del os.environ["DAYTRADE_OPTIMIZATION_LEVEL"]

    def test_no_environment_variables(self):
        """環境変数なしでのデフォルト動作テスト"""
        config = OptimizationConfig.from_env()

        # デフォルト値の確認
        self.assertEqual(config.level, OptimizationLevel.STANDARD)
        self.assertTrue(config.auto_fallback)
        self.assertTrue(config.performance_monitoring)
        self.assertTrue(config.cache_enabled)
        self.assertFalse(config.parallel_processing)
        self.assertEqual(config.batch_size, 100)
        self.assertEqual(config.timeout_seconds, 30)
        self.assertEqual(config.memory_limit_mb, 512)
        self.assertFalse(config.ci_test_mode)


class TestBackwardCompatibility(TestIssue642EnvironmentParsingRobustness):
    """後方互換性テスト"""

    def test_legacy_true_false_strings(self):
        """従来のtrue/false文字列での互換性確認"""
        # 従来の "true" / "false" がまだ動作することを確認
        os.environ["DAYTRADE_AUTO_FALLBACK"] = "true"
        os.environ["DAYTRADE_PARALLEL"] = "false"

        config = OptimizationConfig.from_env()

        self.assertTrue(config.auto_fallback)
        self.assertFalse(config.parallel_processing)

    def test_ci_environment_detection(self):
        """CI環境検出のテスト"""
        # CI=trueの場合
        os.environ["CI"] = "true"
        config = OptimizationConfig.from_env()
        self.assertTrue(config.ci_test_mode)
        del os.environ["CI"]

        # CI=1の場合
        os.environ["CI"] = "1"
        config = OptimizationConfig.from_env()
        self.assertTrue(config.ci_test_mode)
        del os.environ["CI"]

        # CIなしの場合
        config = OptimizationConfig.from_env()
        self.assertFalse(config.ci_test_mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)