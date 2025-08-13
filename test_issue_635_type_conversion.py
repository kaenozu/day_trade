"""
Issue #635: OptimizationConfig.from_fileでの型変換確実性改善テスト

OptimizationConfig.from_fileメソッドにおいて:
- batch_size, timeout_seconds, memory_limit_mbなどのint型フィールドの安全な変換
- bool型フィールドの文字列からの適切な変換
- 範囲チェックと妥当性検証
- エラーハンドリングとデフォルト値の使用
"""

import unittest
from unittest.mock import mock_open, patch
import json
import tempfile
import os

from src.day_trade.core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel
)


class TestSafeTypeConversions(unittest.TestCase):
    """安全な型変換のテスト"""

    def test_safe_str_conversion_normal_cases(self):
        """文字列変換の正常ケーステスト"""
        # 文字列の場合
        result = OptimizationConfig._safe_str_conversion("test_string")
        self.assertEqual(result, "test_string")

        # 数値の場合
        result = OptimizationConfig._safe_str_conversion(123)
        self.assertEqual(result, "123")

        # float の場合
        result = OptimizationConfig._safe_str_conversion(45.67)
        self.assertEqual(result, "45.67")

        # None の場合
        result = OptimizationConfig._safe_str_conversion(None, "default")
        self.assertEqual(result, "default")

    def test_safe_str_conversion_edge_cases(self):
        """文字列変換のエッジケーステスト"""
        # bool の場合
        result = OptimizationConfig._safe_str_conversion(True)
        self.assertEqual(result, "True")

        # 空文字列の場合
        result = OptimizationConfig._safe_str_conversion("")
        self.assertEqual(result, "")

        # リストの場合（文字列化される）
        result = OptimizationConfig._safe_str_conversion([1, 2, 3])
        self.assertEqual(result, "[1, 2, 3]")

    def test_safe_bool_conversion_normal_cases(self):
        """bool変換の正常ケーステスト"""
        # bool の場合
        self.assertTrue(OptimizationConfig._safe_bool_conversion(True))
        self.assertFalse(OptimizationConfig._safe_bool_conversion(False))

        # 文字列の場合（true系）
        true_strings = ["true", "True", "TRUE", "yes", "YES", "1", "on", "ON", "enabled", "ENABLED"]
        for true_str in true_strings:
            with self.subTest(value=true_str):
                result = OptimizationConfig._safe_bool_conversion(true_str)
                self.assertTrue(result)

        # 文字列の場合（false系）
        false_strings = ["false", "False", "FALSE", "no", "NO", "0", "off", "OFF", "disabled", "invalid"]
        for false_str in false_strings:
            with self.subTest(value=false_str):
                result = OptimizationConfig._safe_bool_conversion(false_str)
                self.assertFalse(result)

    def test_safe_bool_conversion_numeric_cases(self):
        """bool変換の数値ケーステスト"""
        # 数値の場合（0以外はTrue）
        self.assertFalse(OptimizationConfig._safe_bool_conversion(0))
        self.assertFalse(OptimizationConfig._safe_bool_conversion(0.0))
        self.assertTrue(OptimizationConfig._safe_bool_conversion(1))
        self.assertTrue(OptimizationConfig._safe_bool_conversion(-1))
        self.assertTrue(OptimizationConfig._safe_bool_conversion(42))
        self.assertTrue(OptimizationConfig._safe_bool_conversion(3.14))

    def test_safe_bool_conversion_edge_cases(self):
        """bool変換のエッジケーステスト"""
        # None の場合
        result = OptimizationConfig._safe_bool_conversion(None, True)
        self.assertTrue(result)

        result = OptimizationConfig._safe_bool_conversion(None, False)
        self.assertFalse(result)

        # 空文字列の場合
        result = OptimizationConfig._safe_bool_conversion("")
        self.assertFalse(result)

    def test_safe_int_conversion_normal_cases(self):
        """int変換の正常ケーステスト"""
        # int の場合
        result = OptimizationConfig._safe_int_conversion(42, 0)
        self.assertEqual(result, 42)

        # float の場合（整数）
        result = OptimizationConfig._safe_int_conversion(100.0, 0)
        self.assertEqual(result, 100)

        # float の場合（小数点あり）
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            result = OptimizationConfig._safe_int_conversion(123.456, 0)
            self.assertEqual(result, 123)
            mock_logger.warning.assert_called()

        # 文字列の場合
        result = OptimizationConfig._safe_int_conversion("789", 0)
        self.assertEqual(result, 789)

        # 文字列（float形式）の場合
        result = OptimizationConfig._safe_int_conversion("456.0", 0)
        self.assertEqual(result, 456)

    def test_safe_int_conversion_range_validation(self):
        """int変換の範囲検証テスト"""
        # 最小値チェック
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            result = OptimizationConfig._safe_int_conversion(50, 100, min_val=60, max_val=200)
            self.assertEqual(result, 60)  # 最小値に調整
            mock_logger.warning.assert_called()

        # 最大値チェック
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            result = OptimizationConfig._safe_int_conversion(250, 100, min_val=60, max_val=200)
            self.assertEqual(result, 200)  # 最大値に調整
            mock_logger.warning.assert_called()

        # 範囲内の値
        result = OptimizationConfig._safe_int_conversion(150, 100, min_val=60, max_val=200)
        self.assertEqual(result, 150)  # そのまま

    def test_safe_int_conversion_error_cases(self):
        """int変換のエラーケーステスト"""
        # None の場合
        result = OptimizationConfig._safe_int_conversion(None, 42)
        self.assertEqual(result, 42)

        # 無効な文字列の場合
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            result = OptimizationConfig._safe_int_conversion("invalid", 42)
            self.assertEqual(result, 42)
            mock_logger.warning.assert_called()

        # 変換不可能な型の場合
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            result = OptimizationConfig._safe_int_conversion([], 42)
            self.assertEqual(result, 42)
            mock_logger.warning.assert_called()


class TestOptimizationConfigFromFile(unittest.TestCase):
    """OptimizationConfig.from_fileのテスト"""

    def test_from_file_normal_json(self):
        """正常なJSONファイルからの読み込みテスト"""
        config_data = {
            "level": "optimized",
            "auto_fallback": True,
            "performance_monitoring": False,
            "cache_enabled": "true",  # 文字列形式
            "parallel_processing": "false",  # 文字列形式
            "batch_size": 200,
            "timeout_seconds": "60",  # 文字列形式
            "memory_limit_mb": 1024.0  # float形式
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = OptimizationConfig.from_file(temp_path)

            # 値の確認
            self.assertEqual(config.level, OptimizationLevel.OPTIMIZED)
            self.assertTrue(config.auto_fallback)
            self.assertFalse(config.performance_monitoring)
            self.assertTrue(config.cache_enabled)  # "true" -> True
            self.assertFalse(config.parallel_processing)  # "false" -> False
            self.assertEqual(config.batch_size, 200)
            self.assertEqual(config.timeout_seconds, 60)  # "60" -> 60
            self.assertEqual(config.memory_limit_mb, 1024)  # 1024.0 -> 1024

            # 型の確認
            self.assertIsInstance(config.batch_size, int)
            self.assertIsInstance(config.timeout_seconds, int)
            self.assertIsInstance(config.memory_limit_mb, int)

        finally:
            os.unlink(temp_path)

    def test_from_file_missing_fields(self):
        """一部フィールドが欠けているJSONファイルのテスト"""
        config_data = {
            "level": "adaptive",
            "batch_size": "500"  # 文字列形式のint
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = OptimizationConfig.from_file(temp_path)

            # 指定されたフィールド
            self.assertEqual(config.level, OptimizationLevel.ADAPTIVE)
            self.assertEqual(config.batch_size, 500)

            # デフォルト値が使用されているフィールド
            self.assertTrue(config.auto_fallback)  # デフォルト: True
            self.assertTrue(config.performance_monitoring)  # デフォルト: True
            self.assertEqual(config.timeout_seconds, 30)  # デフォルト: 30

        finally:
            os.unlink(temp_path)

    def test_from_file_invalid_types(self):
        """無効な型を含むJSONファイルのテスト"""
        config_data = {
            "level": "standard",
            "batch_size": "invalid_number",  # 無効な文字列
            "timeout_seconds": [],  # 配列型（無効）
            "memory_limit_mb": {"invalid": "object"},  # オブジェクト型（無効）
            "auto_fallback": "maybe",  # 曖昧なbool文字列
            "cache_enabled": 42  # 数値形式のbool
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # エラーログが出力されることを確認
                self.assertTrue(mock_logger.warning.called)

                # デフォルト値が使用されることを確認
                self.assertEqual(config.batch_size, 100)  # デフォルト
                self.assertEqual(config.timeout_seconds, 30)  # デフォルト
                self.assertEqual(config.memory_limit_mb, 512)  # デフォルト
                self.assertFalse(config.auto_fallback)  # "maybe" -> False
                self.assertTrue(config.cache_enabled)  # 42 -> True

        finally:
            os.unlink(temp_path)

    def test_from_file_range_violations(self):
        """範囲制限違反のテスト"""
        config_data = {
            "level": "standard",
            "batch_size": 50000,  # 最大値超過
            "timeout_seconds": 0,  # 最小値未満
            "memory_limit_mb": 32  # 最小値未満
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # 警告ログが出力されることを確認
                self.assertTrue(mock_logger.warning.called)

                # 範囲内に調整されることを確認
                self.assertEqual(config.batch_size, 10000)  # max_val
                self.assertEqual(config.timeout_seconds, 1)  # min_val
                self.assertEqual(config.memory_limit_mb, 64)  # min_val

        finally:
            os.unlink(temp_path)

    def test_from_file_nonexistent_file(self):
        """存在しないファイルのテスト"""
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            config = OptimizationConfig.from_file("/nonexistent/path/config.json")

            # エラーログが出力されることを確認
            mock_logger.error.assert_called()

            # デフォルト設定が返されることを確認
            self.assertEqual(config.level, OptimizationLevel.STANDARD)
            self.assertEqual(config.batch_size, 100)
            self.assertEqual(config.timeout_seconds, 30)

    def test_from_file_invalid_json(self):
        """無効なJSONファイルのテスト"""
        invalid_json = "{ invalid json content"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_json)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # エラーログが出力されることを確認
                mock_logger.error.assert_called()

                # デフォルト設定が返されることを確認
                self.assertEqual(config.level, OptimizationLevel.STANDARD)

        finally:
            os.unlink(temp_path)

    def test_from_file_invalid_optimization_level(self):
        """無効な最適化レベルのテスト"""
        config_data = {
            "level": "invalid_level",
            "batch_size": 200
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # エラーログが出力されることを確認
                mock_logger.error.assert_called()

                # デフォルト設定が返されることを確認
                self.assertEqual(config.level, OptimizationLevel.STANDARD)

        finally:
            os.unlink(temp_path)


class TestComplexTypeConversionScenarios(unittest.TestCase):
    """複雑な型変換シナリオのテスト"""

    def test_mixed_type_configuration(self):
        """様々な型が混在する設定のテスト"""
        config_data = {
            "level": 123,  # 数値（文字列化される）
            "auto_fallback": 1,  # 数値形式のbool
            "performance_monitoring": "enabled",  # 文字列形式のbool
            "cache_enabled": 0,  # 数値形式のbool (False)
            "parallel_processing": "",  # 空文字列形式のbool (False)
            "batch_size": "150.75",  # 小数点付き文字列
            "timeout_seconds": True,  # bool（数値化される）
            "memory_limit_mb": [1024]  # 配列（エラー->デフォルト）
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # 変換結果の確認
                self.assertTrue(config.auto_fallback)  # 1 -> True
                self.assertTrue(config.performance_monitoring)  # "enabled" -> True
                self.assertFalse(config.cache_enabled)  # 0 -> False
                self.assertFalse(config.parallel_processing)  # "" -> False
                self.assertEqual(config.batch_size, 150)  # "150.75" -> 150
                self.assertEqual(config.memory_limit_mb, 512)  # [1024] -> default

                # 警告ログの出力を確認
                self.assertTrue(mock_logger.warning.called)

        finally:
            os.unlink(temp_path)

    def test_boundary_values_in_ranges(self):
        """範囲境界値のテスト"""
        config_data = {
            "batch_size": 1,  # 最小値
            "timeout_seconds": 3600,  # 最大値
            "memory_limit_mb": 64  # 最小値
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = OptimizationConfig.from_file(temp_path)

            # 境界値が正しく処理されることを確認
            self.assertEqual(config.batch_size, 1)
            self.assertEqual(config.timeout_seconds, 3600)
            self.assertEqual(config.memory_limit_mb, 64)

        finally:
            os.unlink(temp_path)

    def test_extreme_values_handling(self):
        """極端な値の処理テスト"""
        config_data = {
            "batch_size": -100,  # 負の値
            "timeout_seconds": 999999,  # 極端に大きい値
            "memory_limit_mb": 0.5  # 小数値
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                config = OptimizationConfig.from_file(temp_path)

                # 範囲内に調整されることを確認
                self.assertEqual(config.batch_size, 1)  # min_val
                self.assertEqual(config.timeout_seconds, 3600)  # max_val
                self.assertEqual(config.memory_limit_mb, 64)  # min_val (0.5 -> 0 -> 64)

                # 警告ログが出力されることを確認
                self.assertTrue(mock_logger.warning.called)

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)