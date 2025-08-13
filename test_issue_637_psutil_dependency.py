"""
Issue #637: psutil依存性の解決テスト

OptimizationStrategyFactory._select_adaptive_levelメソッドにおいて:
- psutilが利用できない環境での適切なフォールバック処理
- システムメトリクス取得の堅牢性改善
- エラーハンドリングとログ出力の最適化
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# テスト対象のインポート
from src.day_trade.core.optimization_strategy import (
    OptimizationStrategyFactory,
    OptimizationConfig,
    OptimizationLevel
)


class TestPsutilDependencyHandling(unittest.TestCase):
    """psutil依存性処理のテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.factory = OptimizationStrategyFactory
        self.standard_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            memory_limit_mb=512,
            ci_test_mode=False
        )

    def test_get_system_metrics_with_psutil_available(self):
        """psutilが利用可能な場合のシステムメトリクス取得テスト"""
        # psutilのモック
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.cpu_percent.return_value = 30.0

        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            memory_percent, cpu_percent = self.factory._get_system_metrics()

            self.assertIsNotNone(memory_percent)
            self.assertIsNotNone(cpu_percent)
            self.assertEqual(memory_percent, 50.0)
            self.assertEqual(cpu_percent, 30.0)

            # psutil関数が正しく呼ばれることを確認
            mock_psutil.virtual_memory.assert_called_once()
            mock_psutil.cpu_percent.assert_called_once_with(interval=0.1)

    def test_get_system_metrics_with_psutil_unavailable(self):
        """psutilが利用できない場合のシステムメトリクス取得テスト"""
        # psutilのインポートエラーをシミュレート
        with patch('builtins.__import__', side_effect=ImportError("No module named 'psutil'")):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                memory_percent, cpu_percent = self.factory._get_system_metrics()

                self.assertIsNone(memory_percent)
                self.assertIsNone(cpu_percent)

                # 適切な警告ログが出力されることを確認
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args[0][0]
                self.assertIn("psutilが利用できません", call_args)

    def test_get_system_metrics_with_psutil_exception(self):
        """psutilでの例外発生時のシステムメトリクス取得テスト"""
        # psutilは利用可能だが、実行時に例外が発生
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = Exception("システムエラー")

        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                memory_percent, cpu_percent = self.factory._get_system_metrics()

                self.assertIsNone(memory_percent)
                self.assertIsNone(cpu_percent)

                # 適切な警告ログが出力されることを確認
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args[0][0]
                self.assertIn("システムメトリクス取得エラー", call_args)

    def test_select_level_by_metrics_high_load(self):
        """高負荷時のレベル選択テスト"""
        # 高負荷状況（CPU 90%, Memory 85%）
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_by_metrics(85.0, 90.0)

            self.assertEqual(level, OptimizationLevel.STANDARD)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("高負荷検出", call_args)
            self.assertIn("CPU=90.0%", call_args)
            self.assertIn("MEM=85.0%", call_args)

    def test_select_level_by_metrics_medium_load(self):
        """中負荷時のレベル選択テスト"""
        # 中負荷状況（CPU 70%, Memory 65%）
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_by_metrics(65.0, 70.0)

            self.assertEqual(level, OptimizationLevel.OPTIMIZED)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("中負荷検出", call_args)

    def test_select_level_by_metrics_low_load(self):
        """低負荷時のレベル選択テスト"""
        # 低負荷状況（CPU 30%, Memory 40%）
        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_by_metrics(40.0, 30.0)

            self.assertEqual(level, OptimizationLevel.OPTIMIZED)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("低負荷検出", call_args)

    def test_select_level_fallback_ci_mode(self):
        """CI環境でのフォールバック選択テスト"""
        ci_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            ci_test_mode=True
        )

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_fallback(ci_config)

            self.assertEqual(level, OptimizationLevel.STANDARD)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("CI環境検出", call_args)

    def test_select_level_fallback_low_memory(self):
        """低メモリ制限でのフォールバック選択テスト"""
        low_memory_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            memory_limit_mb=128,  # 256MB未満
            ci_test_mode=False
        )

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_fallback(low_memory_config)

            self.assertEqual(level, OptimizationLevel.STANDARD)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("メモリ制限検出", call_args)
            self.assertIn("128MB", call_args)

    def test_select_level_fallback_default(self):
        """デフォルトフォールバック選択テスト"""
        normal_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            memory_limit_mb=512,
            ci_test_mode=False
        )

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            level = self.factory._select_level_fallback(normal_config)

            self.assertEqual(level, OptimizationLevel.OPTIMIZED)

            # 適切なログが出力されることを確認
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("システム監視なし", call_args)
            self.assertIn("最適化実装選択", call_args)


class TestAdaptiveLevelSelectionIntegration(unittest.TestCase):
    """適応的レベル選択の統合テスト"""

    def setUp(self):
        """テスト前の準備"""
        self.factory = OptimizationStrategyFactory
        self.test_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            memory_limit_mb=512,
            ci_test_mode=False
        )

    def test_select_adaptive_level_with_psutil_success(self):
        """psutil成功時の適応的レベル選択テスト"""
        # psutilが正常に動作する場合
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.cpu_percent.return_value = 40.0

        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                level = self.factory._select_adaptive_level("test_component", self.test_config)

                self.assertEqual(level, OptimizationLevel.OPTIMIZED)

                # 低負荷ログが出力されることを確認
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                self.assertTrue(any("低負荷検出" in call for call in info_calls))

    def test_select_adaptive_level_with_psutil_failure(self):
        """psutil失敗時の適応的レベル選択テスト"""
        # psutilが利用できない場合
        with patch('builtins.__import__', side_effect=ImportError("psutil not found")):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                level = self.factory._select_adaptive_level("test_component", self.test_config)

                self.assertEqual(level, OptimizationLevel.OPTIMIZED)

                # フォールバック処理ログが出力されることを確認
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

                self.assertTrue(any("psutilが利用できません" in call for call in warning_calls))
                self.assertTrue(any("システム監視なし" in call for call in info_calls))

    def test_select_adaptive_level_with_exception(self):
        """例外発生時の適応的レベル選択テスト"""
        # システムメトリクス取得で例外が発生
        with patch.object(self.factory, '_get_system_metrics', side_effect=Exception("システムエラー")):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                level = self.factory._select_adaptive_level("test_component", self.test_config)

                self.assertEqual(level, OptimizationLevel.OPTIMIZED)

                # エラーログが出力されることを確認
                error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                self.assertTrue(any("適応的レベル選択エラー" in call for call in error_calls))

    def test_select_adaptive_level_ci_environment(self):
        """CI環境での適応的レベル選択テスト"""
        ci_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            ci_test_mode=True
        )

        # ロギングシステムの干渉を避けるため、より具体的なパッチを使用
        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("psutil not found")
            return __import__(name, *args, **kwargs)

        with patch('src.day_trade.core.optimization_strategy.logger'):
            with patch('builtins.__import__', side_effect=mock_import):
                level = self.factory._select_adaptive_level("test_component", ci_config)

                self.assertEqual(level, OptimizationLevel.STANDARD)


class TestSystemMetricsEdgeCases(unittest.TestCase):
    """システムメトリクスのエッジケーステスト"""

    def setUp(self):
        """テスト前の準備"""
        self.factory = OptimizationStrategyFactory

    def test_metrics_boundary_values(self):
        """境界値でのメトリクステスト"""
        test_cases = [
            # (memory%, cpu%, expected_level)
            (80.0, 79.9, OptimizationLevel.OPTIMIZED),  # メモリが境界値
            (80.1, 60.0, OptimizationLevel.STANDARD),   # メモリが境界値超え
            (60.0, 80.0, OptimizationLevel.STANDARD),   # CPUが境界値
            (60.1, 60.0, OptimizationLevel.OPTIMIZED),  # 中負荷境界値
            (60.0, 60.1, OptimizationLevel.OPTIMIZED),  # 中負荷境界値
        ]

        for memory_percent, cpu_percent, expected_level in test_cases:
            with self.subTest(memory=memory_percent, cpu=cpu_percent):
                level = self.factory._select_level_by_metrics(memory_percent, cpu_percent)
                self.assertEqual(level, expected_level)

    def test_extreme_metric_values(self):
        """極端な値でのメトリクステスト"""
        # 極端に高い値
        level = self.factory._select_level_by_metrics(100.0, 100.0)
        self.assertEqual(level, OptimizationLevel.STANDARD)

        # 極端に低い値
        level = self.factory._select_level_by_metrics(0.0, 0.0)
        self.assertEqual(level, OptimizationLevel.OPTIMIZED)

        # 負の値（異常ケース）
        level = self.factory._select_level_by_metrics(-10.0, -5.0)
        self.assertEqual(level, OptimizationLevel.OPTIMIZED)

    def test_psutil_import_timing(self):
        """psutilインポートタイミングテスト"""
        # 最初の呼び出しでImportError、2回目で成功
        import_count = 0
        original_import = __builtins__['__import__']

        def mock_import(name, *args, **kwargs):
            nonlocal import_count
            if name == 'psutil':
                import_count += 1
                if import_count == 1:
                    raise ImportError("First import fails")
                # 2回目以降は正常なpsutilモックを返す
                mock_psutil = MagicMock()
                mock_psutil.virtual_memory.return_value.percent = 50.0
                mock_psutil.cpu_percent.return_value = 30.0
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # 最初の呼び出し（失敗）
            memory1, cpu1 = self.factory._get_system_metrics()
            self.assertIsNone(memory1)
            self.assertIsNone(cpu1)

            # 2回目の呼び出し（成功）
            memory2, cpu2 = self.factory._get_system_metrics()
            self.assertEqual(memory2, 50.0)
            self.assertEqual(cpu2, 30.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)