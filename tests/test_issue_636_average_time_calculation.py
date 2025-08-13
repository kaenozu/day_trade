"""
Issue #636: 平均時間計算の堅牢性改善テスト

OptimizationStrategy.record_executionメソッドにおいて:
- ZeroDivisionErrorの防止
- 無効な実行時間値の検証
- 指数移動平均による外れ値対策
- エラーハンドリングとフォールバック処理
"""

import unittest
from unittest.mock import Mock, patch
import math

from src.day_trade.core.optimization_strategy import (
    OptimizationStrategy,
    OptimizationConfig,
    OptimizationLevel
)


class TestStrategy(OptimizationStrategy):
    """テスト用のOptimizationStrategy実装"""

    def execute(self, *args, **kwargs):
        return "test_result"

    def get_strategy_name(self) -> str:
        return "TestStrategy"


class TestExecutionTimeValidation(unittest.TestCase):
    """実行時間検証のテスト"""

    def setUp(self):
        """テスト前の準備"""
        config = OptimizationConfig()
        self.strategy = TestStrategy(config)

    def test_validate_execution_time_valid_values(self):
        """正常な実行時間値の検証テスト"""
        valid_times = [0.0, 0.1, 1.0, 60.0, 3600.0]

        for time_value in valid_times:
            with self.subTest(time=time_value):
                result = self.strategy._validate_execution_time(time_value)
                self.assertTrue(result)

    def test_validate_execution_time_invalid_types(self):
        """無効な型の実行時間検証テスト"""
        invalid_types = ["string", None, [], {}, complex(1, 2)]

        for invalid_value in invalid_types:
            with self.subTest(value=invalid_value):
                result = self.strategy._validate_execution_time(invalid_value)
                self.assertFalse(result)

    def test_validate_execution_time_negative_values(self):
        """負の実行時間検証テスト"""
        negative_values = [-1.0, -0.1, -100.0]

        for negative_time in negative_values:
            with self.subTest(time=negative_time):
                result = self.strategy._validate_execution_time(negative_time)
                self.assertFalse(result)

    def test_validate_execution_time_extreme_values(self):
        """極端な実行時間値の検証テスト"""
        # 1時間を超える値は無効
        result = self.strategy._validate_execution_time(3601.0)
        self.assertFalse(result)

        # 非常に大きな値
        result = self.strategy._validate_execution_time(999999.0)
        self.assertFalse(result)

    def test_validate_execution_time_special_float_values(self):
        """特殊なfloat値の検証テスト"""
        # NaN
        result = self.strategy._validate_execution_time(float('nan'))
        self.assertFalse(result)

        # 正の無限大
        result = self.strategy._validate_execution_time(float('inf'))
        self.assertFalse(result)

        # 負の無限大
        result = self.strategy._validate_execution_time(float('-inf'))
        self.assertFalse(result)

    def test_is_finite_positive(self):
        """有限正数チェックのテスト"""
        # 正常な値
        self.assertTrue(self.strategy._is_finite_positive(1.0))
        self.assertTrue(self.strategy._is_finite_positive(0.0))

        # 無効な値
        self.assertFalse(self.strategy._is_finite_positive(-1.0))
        self.assertFalse(self.strategy._is_finite_positive(float('inf')))
        self.assertFalse(self.strategy._is_finite_positive(float('nan')))


class TestAverageTimeCalculation(unittest.TestCase):
    """平均時間計算のテスト"""

    def setUp(self):
        """テスト前の準備"""
        config = OptimizationConfig()
        self.strategy = TestStrategy(config)

    def test_first_execution_time_recording(self):
        """初回実行時間記録のテスト"""
        execution_time = 2.5

        self.strategy.record_execution(execution_time, True)

        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics["execution_count"], 1)
        self.assertEqual(metrics["total_time"], execution_time)
        self.assertEqual(metrics["average_time"], execution_time)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["error_count"], 0)

    def test_multiple_execution_times_recording(self):
        """複数回の実行時間記録テスト"""
        execution_times = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, time_value in enumerate(execution_times):
            self.strategy.record_execution(time_value, True)

            metrics = self.strategy.get_performance_metrics()
            self.assertEqual(metrics["execution_count"], i + 1)
            self.assertEqual(metrics["total_time"], sum(execution_times[:i+1]))

            # 指数移動平均が正しく計算されているかチェック
            self.assertGreater(metrics["average_time"], 0)
            self.assertLessEqual(metrics["average_time"], max(execution_times[:i+1]))

    def test_exponential_moving_average_calculation(self):
        """指数移動平均計算のテスト"""
        # 初回実行
        self.strategy.record_execution(10.0, True)
        first_avg = self.strategy.performance_metrics["average_time"]
        self.assertEqual(first_avg, 10.0)

        # 2回目実行（大きく異なる値）
        self.strategy.record_execution(2.0, True)
        second_avg = self.strategy.performance_metrics["average_time"]

        # 指数移動平均により、新しい値の影響が制限される
        self.assertGreater(second_avg, 2.0)  # 新しい値より大きい
        self.assertLess(second_avg, 10.0)   # 古い値より小さい

    def test_smoothing_factor_calculation(self):
        """平滑化係数計算のテスト"""
        # 実行回数が少ない場合（高い係数）
        alpha_early = self.strategy._calculate_smoothing_factor(3)
        self.assertEqual(alpha_early, 0.3)

        # 中程度の実行回数
        alpha_medium = self.strategy._calculate_smoothing_factor(15)
        self.assertEqual(alpha_medium, 0.15)

        # 多い実行回数（低い係数）
        alpha_stable = self.strategy._calculate_smoothing_factor(50)
        self.assertEqual(alpha_stable, 0.05)

        # 係数が減少していることを確認
        self.assertGreater(alpha_early, alpha_medium)
        self.assertGreater(alpha_medium, alpha_stable)

    def test_zero_execution_count_handling(self):
        """実行回数0の場合の処理テスト"""
        # 実行回数を人為的に0に設定
        self.strategy.performance_metrics["execution_count"] = 0

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            self.strategy._update_average_time_safely(5.0)

            # 警告ログが出力され、平均時間が0に設定される
            mock_logger.warning.assert_called_once()
            self.assertEqual(self.strategy.performance_metrics["average_time"], 0.0)

    def test_fallback_average_calculation(self):
        """フォールバック平均計算のテスト"""
        # 通常の状態を設定
        self.strategy.performance_metrics = {
            "execution_count": 3,
            "total_time": 15.0,
            "success_count": 2,
            "error_count": 1,
            "average_time": 0.0
        }

        self.strategy._fallback_average_calculation()

        # 単純平均が計算される
        expected_average = 15.0 / 3
        self.assertEqual(self.strategy.performance_metrics["average_time"], expected_average)

    def test_fallback_with_zero_execution_count(self):
        """実行回数0でのフォールバック処理テスト"""
        self.strategy.performance_metrics = {
            "execution_count": 0,
            "total_time": 0.0,
            "success_count": 0,
            "error_count": 0,
            "average_time": 0.0
        }

        self.strategy._fallback_average_calculation()

        # ゼロ除算を避けて0に設定
        self.assertEqual(self.strategy.performance_metrics["average_time"], 0.0)


class TestInvalidInputHandling(unittest.TestCase):
    """無効入力処理のテスト"""

    def setUp(self):
        """テスト前の準備"""
        config = OptimizationConfig()
        self.strategy = TestStrategy(config)

    def test_record_execution_with_invalid_time(self):
        """無効な実行時間での記録テスト"""
        invalid_times = [-1.0, float('nan'), float('inf'), "invalid", None]

        for invalid_time in invalid_times:
            with self.subTest(time=invalid_time):
                with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                    initial_count = self.strategy.performance_metrics["execution_count"]

                    self.strategy.record_execution(invalid_time, True)

                    # 記録がスキップされることを確認
                    final_count = self.strategy.performance_metrics["execution_count"]
                    self.assertEqual(initial_count, final_count)

                    # 警告ログが出力されることを確認
                    mock_logger.warning.assert_called()

    def test_record_execution_success_and_failure(self):
        """成功・失敗記録のテスト"""
        # 成功記録
        self.strategy.record_execution(1.0, True)
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["error_count"], 0)

        # 失敗記録
        self.strategy.record_execution(2.0, False)
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["error_count"], 1)

        # 成功率と失敗率の計算
        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["error_rate"], 0.5)

    def test_edge_case_very_small_times(self):
        """非常に小さい実行時間のテスト"""
        very_small_times = [0.0, 0.001, 0.0001, 1e-6]

        for small_time in very_small_times:
            with self.subTest(time=small_time):
                initial_count = self.strategy.performance_metrics["execution_count"]
                self.strategy.record_execution(small_time, True)

                # 正常に記録されることを確認
                final_count = self.strategy.performance_metrics["execution_count"]
                self.assertEqual(final_count, initial_count + 1)

    def test_boundary_execution_time_values(self):
        """境界値の実行時間テスト"""
        # 境界値: ちょうど1時間
        self.strategy.record_execution(3600.0, True)
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics["execution_count"], 1)

        # 境界値を超える: 1時間+1秒
        with patch('src.day_trade.core.optimization_strategy.logger'):
            initial_count = metrics["execution_count"]
            self.strategy.record_execution(3601.0, True)

            # 記録がスキップされることを確認
            final_metrics = self.strategy.get_performance_metrics()
            self.assertEqual(final_metrics["execution_count"], initial_count)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """エラーハンドリングと復旧のテスト"""

    def setUp(self):
        """テスト前の準備"""
        config = OptimizationConfig()
        self.strategy = TestStrategy(config)

    def test_exception_in_average_time_update(self):
        """平均時間更新での例外処理テスト"""
        # 正常な実行を1回記録
        self.strategy.record_execution(5.0, True)

        # _update_average_time_safelyメソッドで意図的に例外を発生させる
        with patch.object(self.strategy, '_calculate_smoothing_factor', side_effect=Exception("Test exception")):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                self.strategy.record_execution(3.0, True)

                # エラーログが出力されることを確認
                mock_logger.error.assert_called()

                # メトリクスが正常に更新されていることを確認（フォールバック処理）
                metrics = self.strategy.get_performance_metrics()
                self.assertEqual(metrics["execution_count"], 2)  # 2回目の実行

    def test_exception_in_fallback_calculation(self):
        """フォールバック計算での例外処理テスト"""
        # メトリクスを意図的に破損
        self.strategy.performance_metrics["execution_count"] = None
        self.strategy.performance_metrics["total_time"] = "invalid"

        with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
            self.strategy._fallback_average_calculation()

            # エラーログが出力され、安全な値が設定される
            mock_logger.error.assert_called()
            self.assertEqual(self.strategy.performance_metrics["average_time"], 0.0)

    def test_validation_with_exception(self):
        """検証メソッドでの例外処理テスト"""
        # 検証メソッドで例外が発生する状況をモック
        with patch.object(self.strategy, '_is_finite_positive', side_effect=Exception("Test error")):
            with patch('src.day_trade.core.optimization_strategy.logger') as mock_logger:
                result = self.strategy._validate_execution_time(5.0)

                # 例外が発生した場合はFalseを返す
                self.assertFalse(result)

                # エラーログが出力される
                mock_logger.error.assert_called()


class TestMetricsResetAndRetrieval(unittest.TestCase):
    """メトリクスリセットと取得のテスト"""

    def setUp(self):
        """テスト前の準備"""
        config = OptimizationConfig()
        self.strategy = TestStrategy(config)

    def test_metrics_reset(self):
        """メトリクスリセットのテスト"""
        # いくつかの実行を記録
        self.strategy.record_execution(1.0, True)
        self.strategy.record_execution(2.0, False)
        self.strategy.record_execution(3.0, True)

        # リセット前の状態確認
        metrics_before = self.strategy.get_performance_metrics()
        self.assertGreater(metrics_before["execution_count"], 0)
        self.assertGreater(metrics_before["total_time"], 0)

        # リセット実行
        self.strategy.reset_metrics()

        # リセット後の状態確認
        metrics_after = self.strategy.get_performance_metrics()
        self.assertEqual(metrics_after["execution_count"], 0)
        self.assertEqual(metrics_after["total_time"], 0.0)
        self.assertEqual(metrics_after["average_time"], 0.0)
        self.assertEqual(metrics_after["success_count"], 0)
        self.assertEqual(metrics_after["error_count"], 0)

    def test_metrics_retrieval_with_rates(self):
        """率を含むメトリクス取得のテスト"""
        # 成功2回、失敗1回を記録
        self.strategy.record_execution(1.0, True)
        self.strategy.record_execution(2.0, True)
        self.strategy.record_execution(3.0, False)

        metrics = self.strategy.get_performance_metrics()

        # 基本メトリクス
        self.assertEqual(metrics["execution_count"], 3)
        self.assertEqual(metrics["success_count"], 2)
        self.assertEqual(metrics["error_count"], 1)

        # 計算された率
        self.assertAlmostEqual(metrics["success_rate"], 2/3, places=2)
        self.assertAlmostEqual(metrics["error_rate"], 1/3, places=2)

    def test_metrics_retrieval_zero_executions(self):
        """実行回数0でのメトリクス取得テスト"""
        metrics = self.strategy.get_performance_metrics()

        # 基本メトリクス
        self.assertEqual(metrics["execution_count"], 0)
        self.assertEqual(metrics["success_count"], 0)
        self.assertEqual(metrics["error_count"], 0)
        self.assertEqual(metrics["total_time"], 0.0)
        self.assertEqual(metrics["average_time"], 0.0)

        # 率は計算されない（ゼロ除算防止）
        self.assertNotIn("success_rate", metrics)
        self.assertNotIn("error_rate", metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)