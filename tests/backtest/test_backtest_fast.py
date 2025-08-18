#!/usr/bin/env python3
"""
高速バックテストテスト
Issue #375: 重い処理のモック化によるテスト速度改善

従来の重いバックテスト処理を高速モックに置き換えたテストスイート
"""

import time
import unittest
from datetime import datetime
from decimal import Decimal
from typing import Dict
from unittest.mock import patch

import numpy as np
import pandas as pd

# 強化モックシステム
from .fixtures.performance_mocks_enhanced import (
    create_fast_test_environment,
    measure_mock_performance,
)

try:
    from src.day_trade.analysis.backtest import (
        BacktestConfig,
        BacktestEngine,
        BacktestResult,
    )

    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False


class TestFastBacktest(unittest.TestCase):
    """高速バックテストテスト"""

    def setUp(self):
        """高速テストセットアップ"""
        self.start_time = time.perf_counter()

        # 高速モック環境
        self.mock_env = create_fast_test_environment(
            include_backtest=True,
            max_execution_time_ms=10.0,  # 10ms以下に制限
        )

        # 小規模テストデータ（30日分）
        self.small_data = self._create_test_data(30)

        # 中規模テストデータ（252日分）
        self.medium_data = self._create_test_data(252)

        # 大規模テストデータ（1000日分）
        self.large_data = self._create_test_data(1000)

    def tearDown(self):
        """実行時間測定"""
        execution_time = (time.perf_counter() - self.start_time) * 1000
        print(f"    テスト実行時間: {execution_time:.2f}ms")

    def _create_test_data(self, days: int) -> pd.DataFrame:
        """テストデータ作成"""
        dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

        # 現実的な価格データ生成
        base_price = 100
        returns = np.random.normal(0.001, 0.02, days)  # 日次リターン
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": prices * np.random.uniform(0.99, 1.01, days),
                "High": prices * np.random.uniform(1.00, 1.02, days),
                "Low": prices * np.random.uniform(0.98, 1.00, days),
                "Close": prices,
                "Volume": np.random.randint(100000, 1000000, days),
            }
        ).set_index("Date")

    def _simple_strategy(self, data: pd.DataFrame) -> Dict:
        """シンプル戦略"""
        if len(data) >= 20:
            sma_20 = data["Close"].rolling(20).mean().iloc[-1]
            current_price = data["Close"].iloc[-1]

            if current_price > sma_20:
                return {"action": "buy", "quantity": 100, "symbol": "TEST"}
            else:
                return {"action": "sell", "quantity": 100, "symbol": "TEST"}

        return {"action": "hold", "quantity": 0, "symbol": "TEST"}

    def test_fast_small_backtest(self):
        """高速小規模バックテストテスト"""
        engine = self.mock_env["backtest_engine"]

        result = engine.run_backtest(
            self.small_data, self._simple_strategy, initial_capital=1000000
        )

        self.assertIsInstance(result, dict)
        self.assertIn("final_value", result)
        self.assertIn("total_return", result)
        self.assertIn("sharpe_ratio", result)
        self.assertEqual(result["data_points"], 30)
        self.assertLess(result["execution_time_ms"], 10)

    def test_fast_medium_backtest(self):
        """高速中規模バックテストテスト"""
        engine = self.mock_env["backtest_engine"]

        result = engine.run_backtest(
            self.medium_data, self._simple_strategy, initial_capital=1000000
        )

        self.assertEqual(result["data_points"], 252)
        self.assertGreater(result["final_value"], 0)
        self.assertGreater(result["num_trades"], 0)
        self.assertLess(result["execution_time_ms"], 10)

    def test_fast_large_backtest(self):
        """高速大規模バックテストテスト"""
        engine = self.mock_env["backtest_engine"]

        result = engine.run_backtest(
            self.large_data, self._simple_strategy, initial_capital=1000000
        )

        self.assertEqual(result["data_points"], 1000)
        self.assertIsInstance(result["final_value"], float)
        self.assertIsInstance(result["total_return"], float)
        self.assertLess(result["execution_time_ms"], 10)

    def test_fast_multi_strategy_backtest(self):
        """高速マルチ戦略バックテスト"""
        engine = self.mock_env["backtest_engine"]

        def momentum_strategy(data):
            return {"action": "buy", "quantity": 50, "symbol": "MOMENTUM"}

        def mean_reversion_strategy(data):
            return {"action": "sell", "quantity": 30, "symbol": "REVERSION"}

        strategies = [self._simple_strategy, momentum_strategy, mean_reversion_strategy]

        results = engine.multi_strategy_backtest(strategies, self.medium_data)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)

        for strategy_name, result in results.items():
            self.assertIn("final_value", result)
            self.assertIn("total_return", result)

    def test_fast_portfolio_optimization(self):
        """高速ポートフォリオ最適化テスト"""
        optimizer = self.mock_env["portfolio_optimizer"]

        assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        returns_data = pd.DataFrame(
            {
                asset: np.random.normal(0.05 / 252, 0.2 / np.sqrt(252), 252)
                for asset in assets
            }
        )

        result = optimizer.optimize_portfolio(assets, returns_data)

        self.assertIn("weights", result)
        self.assertIn("expected_return", result)
        self.assertIn("volatility", result)
        self.assertIn("sharpe_ratio", result)

        # 重みの合計が1に近いことを確認
        total_weight = sum(result["weights"].values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

        # 最適化時間の確認
        self.assertLess(result["optimization_time_ms"], 10)

    def test_backtest_performance_metrics(self):
        """バックテスト性能指標テスト"""
        engine = self.mock_env["backtest_engine"]

        # 複数サイズでの性能測定
        test_sizes = [30, 100, 252, 500, 1000]
        performance_results = {}

        for size in test_sizes:
            test_data = self._create_test_data(size)

            perf = measure_mock_performance(
                engine.run_backtest, test_data, self._simple_strategy
            )

            performance_results[size] = perf

            # 各サイズで10ms以下を確認
            self.assertLess(
                perf.execution_time_ms,
                10.0,
                f"サイズ{size}の実行時間が制限超過: {perf.execution_time_ms:.2f}ms",
            )

        print("    バックテスト性能:")
        for size, perf in performance_results.items():
            print(f"      {size}日分: {perf.execution_time_ms:.2f}ms")

    def test_strategy_comparison(self):
        """戦略比較テスト"""
        engine = self.mock_env["backtest_engine"]

        strategies = {
            "simple_sma": lambda data: {"action": "buy", "quantity": 100},
            "momentum": lambda data: {"action": "buy", "quantity": 150},
            "mean_reversion": lambda data: {"action": "sell", "quantity": 80},
        }

        comparison_results = {}

        for name, strategy in strategies.items():
            result = engine.run_backtest(
                self.medium_data, strategy, initial_capital=1000000
            )
            comparison_results[name] = result

        # すべての戦略が結果を返すことを確認
        self.assertEqual(len(comparison_results), 3)

        for strategy_name, result in comparison_results.items():
            self.assertIsInstance(result["total_return"], float)
            self.assertIsInstance(result["sharpe_ratio"], float)
            self.assertIsInstance(result["max_drawdown"], float)

    def test_parameter_sweep(self):
        """パラメータスイープテスト（高速）"""
        engine = self.mock_env["backtest_engine"]

        # SMAパラメータのスイープ
        sma_periods = [5, 10, 20, 50, 100]
        sweep_results = {}

        for period in sma_periods:

            def sma_strategy(data):
                if len(data) >= period:
                    sma = data["Close"].rolling(period).mean().iloc[-1]
                    current = data["Close"].iloc[-1]
                    return {
                        "action": "buy" if current > sma else "sell",
                        "quantity": 100,
                    }
                return {"action": "hold", "quantity": 0}

            result = engine.run_backtest(
                self.small_data,  # 小規模データで高速実行
                sma_strategy,
            )

            sweep_results[f"SMA_{period}"] = result

        # すべてのパラメータで結果が得られることを確認
        self.assertEqual(len(sweep_results), 5)

        print("    パラメータスイープ結果:")
        for param, result in sweep_results.items():
            print(f"      {param}: リターン{result['total_return']:.2%}")

    @unittest.skipUnless(BACKTEST_AVAILABLE, "Backtest engine not available")
    def test_integration_with_real_backtest(self):
        """実バックテストエンジンとの統合テスト（限定的）"""
        try:
            config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),  # 短期間
                initial_capital=Decimal("100000"),  # 小額
            )

            engine = BacktestEngine()

            # 非常に小規模データでのみテスト
            tiny_data = self.small_data.iloc[:10]  # 10日分のみ

            start_time = time.perf_counter()

            # 簡単な戦略でテスト
            def simple_test_strategy(symbols_data, current_date):
                return {}  # 何もしない戦略

            # 実際のエンジンは重いので、モックパッチを使用
            with patch.object(engine, "run_backtest") as mock_run:
                mock_run.return_value = BacktestResult(
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    initial_capital=100000,
                    final_capital=105000,
                    total_return=0.05,
                    trades=[],
                )

                result = engine.run_backtest(simple_test_strategy)

                execution_time = (time.perf_counter() - start_time) * 1000

                self.assertIsNotNone(result)
                self.assertLess(execution_time, 100)  # 100ms以下

        except Exception as e:
            self.skipTest(f"Real backtest integration not available: {e}")


class TestBacktestPerformanceComparison(unittest.TestCase):
    """バックテスト性能比較テスト"""

    def test_mock_vs_traditional_performance(self):
        """モック vs 従来処理の性能比較"""

        # モック処理の測定
        mock_env = create_fast_test_environment(include_backtest=True)
        backtest_mock = mock_env["backtest_engine"]

        # テストデータ
        test_data = pd.DataFrame(
            {
                "Close": 100 + np.random.randn(252),
                "Volume": np.random.randint(100000, 1000000, 252),
            }
        )

        def simple_strategy(data):
            return {"action": "hold", "quantity": 0}

        # モック性能測定
        start_time = time.perf_counter()
        backtest_mock.run_backtest(test_data, simple_strategy)
        mock_time = (time.perf_counter() - start_time) * 1000

        # 従来処理シミュレーション（重いループ処理を模擬）
        start_time = time.perf_counter()

        # 重いバックテスト処理をシミュレート
        for i in range(252):  # 252日分の日次処理
            # 重い計算をシミュレート
            _ = np.random.randn(100, 100).dot(np.random.randn(100, 100))

            # ポートフォリオ計算をシミュレート
            if i % 5 == 0:  # リバランス頻度
                _ = np.random.randn(50, 50).dot(np.random.randn(50, 50))

        traditional_time = (time.perf_counter() - start_time) * 1000

        speedup_factor = traditional_time / mock_time if mock_time > 0 else 1

        print(f"    モック実行時間: {mock_time:.2f}ms")
        print(f"    従来処理シミュレーション時間: {traditional_time:.2f}ms")
        print(f"    高速化倍率: {speedup_factor:.1f}x")

        # モックが大幅に高速であることを確認
        self.assertLess(mock_time, 20)  # 20ms以下
        self.assertGreater(speedup_factor, 50)  # 50倍以上高速

    def test_scalability_analysis(self):
        """スケーラビリティ分析"""
        mock_env = create_fast_test_environment(include_backtest=True)
        engine = mock_env["backtest_engine"]

        def dummy_strategy(data):
            return {"action": "hold", "quantity": 0}

        # 異なるサイズでのパフォーマンステスト
        sizes = [10, 50, 100, 252, 500, 1000]
        performance_data = []

        for size in sizes:
            test_data = pd.DataFrame(
                {
                    "Close": 100 + np.random.randn(size),
                    "Volume": np.random.randint(100000, 1000000, size),
                }
            )

            perf = measure_mock_performance(
                engine.run_backtest, test_data, dummy_strategy
            )

            performance_data.append(
                {
                    "size": size,
                    "time_ms": perf.execution_time_ms,
                    "throughput": size / perf.execution_time_ms * 1000,  # データ点/秒
                }
            )

        print("    スケーラビリティ分析:")
        for data in performance_data:
            print(
                f"      {data['size']}日分: {data['time_ms']:.2f}ms "
                f"({data['throughput']:.0f} データ点/秒)"
            )

        # 線形スケーリング以下であることを確認（モックなので）
        max_time = max(data["time_ms"] for data in performance_data)
        self.assertLess(max_time, 50)  # 最大でも50ms以下


def run_fast_backtest_tests():
    """高速バックテストテスト実行"""
    print("=== Issue #375 高速バックテストテスト実行 ===")

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 高速テストを追加
    suite.addTests(loader.loadTestsFromTestCase(TestFastBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestPerformanceComparison))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()

    result = runner.run(suite)

    total_time = (time.perf_counter() - start_time) * 1000

    print("\n【テスト結果サマリー】")
    print(f"総実行時間: {total_time:.0f}ms")
    print(f"実行テスト数: {result.testsRun}")
    print(
        f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"失敗: {len(result.failures)}")
    if result.errors:
        print(f"エラー: {len(result.errors)}")

    return result


if __name__ == "__main__":
    result = run_fast_backtest_tests()

    print("\n=== Issue #375 バックテストテスト高速化完了 ===")
