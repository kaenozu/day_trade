#!/usr/bin/env python3
"""
A/Bテストフレームワーク テストスイート

Issue #733対応: A/Bテストフレームワークの包括的テスト
"""

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

# テスト対象のインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.day_trade.ml.ab_testing_framework import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentResult,
    TrafficSplitter,
    StatisticalAnalyzer,
    TrafficSplitStrategy,
    ExperimentStatus,
    create_simple_ab_experiment
)


class TestTrafficSplitter(unittest.TestCase):
    """トラフィック分割システムのテスト"""

    def setUp(self):
        """テスト前準備"""
        # シンプルなA/B実験設定
        self.experiment_config = create_simple_ab_experiment(
            experiment_name="Test Experiment",
            control_config={"model": "control"},
            test_config={"model": "test"},
            traffic_split=0.3
        )

        self.splitter = TrafficSplitter(self.experiment_config)

    def test_traffic_allocation_consistency(self):
        """トラフィック配分の一貫性テスト"""
        identifier = "test_symbol_7203"

        # 同じ識別子で複数回呼び出し
        assignments = []
        for _ in range(10):
            group_id = self.splitter.assign_to_group(identifier)
            assignments.append(group_id)

        # ハッシュベースでは同じ識別子は常に同じグループ
        if self.splitter.strategy == TrafficSplitStrategy.HASH_BASED:
            self.assertTrue(all(a == assignments[0] for a in assignments))

    def test_traffic_distribution(self):
        """トラフィック分散の検証"""
        # 大量の識別子で分散を確認
        assignments = []
        for i in range(1000):
            identifier = f"symbol_{i}"
            group_id = self.splitter.assign_to_group(identifier)
            assignments.append(group_id)

        # 各グループの割合を計算
        control_count = assignments.count("control")
        test_count = assignments.count("test")

        control_ratio = control_count / len(assignments)
        test_ratio = test_count / len(assignments)

        # 期待値に近いかチェック（±5%の誤差許容）
        expected_control_ratio = 0.7  # 70%
        expected_test_ratio = 0.3     # 30%

        self.assertAlmostEqual(control_ratio, expected_control_ratio, delta=0.05)
        self.assertAlmostEqual(test_ratio, expected_test_ratio, delta=0.05)

    def test_symbol_based_assignment(self):
        """銘柄ベース割り当てテスト"""
        symbol_config = create_simple_ab_experiment(
            experiment_name="Symbol Based Test",
            control_config={"model": "control"},
            test_config={"model": "test"},
            traffic_split=0.5
        )
        symbol_config.traffic_split_strategy = TrafficSplitStrategy.SYMBOL_BASED

        symbol_splitter = TrafficSplitter(symbol_config)

        # 同じ銘柄は常に同じグループに
        assignments = []
        for _ in range(5):
            group_id = symbol_splitter.assign_to_group("7203")
            assignments.append(group_id)

        self.assertTrue(all(a == assignments[0] for a in assignments))


class TestStatisticalAnalyzer(unittest.TestCase):
    """統計分析エンジンのテスト"""

    def setUp(self):
        """テスト前準備"""
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)

    def test_t_test_significant_difference(self):
        """統計的有意差のあるt検定"""
        # 明確に異なる2つのグループ
        group_a = np.random.normal(100, 10, 100)  # 平均100
        group_b = np.random.normal(110, 10, 100)  # 平均110

        result = self.analyzer.perform_t_test(group_a, group_b)

        # 結果の検証
        self.assertIsInstance(result.statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertIsNotNone(result.effect_size)

        # 有意差があるはず
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

    def test_t_test_no_difference(self):
        """統計的有意差のないt検定"""
        # 同じ分布の2つのグループ
        group_a = np.random.normal(100, 10, 50)
        group_b = np.random.normal(100, 10, 50)

        result = self.analyzer.perform_t_test(group_a, group_b)

        # 有意差はない可能性が高い
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size, float)

    def test_chi_squared_test(self):
        """カイ二乗検定のテスト"""
        # 観測度数の例
        observed_a = np.array([50, 30, 20])  # グループA
        observed_b = np.array([40, 35, 25])  # グループB

        result = self.analyzer.perform_chi_squared_test(observed_a, observed_b)

        # 結果の検証
        self.assertIsInstance(result.statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsNotNone(result.effect_size)


class TestABTestingFramework(unittest.TestCase):
    """A/Bテストフレームワークのテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ABTestingFramework(storage_path=self.temp_dir)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_create_experiment(self):
        """実験作成のテスト"""
        config = create_simple_ab_experiment(
            experiment_name="Test Experiment Creation",
            control_config={"model_type": "RandomForest"},
            test_config={"model_type": "XGBoost"},
            traffic_split=0.4
        )

        # 非同期テスト実行
        async def run_test():
            success = await self.framework.create_experiment(config)
            self.assertTrue(success)

            # 実験がフレームワークに登録されているかチェック
            self.assertIn(config.experiment_id, self.framework.active_experiments)
            self.assertIn(config.experiment_id, self.framework.traffic_splitters)

            return True

        # asyncioでテスト実行
        result = asyncio.run(run_test())
        self.assertTrue(result)

    def test_assign_to_experiment(self):
        """実験グループ割り当てのテスト"""
        config = create_simple_ab_experiment(
            experiment_name="Assignment Test",
            control_config={"model": "control"},
            test_config={"model": "test"}
        )

        async def run_test():
            # 実験作成
            await self.framework.create_experiment(config)

            # 実験をアクティブに
            self.framework.active_experiments[config.experiment_id].status = ExperimentStatus.ACTIVE

            # グループ割り当て
            group_id = await self.framework.assign_to_experiment(
                config.experiment_id, "test_symbol"
            )

            self.assertIsNotNone(group_id)
            self.assertIn(group_id, ["control", "test"])

            return True

        result = asyncio.run(run_test())
        self.assertTrue(result)

    def test_record_result(self):
        """実験結果記録のテスト"""
        config = create_simple_ab_experiment(
            experiment_name="Result Recording Test",
            control_config={"model": "control"},
            test_config={"model": "test"}
        )

        async def run_test():
            # 実験作成
            await self.framework.create_experiment(config)

            # テスト結果の作成
            result = ExperimentResult(
                result_id="test_result_1",
                experiment_id=config.experiment_id,
                group_id="control",
                symbol="7203",
                timestamp=datetime.now(),
                prediction=105.5,
                actual_value=107.2,
                latency_ms=45.0
            )

            # 結果の記録
            success = await self.framework.record_result(result)
            self.assertTrue(success)

            # 結果が保存されているかチェック
            results = self.framework.results_storage.get(config.experiment_id, [])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].result_id, "test_result_1")

            return True

        result = asyncio.run(run_test())
        self.assertTrue(result)

    def test_generate_report(self):
        """レポート生成のテスト"""
        config = create_simple_ab_experiment(
            experiment_name="Report Generation Test",
            control_config={"model": "control"},
            test_config={"model": "test"}
        )

        async def run_test():
            # 実験作成
            await self.framework.create_experiment(config)

            # 複数の結果を追加
            results = [
                ExperimentResult(
                    result_id=f"result_{i}",
                    experiment_id=config.experiment_id,
                    group_id="control" if i % 2 == 0 else "test",
                    symbol="7203",
                    timestamp=datetime.now(),
                    prediction=100.0 + np.random.normal(0, 5),
                    actual_value=100.0 + np.random.normal(0, 5),
                    latency_ms=50.0 + np.random.uniform(-10, 10)
                )
                for i in range(100)
            ]

            for result in results:
                await self.framework.record_result(result)

            # レポート生成
            report = await self.framework.generate_report(config.experiment_id)

            self.assertIsNotNone(report)
            self.assertEqual(report.experiment_id, config.experiment_id)
            self.assertEqual(report.total_samples, 100)
            self.assertIsInstance(report.group_metrics, dict)
            self.assertIsInstance(report.statistical_tests, list)

            # グループメトリクスの確認
            self.assertIn("control", report.group_metrics)
            self.assertIn("test", report.group_metrics)

            return True

        result = asyncio.run(run_test())
        self.assertTrue(result)


class TestExperimentScenarios(unittest.TestCase):
    """実験シナリオの統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ABTestingFramework(storage_path=self.temp_dir)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_experiment_workflow(self):
        """完全な実験ワークフローのテスト"""
        async def run_complete_workflow():
            # 1. 実験作成
            config = create_simple_ab_experiment(
                experiment_name="Complete Workflow Test",
                control_config={"model_type": "RandomForest", "n_estimators": 100},
                test_config={"model_type": "GradientBoosting", "n_estimators": 100},
                traffic_split=0.5
            )

            success = await self.framework.create_experiment(config)
            self.assertTrue(success)

            # 実験をアクティブに設定
            self.framework.active_experiments[config.experiment_id].status = ExperimentStatus.ACTIVE

            # 2. 実験データの生成・記録
            symbols = ["7203", "8306", "9984"]

            for i in range(200):
                symbol = np.random.choice(symbols)

                # グループ割り当て
                group_id = await self.framework.assign_to_experiment(
                    config.experiment_id, f"{symbol}_{i}"
                )

                # グループによって異なる性能をシミュレート
                if group_id == "control":
                    prediction = np.random.normal(100, 10)
                    latency = np.random.uniform(40, 60)
                else:  # test group
                    prediction = np.random.normal(105, 10)  # 少し良い性能
                    latency = np.random.uniform(35, 55)     # 少し速い

                actual_value = prediction + np.random.normal(0, 2)

                result = ExperimentResult(
                    result_id=f"workflow_result_{i}",
                    experiment_id=config.experiment_id,
                    group_id=group_id,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    prediction=prediction,
                    actual_value=actual_value,
                    latency_ms=latency
                )

                await self.framework.record_result(result)

            # 3. レポート生成・分析
            report = await self.framework.generate_report(config.experiment_id)

            self.assertIsNotNone(report)
            self.assertEqual(report.total_samples, 200)

            # 統計テストが実行されているか確認
            self.assertGreater(len(report.statistical_tests), 0)

            # グループメトリクスの確認
            control_metrics = report.group_metrics.get("control", {})
            test_metrics = report.group_metrics.get("test", {})

            self.assertGreater(control_metrics.get("sample_size", 0), 0)
            self.assertGreater(test_metrics.get("sample_size", 0), 0)

            # レイテンシがテストグループで改善されているかチェック
            control_latency = control_metrics.get("avg_latency_ms", 0)
            test_latency = test_metrics.get("avg_latency_ms", 0)

            # テストグループの方が速いはず（シミュレートされたデータ）
            self.assertLess(test_latency, control_latency + 5)  # 5ms以内の誤差許容

            return True

        result = asyncio.run(run_complete_workflow())
        self.assertTrue(result)


def run_ab_testing_tests():
    """A/Bテストフレームワークのテスト実行"""
    print("=" * 60)
    print("A/Bテストフレームワーク テストスイート実行")
    print("=" * 60)

    # テストスイートの作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_suite.addTest(unittest.makeSuite(TestTrafficSplitter))
    test_suite.addTest(unittest.makeSuite(TestStatisticalAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestABTestingFramework))
    test_suite.addTest(unittest.makeSuite(TestExperimentScenarios))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("A/Bテストフレームワーク テストサマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures[:3]:
            print(f"- {test}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors[:3]:
            print(f"- {test}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ テスト成功率: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ab_testing_tests()
    exit(0 if success else 1)