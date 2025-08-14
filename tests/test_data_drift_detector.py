import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.day_trade.ml.data_drift_detector import DataDriftDetector


class TestDataDriftDetector(unittest.TestCase):
    def setUp(self):
        self.test_file_path = Path("test_baseline_stats.json")
        self.detector = DataDriftDetector(threshold=0.05)

        # ベースラインデータ
        self.baseline_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(10, 2, 100),
                "feature3": np.random.randint(0, 10, 100),
            }
        )

    def tearDown(self):
        if self.test_file_path.exists():
            self.test_file_path.unlink()

    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.threshold, 0.05)
        self.assertEqual(self.detector.baseline_stats, {})

    def test_fit_method(self):
        self.detector.fit(self.baseline_data)
        self.assertIn("feature1", self.detector.baseline_stats)
        self.assertIn("feature2", self.detector.baseline_stats)
        self.assertIn("feature3", self.detector.baseline_stats)
        self.assertAlmostEqual(
            self.detector.baseline_stats["feature1"]["mean"], 0, delta=0.2
        )
        self.assertAlmostEqual(
            self.detector.baseline_stats["feature2"]["std"], 2, delta=0.2
        )

    def test_detect_drift_no_drift(self):
        self.detector.fit(self.baseline_data)

        # ドリフトなしの新しいデータ (ベースラインと同じ分布)
        new_data_no_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(10, 2, 100),
                "feature3": np.random.randint(0, 10, 100),
            }
        )
        results = self.detector.detect_drift(new_data_no_drift)
        self.assertFalse(results["drift_detected"])
        self.assertFalse(results["features"]["feature1"]["drift_detected"])

    def test_detect_drift_with_drift(self):
        self.detector.fit(self.baseline_data)

        # ドリフトありの新しいデータ (平均が異なる)
        new_data_with_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(5, 1, 100),  # 平均をずらす
                "feature2": np.random.normal(10, 2, 100),
                "feature3": np.random.randint(0, 10, 100),
            }
        )
        results = self.detector.detect_drift(new_data_with_drift)
        self.assertTrue(results["drift_detected"])
        self.assertTrue(results["features"]["feature1"]["drift_detected"])

    def test_save_load_baseline(self):
        self.detector.fit(self.baseline_data)
        self.detector.save_baseline(str(self.test_file_path))
        self.assertTrue(self.test_file_path.exists())

        new_detector = DataDriftDetector()
        new_detector.load_baseline(str(self.test_file_path))

        self.assertIn("feature1", new_detector.baseline_stats)
        self.assertAlmostEqual(
            new_detector.baseline_stats["feature1"]["mean"],
            self.detector.baseline_stats["feature1"]["mean"],
        )
        # 'values'キーはリストとして保存され、NumPy配列としてロードされるため、直接比較はしない
        self.assertIsInstance(
            new_detector.baseline_stats["feature1"]["values"], np.ndarray
        )
        self.assertEqual(len(new_detector.baseline_stats["feature1"]["values"]), 100)

    def test_non_numeric_columns(self):
        self.detector.fit(self.baseline_data)
        data_with_non_numeric = pd.DataFrame(
            {"feature1": np.random.normal(0, 1, 100), "category": ["A", "B"] * 50}
        )
        results = self.detector.detect_drift(data_with_non_numeric)
        self.assertFalse(results["features"]["category"]["drift_detected"])
        self.assertIn("reason", results["features"]["category"])

    def test_empty_new_data_series(self):
        self.detector.fit(self.baseline_data)
        empty_data = pd.DataFrame(
            {"feature1": [np.nan] * 100, "feature2": np.random.normal(10, 2, 100)}
        )
        results = self.detector.detect_drift(empty_data)
        self.assertIn("feature1", results["features"])
        self.assertIn("error", results["features"]["feature1"])


class TestDataQuality(unittest.TestCase):
    """Issue #576: データ品質計算ロジックテスト"""

    def setUp(self):
        """テストデータシナリオ作成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        base_price = 100
        price_changes = np.random.randn(60) * 0.02
        prices = base_price * np.exp(np.cumsum(price_changes))
        volumes = np.random.randint(50000, 200000, 60)

        self.high_quality_data = pd.DataFrame({
            '始値': np.roll(prices, 1),
            '高値': prices * 1.01,
            '安値': prices * 0.99,
            '終値': prices,
            '出来高': volumes,
        }, index=dates)

    def test_data_quality_calculation_high_quality(self):
        """高品質データの品質スコア計算テスト"""
        try:
            from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

            fetcher = AdvancedBatchDataFetcher(
                max_workers=1,
                enable_kafka=False,
                enable_redis=False
            )

            quality_score = fetcher._calculate_data_quality(self.high_quality_data)

            # 高品質データは80点以上のスコアを期待
            self.assertGreaterEqual(quality_score, 80.0)

        except ImportError:
            self.skipTest("AdvancedBatchDataFetcher not available")

    def test_data_quality_with_missing_values(self):
        """欠損値ありデータの品質スコアテスト"""
        try:
            from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

            fetcher = AdvancedBatchDataFetcher(
                max_workers=1,
                enable_kafka=False,
                enable_redis=False
            )

            # 欠損値を含むデータ作成
            missing_data = self.high_quality_data.copy()
            missing_data.iloc[10:15, 1] = np.nan  # 高値に欠損
            missing_data.iloc[20:25, 4] = np.nan  # 出来高に欠損

            quality_score = fetcher._calculate_data_quality(missing_data)
            high_quality_score = fetcher._calculate_data_quality(self.high_quality_data)

            # 欠損値ありデータは元データより品質スコアが低い
            self.assertLess(quality_score, high_quality_score)

        except ImportError:
            self.skipTest("AdvancedBatchDataFetcher not available")

    def test_data_quality_insufficient_volume(self):
        """データ量不足の品質スコアテスト"""
        try:
            from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

            fetcher = AdvancedBatchDataFetcher(
                max_workers=1,
                enable_kafka=False,
                enable_redis=False
            )

            # データ量不足（10日間のみ）
            insufficient_data = self.high_quality_data.iloc[:10].copy()

            quality_score = fetcher._calculate_data_quality(insufficient_data)
            high_quality_score = fetcher._calculate_data_quality(self.high_quality_data)

            # データ量不足は品質スコアが低い
            self.assertLess(quality_score, high_quality_score)

        except ImportError:
            self.skipTest("AdvancedBatchDataFetcher not available")

    def test_data_quality_with_extremes(self):
        """異常値含みデータの品質スコアテスト"""
        try:
            from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

            fetcher = AdvancedBatchDataFetcher(
                max_workers=1,
                enable_kafka=False,
                enable_redis=False
            )

            # 異常値含みデータ
            extreme_data = self.high_quality_data.copy()
            extreme_data.iloc[30, 3] = extreme_data.iloc[29, 3] * 1.3  # 30%上昇
            extreme_data.iloc[35, 3] = extreme_data.iloc[34, 3] * 0.7  # 30%下落

            quality_score = fetcher._calculate_data_quality(extreme_data)
            high_quality_score = fetcher._calculate_data_quality(self.high_quality_data)

            # 異常値含みデータは品質スコアが低い
            self.assertLess(quality_score, high_quality_score)

        except ImportError:
            self.skipTest("AdvancedBatchDataFetcher not available")

    def test_quality_metrics_components(self):
        """品質メトリクス構成要素テスト"""
        try:
            from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

            fetcher = AdvancedBatchDataFetcher(
                max_workers=1,
                enable_kafka=False,
                enable_redis=False
            )

            # 詳細メトリクス計算
            metrics = fetcher._calculate_quality_metrics(self.high_quality_data)

            # 基本メトリクスの存在確認
            required_metrics = [
                'missing_ratio', 'data_volume', 'duplicate_ratio',
                'invalid_numeric_ratio'
            ]

            for metric in required_metrics:
                self.assertIn(metric, metrics)

            # 高品質データの基本チェック
            self.assertEqual(metrics['missing_ratio'], 0.0)  # 欠損なし
            self.assertEqual(metrics['duplicate_ratio'], 0.0)  # 重複なし
            self.assertGreater(metrics['data_volume'], 50)  # 十分なデータ量

        except ImportError:
            self.skipTest("AdvancedBatchDataFetcher not available")


if __name__ == "__main__":
    unittest.main()
