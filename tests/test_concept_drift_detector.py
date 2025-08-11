import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.day_trade.ml.concept_drift_detector import ConceptDriftDetector

class TestConceptDriftDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ConceptDriftDetector(metric_threshold=0.1, window_size=10)

    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.metric_threshold, 0.1)
        self.assertEqual(self.detector.window_size, 10)
        self.assertEqual(self.detector.performance_history, [])
        self.assertIsNone(self.detector.baseline_metric)

    def test_add_performance_data(self):
        predictions = np.array([1, 2, 3])
        actuals = np.array([1.1, 2.2, 3.3])
        self.detector.add_performance_data(predictions, actuals)
        self.assertEqual(len(self.detector.performance_history), 1)
        self.assertIn('mae', self.detector.performance_history[0])
        self.assertIn('rmse', self.detector.performance_history[0])

        # 履歴サイズ制限のテスト
        for i in range(15):
            self.detector.add_performance_data(predictions, actuals)
        self.assertEqual(len(self.detector.performance_history), 10) # window_sizeに制限される

    def test_detect_drift_no_drift(self):
        # 安定した性能データを追加
        for i in range(10):
            predictions = np.random.rand(10) * 10
            actuals = predictions + np.random.normal(0, 0.5, 10) # 小さなノイズ
            self.detector.add_performance_data(predictions, actuals)
        
        # ベースラインが設定されることを確認
        self.assertIsNotNone(self.detector.baseline_metric)

        # ドリフト検出
        result = self.detector.detect_drift()
        self.assertFalse(result['drift_detected'])
        self.assertIn("性能は安定しています", result['reason'])

    def test_detect_drift_with_drift(self):
        # 安定した性能データを追加してベースラインを設定
        for i in range(5):
            predictions = np.random.rand(10) * 10
            actuals = predictions + np.random.normal(0, 0.5, 10)
            self.detector.add_performance_data(predictions, actuals)
        
        # ベースラインが設定されることを確認
        self.assertIsNotNone(self.detector.baseline_metric)

        # 意図的に性能を低下させるデータを追加
        for i in range(5):
            predictions = np.random.rand(10) * 10
            actuals = predictions + np.random.normal(0, 5.0, 10) # 大きなノイズでMAEを増加
            self.detector.add_performance_data(predictions, actuals)

        # ドリフト検出
        result = self.detector.detect_drift()
        self.assertTrue(result['drift_detected'])
        self.assertIn("MAEが閾値", result['reason'])
        self.assertGreater(result['mae_increase_ratio'], self.detector.metric_threshold)

    def test_reset_baseline(self):
        predictions = np.random.rand(10)
        actuals = predictions + np.random.normal(0, 0.5, 10)
        self.detector.add_performance_data(predictions, actuals)
        self.detector.detect_drift() # ベースラインを設定
        self.assertIsNotNone(self.detector.baseline_metric)

        self.detector.reset_baseline()
        self.assertIsNone(self.detector.baseline_metric)

    def test_get_performance_summary(self):
        predictions = np.random.rand(10)
        actuals = predictions + np.random.normal(0, 0.5, 10)
        self.detector.add_performance_data(predictions, actuals)
        summary = self.detector.get_performance_summary()
        self.assertIn('latest_mae', summary)
        self.assertIn('average_mae', summary)
        self.assertIn('history_length', summary)
        self.assertEqual(summary['history_length'], 1)

        # データがない場合
        empty_detector = ConceptDriftDetector()
        empty_summary = empty_detector.get_performance_summary()
        self.assertEqual(empty_summary['status'], 'no_data')

if __name__ == '__main__':
    unittest.main()
