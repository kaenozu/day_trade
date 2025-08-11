import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

from src.day_trade.ml.data_drift_detector import DataDriftDetector

class TestDataDriftDetector(unittest.TestCase):

    def setUp(self):
        self.test_file_path = Path("test_baseline_stats.json")
        self.detector = DataDriftDetector(threshold=0.05)

        # ベースラインデータ
        self.baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.randint(0, 10, 100)
        })

    def tearDown(self):
        if self.test_file_path.exists():
            self.test_file_path.unlink()

    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.threshold, 0.05)
        self.assertEqual(self.detector.baseline_stats, {})

    def test_fit_method(self):
        self.detector.fit(self.baseline_data)
        self.assertIn('feature1', self.detector.baseline_stats)
        self.assertIn('feature2', self.detector.baseline_stats)
        self.assertIn('feature3', self.detector.baseline_stats)
        self.assertAlmostEqual(self.detector.baseline_stats['feature1']['mean'], 0, delta=0.2)
        self.assertAlmostEqual(self.detector.baseline_stats['feature2']['std'], 2, delta=0.2)

    def test_detect_drift_no_drift(self):
        self.detector.fit(self.baseline_data)
        
        # ドリフトなしの新しいデータ (ベースラインと同じ分布)
        new_data_no_drift = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.randint(0, 10, 100)
        })
        results = self.detector.detect_drift(new_data_no_drift)
        self.assertFalse(results['drift_detected'])
        self.assertFalse(results['features']['feature1']['drift_detected'])

    def test_detect_drift_with_drift(self):
        self.detector.fit(self.baseline_data)

        # ドリフトありの新しいデータ (平均が異なる)
        new_data_with_drift = pd.DataFrame({
            'feature1': np.random.normal(5, 1, 100), # 平均をずらす
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.randint(0, 10, 100)
        })
        results = self.detector.detect_drift(new_data_with_drift)
        self.assertTrue(results['drift_detected'])
        self.assertTrue(results['features']['feature1']['drift_detected'])

    def test_save_load_baseline(self):
        self.detector.fit(self.baseline_data)
        self.detector.save_baseline(str(self.test_file_path))
        self.assertTrue(self.test_file_path.exists())

        new_detector = DataDriftDetector()
        new_detector.load_baseline(str(self.test_file_path))

        self.assertIn('feature1', new_detector.baseline_stats)
        self.assertAlmostEqual(new_detector.baseline_stats['feature1']['mean'],
                               self.detector.baseline_stats['feature1']['mean'])
        # 'values'キーはリストとして保存され、NumPy配列としてロードされるため、直接比較はしない
        self.assertIsInstance(new_detector.baseline_stats['feature1']['values'], np.ndarray)
        self.assertEqual(len(new_detector.baseline_stats['feature1']['values']), 100)

    def test_non_numeric_columns(self):
        self.detector.fit(self.baseline_data)
        data_with_non_numeric = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'category': ['A', 'B'] * 50
        })
        results = self.detector.detect_drift(data_with_non_numeric)
        self.assertFalse(results['features']['category']['drift_detected'])
        self.assertIn('reason', results['features']['category'])

    def test_empty_new_data_series(self):
        self.detector.fit(self.baseline_data)
        empty_data = pd.DataFrame({
            'feature1': [np.nan] * 100,
            'feature2': np.random.normal(10, 2, 100)
        })
        results = self.detector.detect_drift(empty_data)
        self.assertIn('feature1', results['features'])
        self.assertIn('error', results['features']['feature1'])

if __name__ == '__main__':
    unittest.main()
