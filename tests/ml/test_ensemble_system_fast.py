#!/usr/bin/env python3
"""
Issue #471対応: EnsembleSystemの高速・軽量テストスイート

軽量かつ実用的なテスト：
- モックによる高速テスト
- プレースホルダー完全削除
- 重要機能に特化
- 実行時間最小化
"""

import unittest
import numpy as np
import tempfile
import time
import warnings
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.ml.ensemble_system import (
        EnsembleSystem,
        EnsembleConfig,
        EnsemblePrediction
    )
    from src.day_trade.ml.base_models.base_model_interface import ModelPrediction
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"EnsembleSystem import failed: {e}")
    ENSEMBLE_AVAILABLE = False

warnings.filterwarnings("ignore")


class TestEnsembleSystemFast(unittest.TestCase):
    """EnsembleSystemの高速テスト"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        # 最低限の設定
        self.config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False,
            enable_stacking=False,
            enable_dynamic_weighting=False
        )

    def test_initialization(self):
        """初期化テスト"""
        ensemble = EnsembleSystem(self.config)

        self.assertIsInstance(ensemble, EnsembleSystem)
        self.assertFalse(ensemble.is_trained)
        self.assertEqual(len(ensemble.base_models), 1)

    @patch('src.day_trade.ml.base_models.random_forest_model.RandomForestModel')
    def test_training_with_mock(self, mock_rf_class):
        """モックを使った学習テスト"""
        # モック設定
        mock_model = Mock()
        mock_model.fit.return_value = {
            'status': '成功',
            'training_time': 0.5,
            'model_metrics': {'accuracy': 0.85}
        }
        mock_model.is_trained = True
        mock_rf_class.return_value = mock_model

        # テスト実行
        ensemble = EnsembleSystem(self.config)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        result = ensemble.fit(X, y, ["f1", "f2", "f3"])

        # 検証
        self.assertIsInstance(result, dict)
        self.assertTrue(ensemble.is_trained)
        mock_model.fit.assert_called_once()

    @patch('src.day_trade.ml.base_models.random_forest_model.RandomForestModel')
    def test_prediction_with_mock(self, mock_rf_class):
        """モックを使った予測テスト"""
        # モック設定
        mock_model = Mock()
        mock_model.fit.return_value = {'status': '成功', 'training_time': 0.1}
        mock_model.predict.return_value = ModelPrediction(
            predictions=np.array([1.0, 2.0, 3.0]),
            probabilities=None,
            confidence_scores=np.array([0.8, 0.9, 0.7]),
            prediction_time=0.05
        )
        mock_model.is_trained = True
        mock_rf_class.return_value = mock_model

        # テスト実行
        ensemble = EnsembleSystem(self.config)
        X_train = np.random.randn(10, 3)
        y_train = np.random.randn(10)
        X_test = np.random.randn(3, 3)

        ensemble.fit(X_train, y_train, ["f1", "f2", "f3"])
        prediction = ensemble.predict(X_test)

        # 検証
        self.assertIsInstance(prediction, EnsemblePrediction)
        self.assertEqual(len(prediction.final_predictions), 3)
        mock_model.predict.assert_called_once()

    def test_untrained_prediction_error(self):
        """未学習状態での予測エラーテスト"""
        ensemble = EnsembleSystem(self.config)
        X = np.random.randn(5, 3)

        with self.assertRaises((RuntimeError, ValueError)):
            ensemble.predict(X)

    def test_invalid_config_error(self):
        """不正設定でのエラーテスト"""
        invalid_config = EnsembleConfig(
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False
        )

        with self.assertRaises((ValueError, RuntimeError)):
            EnsembleSystem(invalid_config)

    def test_empty_data_error(self):
        """空データでのエラーテスト"""
        ensemble = EnsembleSystem(self.config)
        empty_X = np.array([]).reshape(0, 3)
        empty_y = np.array([])

        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(empty_X, empty_y, ["f1", "f2", "f3"])

    def test_dimension_mismatch_error(self):
        """次元不一致でのエラーテスト"""
        ensemble = EnsembleSystem(self.config)
        X = np.random.randn(10, 3)
        y = np.random.randn(5)  # 意図的な不一致

        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(X, y, ["f1", "f2", "f3"])

    def test_save_load_basic(self):
        """基本的な保存・読み込みテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble = EnsembleSystem(self.config)
            save_path = Path(temp_dir) / "test_ensemble.json"

            # 保存テスト
            result = ensemble.save_ensemble(str(save_path))
            self.assertTrue(result)
            self.assertTrue(save_path.exists())

            # 読み込みテスト
            new_ensemble = EnsembleSystem(self.config)
            load_result = new_ensemble.load_ensemble(str(save_path))
            self.assertTrue(load_result)

    def test_performance_timing(self):
        """パフォーマンス時間計測テスト"""
        start_time = time.time()

        # 軽量初期化
        ensemble = EnsembleSystem(self.config)

        init_time = time.time() - start_time

        # 1秒以内の初期化を期待
        self.assertLess(init_time, 1.0)


class TestEnsembleSystemMockIntegration(unittest.TestCase):
    """モック統合テスト"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

    @patch('src.day_trade.ml.base_models.random_forest_model.RandomForestModel')
    @patch('src.day_trade.ml.dynamic_weighting_system.DynamicWeightingSystem')
    def test_full_workflow_mock(self, mock_dw_class, mock_rf_class):
        """完全ワークフローのモックテスト"""
        # RandomForestModelのモック
        mock_rf = Mock()
        mock_rf.fit.return_value = {'status': '成功', 'training_time': 0.1}
        mock_rf.predict.return_value = ModelPrediction(
            predictions=np.array([1.0, 2.0]),
            probabilities=None,
            confidence_scores=np.array([0.8, 0.9]),
            prediction_time=0.01
        )
        mock_rf.is_trained = True
        mock_rf_class.return_value = mock_rf

        # DynamicWeightingSystemのモック
        mock_dw = Mock()
        mock_dw.update_weights.return_value = {'random_forest': 1.0}
        mock_dw.get_current_weights.return_value = {'random_forest': 1.0}
        mock_dw_class.return_value = mock_dw

        # テスト実行
        config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False,
            enable_dynamic_weighting=True
        )

        ensemble = EnsembleSystem(config)

        # 学習
        X_train = np.random.randn(10, 2)
        y_train = np.random.randn(10)
        ensemble.fit(X_train, y_train, ["f1", "f2"])

        # 予測
        X_test = np.random.randn(2, 2)
        prediction = ensemble.predict(X_test)

        # 検証
        self.assertTrue(ensemble.is_trained)
        self.assertEqual(len(prediction.final_predictions), 2)
        mock_rf.fit.assert_called_once()
        mock_rf.predict.assert_called_once()


def run_fast_ensemble_tests():
    """高速EnsembleSystemテストの実行"""
    print("=" * 50)
    print("Issue #471対応: EnsembleSystem高速テストスイート")
    print("=" * 50)

    start_time = time.time()

    # テストスイート作成
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemFast))
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemMockIntegration))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    execution_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("高速テストサマリー")
    print("=" * 50)
    print(f"実行テスト数: {result.testsRun}")
    print(f"実行時間: {execution_time:.2f}秒")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"成功率: {success_rate:.1f}%")

    if result.failures:
        print("\n失敗:")
        for test, traceback in result.failures:
            print(f"- {test}")

    if result.errors:
        print("\nエラー:")
        for test, traceback in result.errors:
            print(f"- {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_fast_ensemble_tests()
    exit(0 if success else 1)