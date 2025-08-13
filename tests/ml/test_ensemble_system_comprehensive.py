#!/usr/bin/env python3
"""
EnsembleSystem包括的テストスイート

Issue #471対応: EnsembleSystemのテストカバレッジと信頼性向上

包括的テストケース:
- 正常ケース・異常ケース
- エッジケース・パフォーマンステスト
- エラーハンドリング・モック使用
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import warnings
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import pytest

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.ml.ensemble_system import (
        EnsembleSystem,
        EnsembleConfig,
        EnsembleMethod,
        EnsemblePrediction
    )
    from src.day_trade.ml.base_models.base_model_interface import ModelPrediction
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"EnsembleSystem import failed: {e}")
    ENSEMBLE_AVAILABLE = False

warnings.filterwarnings("ignore")


class TestEnsembleSystemBasic(unittest.TestCase):
    """EnsembleSystem基本機能テスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        # テスト用設定
        self.test_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,  # テスト高速化のためSVRは無効化
            enable_stacking=False,  # テスト高速化のためスタッキングは無効化
            enable_dynamic_weighting=False,
        )

        # テストデータ生成
        np.random.seed(42)
        self.test_X = np.random.randn(100, 5)
        self.test_y = np.random.randn(100)
        self.test_feature_names = [f"feature_{i}" for i in range(5)]

    def test_ensemble_system_initialization(self):
        """EnsembleSystem初期化テスト"""
        # 正常初期化
        ensemble = EnsembleSystem(self.test_config)

        # 初期状態確認
        self.assertFalse(ensemble.is_trained)
        self.assertIsNotNone(ensemble.base_models)
        self.assertIsInstance(ensemble.model_weights, dict)
        self.assertIn('random_forest', ensemble.base_models)
        self.assertIn('gradient_boosting', ensemble.base_models)

    def test_ensemble_system_training(self):
        """アンサンブル学習テスト"""
        ensemble = EnsembleSystem(self.test_config)

        # 学習実行
        training_results = ensemble.fit(
            self.test_X,
            self.test_y,
            feature_names=self.test_feature_names
        )

        # 学習結果検証
        self.assertTrue(ensemble.is_trained)
        self.assertIsInstance(training_results, dict)
        self.assertIn('random_forest', training_results)
        self.assertIn('gradient_boosting', training_results)

        # 各モデルの学習結果検証
        for model_name, result in training_results.items():
            self.assertIn('training_time', result)
            self.assertIsInstance(result['training_time'], (int, float))

    def test_ensemble_prediction(self):
        """アンサンブル予測テスト"""
        ensemble = EnsembleSystem(self.test_config)

        # 学習
        ensemble.fit(self.test_X, self.test_y, feature_names=self.test_feature_names)

        # 予測実行
        prediction_result = ensemble.predict(self.test_X[:10])

        # 予測結果検証
        self.assertIsInstance(prediction_result, EnsemblePrediction)
        self.assertEqual(len(prediction_result.final_predictions), 10)
        self.assertIsInstance(prediction_result.individual_predictions, dict)
        self.assertIsInstance(prediction_result.ensemble_confidence, np.ndarray)

        # 個別モデル予測結果確認
        self.assertIn('random_forest', prediction_result.individual_predictions)
        self.assertIn('gradient_boosting', prediction_result.individual_predictions)

    def test_ensemble_methods(self):
        """アンサンブル手法テスト"""
        ensemble = EnsembleSystem(self.test_config)
        ensemble.fit(self.test_X, self.test_y, feature_names=self.test_feature_names)

        # 投票アンサンブル
        voting_result = ensemble.predict(self.test_X[:5], method=EnsembleMethod.VOTING)
        self.assertIsInstance(voting_result, EnsemblePrediction)
        self.assertEqual(voting_result.method_used, 'voting')

        # 重み付きアンサンブル
        weighted_result = ensemble.predict(self.test_X[:5], method=EnsembleMethod.WEIGHTED)
        self.assertIsInstance(weighted_result, EnsemblePrediction)
        self.assertEqual(weighted_result.method_used, 'weighted')


class TestEnsembleSystemEdgeCases(unittest.TestCase):
    """EnsembleSystemエッジケーステスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.minimal_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
        )

    def test_single_model_ensemble(self):
        """単一モデルアンサンブルテスト"""
        ensemble = EnsembleSystem(self.minimal_config)

        # 最小データでテスト
        X_small = np.random.randn(10, 3)
        y_small = np.random.randn(10)

        training_results = ensemble.fit(X_small, y_small)
        self.assertTrue(ensemble.is_trained)

        # 予測実行
        prediction = ensemble.predict(X_small[:2])
        self.assertEqual(len(prediction.final_predictions), 2)

    def test_empty_data_handling(self):
        """空データ処理テスト"""
        ensemble = EnsembleSystem(self.minimal_config)

        # 空のデータでテスト
        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(np.array([]), np.array([]))

    def test_mismatched_data_shapes(self):
        """データ形状不一致テスト"""
        ensemble = EnsembleSystem(self.minimal_config)

        # X, yのサイズが異なる場合
        X_wrong = np.random.randn(10, 3)
        y_wrong = np.random.randn(8)  # サイズ不一致

        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(X_wrong, y_wrong)

    def test_untrained_prediction(self):
        """未学習状態での予測テスト"""
        ensemble = EnsembleSystem(self.minimal_config)

        # 未学習状態で予測を試行
        X_test = np.random.randn(5, 3)
        with self.assertRaises(ValueError):
            ensemble.predict(X_test)

    def test_extreme_values_handling(self):
        """極値処理テスト"""
        ensemble = EnsembleSystem(self.minimal_config)

        # 極値を含むデータ
        X_extreme = np.array([[1e10, -1e10, 0], [np.inf, 1, 2], [1, 2, 3]])
        y_extreme = np.array([1e6, -1e6, 0])

        # inf, nanを除去してテスト
        X_clean = np.nan_to_num(X_extreme, posinf=1e6, neginf=-1e6)

        try:
            ensemble.fit(X_clean, y_extreme)
            prediction = ensemble.predict(X_clean[:1])
            self.assertIsNotNone(prediction)
        except Exception as e:
            # 極値処理でエラーが発生することは想定内
            self.assertIsInstance(e, (ValueError, RuntimeError, np.linalg.LinAlgError))


class TestEnsembleSystemMocking(unittest.TestCase):
    """EnsembleSystemモック使用テスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.mock_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
        )

    @patch('src.day_trade.ml.base_models.RandomForestModel')
    def test_random_forest_model_mock(self, mock_rf_class):
        """RandomForestModelモックテスト"""
        # モックの設定
        mock_rf = Mock()
        mock_rf.fit.return_value = {'training_time': 1.0, 'status': 'success'}
        mock_rf.predict.return_value = ModelPrediction(
            predictions=np.array([0.1, 0.2, 0.3]),
            confidence=np.array([0.8, 0.7, 0.9]),
            feature_importance={'feature_0': 0.5, 'feature_1': 0.3, 'feature_2': 0.2},
            model_name='random_forest',
            processing_time=0.1
        )
        mock_rf.is_trained = True
        mock_rf_class.return_value = mock_rf

        # EnsembleSystemでモックを使用
        ensemble = EnsembleSystem(self.mock_config)

        # テストデータ
        X_test = np.random.randn(3, 3)
        y_test = np.random.randn(3)

        # 学習と予測のテスト
        training_results = ensemble.fit(X_test, y_test)
        self.assertIn('random_forest', training_results)

        prediction = ensemble.predict(X_test)
        self.assertIsNotNone(prediction.final_predictions)

    def test_dynamic_weighting_system_behavior(self):
        """動的重み付けシステム動作テスト"""
        # 動的重み付けを有効にした設定
        dynamic_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,
            enable_stacking=False,
            enable_dynamic_weighting=True,  # 動的重み付け有効
        )

        ensemble = EnsembleSystem(dynamic_config)

        # 動的重み付けシステムが初期化されているかテスト
        self.assertIsNotNone(ensemble.dynamic_weighting)

    def test_model_weights_update(self):
        """モデル重み更新テスト"""
        ensemble = EnsembleSystem(self.mock_config)

        # 初期重み確認
        initial_weights = ensemble.model_weights.copy()
        self.assertIsInstance(initial_weights, dict)

        # テストデータで学習
        X_test = np.random.randn(20, 4)
        y_test = np.random.randn(20)

        ensemble.fit(X_test, y_test)

        # 学習後の重み確認
        post_training_weights = ensemble.model_weights
        self.assertIsInstance(post_training_weights, dict)

        # 重みが正規化されているかチェック
        total_weight = sum(post_training_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestEnsembleSystemPerformance(unittest.TestCase):
    """EnsembleSystemパフォーマンステスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.perf_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,  # パフォーマンステストのため無効化
            use_svr=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
        )

    def test_training_time_reasonable(self):
        """学習時間が妥当であることをテスト"""
        ensemble = EnsembleSystem(self.perf_config)

        # 中規模データでテスト
        X_medium = np.random.randn(200, 10)
        y_medium = np.random.randn(200)

        import time
        start_time = time.time()

        training_results = ensemble.fit(X_medium, y_medium)

        training_time = time.time() - start_time

        # 学習時間が60秒以内であることをテスト
        self.assertLess(training_time, 60.0)

        # 学習結果にトレーニング時間が記録されていることを確認
        for model_name, result in training_results.items():
            self.assertIn('training_time', result)
            self.assertIsInstance(result['training_time'], (int, float))

    def test_prediction_time_reasonable(self):
        """予測時間が妥当であることをテスト"""
        ensemble = EnsembleSystem(self.perf_config)

        # 学習データ
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        ensemble.fit(X_train, y_train)

        # 予測データ
        X_predict = np.random.randn(50, 5)

        import time
        start_time = time.time()

        prediction_result = ensemble.predict(X_predict)

        prediction_time = time.time() - start_time

        # 予測時間が10秒以内であることをテスト
        self.assertLess(prediction_time, 10.0)

        # 予測結果の処理時間が記録されていることを確認
        self.assertIsInstance(prediction_result.processing_time, (int, float))

    def test_memory_usage_reasonable(self):
        """メモリ使用量が妥当であることをテスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # テスト開始時のメモリ使用量
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        ensemble = EnsembleSystem(self.perf_config)

        # 大規模データでテスト
        X_large = np.random.randn(500, 20)
        y_large = np.random.randn(500)

        ensemble.fit(X_large, y_large)
        ensemble.predict(X_large[:100])

        # テスト後のメモリ使用量
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # メモリ増加量が1GB以下であることをテスト
        self.assertLess(memory_increase, 1000.0)  # 1000MB = 1GB


class TestEnsembleSystemSaveLoad(unittest.TestCase):
    """EnsembleSystemモデル保存・読み込みテスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.saveload_config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
        )

        # テンポラリディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_ensemble(self):
        """アンサンブル保存・読み込みテスト"""
        # 学習済みアンサンブル作成
        ensemble = EnsembleSystem(self.saveload_config)
        X_train = np.random.randn(50, 4)
        y_train = np.random.randn(50)

        ensemble.fit(X_train, y_train)

        # 保存パス
        save_path = os.path.join(self.temp_dir, 'test_ensemble.joblib')

        # 保存実行
        save_result = ensemble.save_ensemble(save_path)
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(save_path))

        # 新しいアンサンブルインスタンス作成
        new_ensemble = EnsembleSystem(self.saveload_config)

        # 読み込み実行
        load_result = new_ensemble.load_ensemble(save_path)
        self.assertTrue(load_result)
        self.assertTrue(new_ensemble.is_trained)

        # 予測比較
        X_test = np.random.randn(10, 4)
        original_prediction = ensemble.predict(X_test)
        loaded_prediction = new_ensemble.predict(X_test)

        # 予測結果が近似していることを確認
        np.testing.assert_array_almost_equal(
            original_prediction.final_predictions,
            loaded_prediction.final_predictions,
            decimal=3
        )

    def test_save_load_error_handling(self):
        """保存・読み込みエラーハンドリングテスト"""
        ensemble = EnsembleSystem(self.saveload_config)

        # 存在しないパスからの読み込み
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.joblib')
        load_result = ensemble.load_ensemble(nonexistent_path)
        self.assertFalse(load_result)

        # 不正なパスへの保存
        invalid_path = "/invalid/path/ensemble.joblib"
        save_result = ensemble.save_ensemble(invalid_path)
        self.assertFalse(save_result)


def run_ensemble_system_tests():
    """EnsembleSystemテストスイートの実行"""
    print("=" * 60)
    print("EnsembleSystem包括的テストスイート実行")
    print("=" * 60)

    if not ENSEMBLE_AVAILABLE:
        print("❌ EnsembleSystemが利用できません。テストをスキップします。")
        return

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 基本機能テスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemBasic))

    # エッジケーステスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemEdgeCases))

    # モックテスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemMocking))

    # パフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemPerformance))

    # 保存・読み込みテスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemSaveLoad))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("テストサマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ テスト成功率: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ensemble_system_tests()
    exit(0 if success else 1)