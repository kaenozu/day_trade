#!/usr/bin/env python3
"""
Issue #471対応: EnsembleSystemの改良されたテストスイート

効率的で信頼性の高いテスト：
- モックとダミーデータの活用
- エッジケース・エラーハンドリング重視
- パフォーマンス・実用性重視
- プレースホルダー削除
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import time
import warnings
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

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


class TestEnsembleSystemCore(unittest.TestCase):
    """EnsembleSystemのコア機能テスト"""

    def setUp(self):
        """テスト前準備"""
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        # 一貫したランダムシード
        np.random.seed(42)

        # 最小限で実用的なテストデータ
        self.n_samples = 50  # 高速テスト用
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.sum(self.X[:, :2], axis=1) + 0.1 * np.random.randn(self.n_samples)
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # 軽量な設定
        self.config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,  # テスト高速化
            use_svr=False,
            use_lstm_transformer=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
        )

    def test_initialization_success(self):
        """正常な初期化テスト"""
        ensemble = EnsembleSystem(self.config)

        self.assertIsInstance(ensemble, EnsembleSystem)
        self.assertEqual(ensemble.config, self.config)
        self.assertFalse(ensemble.is_trained)
        self.assertTrue(len(ensemble.base_models) >= 1)

    def test_initialization_with_invalid_config(self):
        """不正な設定での初期化テスト"""
        # すべてのモデルが無効
        invalid_config = EnsembleConfig(
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False
        )

        with self.assertRaises((ValueError, RuntimeError)):
            EnsembleSystem(invalid_config)

    def test_training_basic(self):
        """基本的な学習テスト"""
        ensemble = EnsembleSystem(self.config)

        start_time = time.time()
        result = ensemble.fit(self.X, self.y, self.feature_names)
        training_time = time.time() - start_time

        # 基本的な成功条件
        self.assertIsInstance(result, dict)
        self.assertTrue(ensemble.is_trained)
        self.assertLess(training_time, 30.0)  # 30秒以内

        # 学習結果の検証
        self.assertIn('random_forest', result)
        rf_result = result['random_forest']
        self.assertEqual(rf_result.get('status'), '成功')
        self.assertIsInstance(rf_result.get('training_time'), float)

    def test_prediction_basic(self):
        """基本的な予測テスト"""
        ensemble = EnsembleSystem(self.config)
        ensemble.fit(self.X[:30], self.y[:30], self.feature_names)

        start_time = time.time()
        prediction = ensemble.predict(self.X[30:40])
        prediction_time = time.time() - start_time

        # 基本的な予測条件
        self.assertIsInstance(prediction, EnsemblePrediction)
        self.assertEqual(len(prediction.final_predictions), 10)
        self.assertLess(prediction_time, 5.0)  # 5秒以内

        # 予測値の妥当性
        predictions = prediction.final_predictions
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        self.assertFalse(any(np.isnan(predictions)))

    def test_untrained_prediction_error(self):
        """未学習状態での予測エラーテスト"""
        ensemble = EnsembleSystem(self.config)

        with self.assertRaises((RuntimeError, ValueError)):
            ensemble.predict(self.X[:10])


class TestEnsembleSystemEdgeCases(unittest.TestCase):
    """エッジケースとエラーハンドリング"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False,
        )

    def test_empty_data_handling(self):
        """空データのハンドリング"""
        ensemble = EnsembleSystem(self.config)

        empty_X = np.array([]).reshape(0, 5)
        empty_y = np.array([])

        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(empty_X, empty_y, ["f1", "f2", "f3", "f4", "f5"])

    def test_mismatched_dimensions(self):
        """次元不一致データの処理"""
        ensemble = EnsembleSystem(self.config)

        X_train = np.random.randn(20, 3)
        y_train = np.random.randn(15)  # 意図的にサイズ不一致

        with self.assertRaises((ValueError, IndexError)):
            ensemble.fit(X_train, y_train, ["f1", "f2", "f3"])

    def test_extreme_values_handling(self):
        """極値データの処理"""
        ensemble = EnsembleSystem(self.config)

        # 極値を含むデータ
        X_extreme = np.array([
            [1e6, -1e6, 0, 1, 2],
            [1e-6, 1e-6, 3, 4, 5],
            [np.inf, -np.inf, 6, 7, 8],
            [1, 2, 3, 4, 5]
        ])
        y_extreme = np.array([1, 2, 3, 4])

        # 極値があっても例外を投げない（ログ警告程度）
        try:
            result = ensemble.fit(X_extreme, y_extreme, ["f1", "f2", "f3", "f4", "f5"])
            # 何らかの結果を返すはず（警告付きでも）
            self.assertIsInstance(result, dict)
        except (ValueError, RuntimeWarning):
            # 極値によるエラーも許容（適切な例外処理）
            pass

    def test_nan_data_handling(self):
        """NaN値を含むデータの処理"""
        ensemble = EnsembleSystem(self.config)

        X_nan = np.array([
            [1, 2, np.nan, 4, 5],
            [6, np.nan, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        y_nan = np.array([1, np.nan, 3])

        with self.assertRaises((ValueError, TypeError)):
            ensemble.fit(X_nan, y_nan, ["f1", "f2", "f3", "f4", "f5"])


class TestEnsembleSystemMocking(unittest.TestCase):
    """モック使用によるユニットテスト"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

    @patch('src.day_trade.ml.base_models.random_forest_model.RandomForestModel')
    def test_random_forest_model_integration(self, mock_rf_class):
        """RandomForestModelとの統合テスト（モック使用）"""
        # モックの設定
        mock_rf_instance = Mock()
        mock_rf_instance.fit.return_value = {
            'status': '成功',
            'training_time': 1.5,
            'model_metrics': {'accuracy': 0.85}
        }
        mock_rf_instance.predict.return_value = ModelPrediction(
            predictions=np.array([1.0, 2.0, 3.0]),
            probabilities=None,
            confidence_scores=np.array([0.8, 0.9, 0.7]),
            prediction_time=0.1
        )
        mock_rf_instance.is_trained = True
        mock_rf_class.return_value = mock_rf_instance

        # テスト実行
        config = EnsembleConfig(use_random_forest=True, use_gradient_boosting=False, use_svr=False, use_lstm_transformer=False)
        ensemble = EnsembleSystem(config)

        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        # 学習テスト
        result = ensemble.fit(X, y, ["f1", "f2", "f3"])
        mock_rf_instance.fit.assert_called_once()
        self.assertIn('random_forest', result)

        # 予測テスト
        prediction = ensemble.predict(X[:5])
        mock_rf_instance.predict.assert_called_once()
        self.assertEqual(len(prediction.final_predictions), 5)

    @patch('src.day_trade.ml.dynamic_weighting_system.DynamicWeightingSystem')
    def test_dynamic_weighting_interaction(self, mock_dw_class):
        """動的重み調整システムとの連携テスト（モック使用）"""
        mock_dw_instance = Mock()
        mock_dw_instance.update_weights.return_value = {'random_forest': 0.7}
        mock_dw_instance.get_current_weights.return_value = {'random_forest': 0.7}
        mock_dw_class.return_value = mock_dw_instance

        config = EnsembleConfig(
            use_random_forest=True,
            enable_dynamic_weighting=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False
        )

        ensemble = EnsembleSystem(config)

        # 動的重み調整システムが作成されていることを確認
        self.assertIsNotNone(ensemble.dynamic_weighting_system)


class TestEnsembleSystemPerformance(unittest.TestCase):
    """パフォーマンステスト"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        # 最低限のモデルのみ使用
        self.config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_lstm_transformer=False,
        )

    def test_training_time_reasonable(self):
        """学習時間の妥当性テスト"""
        ensemble = EnsembleSystem(self.config)

        # 中規模データでのテスト
        X = np.random.randn(200, 10)
        y = np.random.randn(200)
        feature_names = [f"feature_{i}" for i in range(10)]

        start_time = time.time()
        result = ensemble.fit(X, y, feature_names)
        training_time = time.time() - start_time

        # 3分以内での完了を期待
        self.assertLess(training_time, 180.0)

        # 学習時間が記録されていることを確認
        if 'random_forest' in result:
            self.assertIn('training_time', result['random_forest'])

    def test_prediction_time_reasonable(self):
        """予測時間の妥当性テスト"""
        ensemble = EnsembleSystem(self.config)

        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_test = np.random.randn(50, 5)

        # 学習
        ensemble.fit(X_train, y_train, [f"f_{i}" for i in range(5)])

        # 予測時間測定
        start_time = time.time()
        prediction = ensemble.predict(X_test)
        prediction_time = time.time() - start_time

        # 10秒以内での完了を期待
        self.assertLess(prediction_time, 10.0)
        self.assertEqual(len(prediction.final_predictions), 50)

    def test_memory_usage_reasonable(self):
        """メモリ使用量の妥当性テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        ensemble = EnsembleSystem(self.config)
        X = np.random.randn(100, 8)
        y = np.random.randn(100)

        ensemble.fit(X, y, [f"f_{i}" for i in range(8)])
        ensemble.predict(X[:20])

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 500MB以内のメモリ増加を期待
        self.assertLess(memory_increase, 500.0)


class TestEnsembleSystemSaveLoad(unittest.TestCase):
    """保存・読み込み機能テスト"""

    def setUp(self):
        if not ENSEMBLE_AVAILABLE:
            self.skipTest("EnsembleSystem not available")

        self.config = EnsembleConfig(use_random_forest=True, use_gradient_boosting=False, use_svr=False, use_lstm_transformer=False)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """クリーンアップ"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_save_untrained_model(self):
        """未学習モデルの保存テスト"""
        ensemble = EnsembleSystem(self.config)
        save_path = Path(self.temp_dir) / "untrained_ensemble.json"

        # 未学習でも保存は可能
        result = ensemble.save_ensemble(str(save_path))
        self.assertTrue(result)
        self.assertTrue(save_path.exists())

    def test_save_and_load_trained_model(self):
        """学習済みモデルの保存・読み込みテスト"""
        # 学習
        ensemble = EnsembleSystem(self.config)
        X = np.random.randn(30, 4)
        y = np.random.randn(30)
        feature_names = ["f1", "f2", "f3", "f4"]

        ensemble.fit(X, y, feature_names)
        original_prediction = ensemble.predict(X[:5])

        # 保存
        save_path = Path(self.temp_dir) / "trained_ensemble.json"
        save_result = ensemble.save_ensemble(str(save_path))
        self.assertTrue(save_result)
        self.assertTrue(save_path.exists())

        # 読み込み
        new_ensemble = EnsembleSystem(self.config)
        load_result = new_ensemble.load_ensemble(str(save_path))
        self.assertTrue(load_result)
        self.assertTrue(new_ensemble.is_trained)

        # 予測結果の一貫性確認
        loaded_prediction = new_ensemble.predict(X[:5])
        np.testing.assert_array_almost_equal(
            original_prediction.final_predictions,
            loaded_prediction.final_predictions,
            decimal=3
        )

    def test_load_nonexistent_file(self):
        """存在しないファイルの読み込みテスト"""
        ensemble = EnsembleSystem(self.config)
        nonexistent_path = Path(self.temp_dir) / "nonexistent.json"

        result = ensemble.load_ensemble(str(nonexistent_path))
        self.assertFalse(result)
        self.assertFalse(ensemble.is_trained)


def run_enhanced_ensemble_tests():
    """改良されたEnsembleSystemテストの実行"""
    print("=" * 60)
    print("Issue #471対応: EnsembleSystem改良テストスイート実行")
    print("=" * 60)

    # テストスイートの作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemCore))
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemEdgeCases))
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemMocking))
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemPerformance))
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemSaveLoad))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("Issue #471対応: 改良テストサマリー")
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
    success = run_enhanced_ensemble_tests()
    exit(0 if success else 1)