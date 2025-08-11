#!/usr/bin/env python3
"""
高速ML モデルテスト
Issue #375: 重い処理のモック化によるテスト速度改善

従来の重い機械学習処理を高速モックに置き換えたテストスイート
"""

import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 強化モックシステム
from .fixtures.performance_mocks_enhanced import (
    create_fast_test_environment,
    measure_mock_performance,
)

try:
    from src.day_trade.analysis.ml_models import MLModelManager, ModelConfig

    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False


class TestFastMLModels(unittest.TestCase):
    """高速ML モデルテスト"""

    def setUp(self):
        """高速テストセットアップ"""
        self.start_time = time.perf_counter()

        # 高速モック環境
        self.mock_env = create_fast_test_environment(
            include_ml=True,
            max_execution_time_ms=5.0,  # 5ms以下に制限
        )

        # 小規模テストデータ
        self.X_small = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        self.y_small = pd.Series(np.random.randn(100))

        # 中規模テストデータ
        self.X_medium = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.randn(1000),
            }
        )
        self.y_medium = pd.Series(np.random.randn(1000))

    def tearDown(self):
        """実行時間測定"""
        execution_time = (time.perf_counter() - self.start_time) * 1000
        print(f"    テスト実行時間: {execution_time:.2f}ms")

    def test_fast_model_training(self):
        """高速モデル訓練テスト"""
        ml_manager = self.mock_env["ml_manager"]

        # 小規模データでの訓練
        result = ml_manager.train_model("fast_model", self.X_small, self.y_small)

        self.assertIsNotNone(result)
        self.assertEqual(result["model_name"], "fast_model")
        self.assertEqual(result["training_samples"], 100)
        self.assertLess(result["training_time_ms"], 10.0)
        self.assertGreater(result["accuracy"], 0.5)

    def test_fast_model_prediction(self):
        """高速モデル予測テスト"""
        ml_manager = self.mock_env["ml_manager"]

        # 訓練（高速）
        ml_manager.train_model("pred_model", self.X_small, self.y_small)

        # 予測（高速）
        predictions = ml_manager.predict("pred_model", self.X_medium)

        self.assertEqual(len(predictions), 1000)
        self.assertTrue(isinstance(predictions, np.ndarray))
        # 予測値の妥当性チェック（-50%〜+50%の範囲）
        self.assertTrue(np.all(predictions > -0.5))
        self.assertTrue(np.all(predictions < 0.5))

    def test_fast_batch_prediction(self):
        """高速バッチ予測テスト"""
        ml_manager = self.mock_env["ml_manager"]

        models = ["model_1", "model_2", "model_3"]

        # バッチ予測（高速）
        results = ml_manager.batch_predict(models, self.X_small)

        self.assertEqual(len(results), 3)
        for model_name in models:
            self.assertIn(model_name, results)
            self.assertEqual(len(results[model_name]), 100)

    def test_fast_model_persistence(self):
        """高速モデル保存・読み込みテスト"""
        ml_manager = self.mock_env["ml_manager"]

        model_name = "persistent_model"
        model_path = "/tmp/test_model.pkl"

        # モデル保存（高速）
        save_result = ml_manager.save_model(model_name, model_path)

        self.assertEqual(save_result["status"], "saved")
        self.assertEqual(save_result["path"], model_path)
        self.assertGreater(save_result["size_mb"], 0)

        # モデル読み込み（高速）
        load_result = ml_manager.load_model(model_name, model_path)

        self.assertEqual(load_result["status"], "loaded")
        self.assertEqual(load_result["model_name"], model_name)
        self.assertGreater(load_result["accuracy"], 0.5)

    def test_feature_engineering_performance(self):
        """高速特徴量エンジニアリングテスト"""
        feature_manager = self.mock_env["feature_manager"]

        # 価格データ風のテストデータ
        price_data = pd.DataFrame(
            {
                "Close": 100 + np.random.randn(252),  # 1年分の日次データ
                "Volume": np.random.randint(1000000, 10000000, 252),
                "High": 102 + np.random.randn(252),
                "Low": 98 + np.random.randn(252),
            }
        )

        # 特徴量生成（高速）
        features = feature_manager.generate_features(price_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 252)
        self.assertIn("sma_20", features.columns)
        self.assertIn("rsi", features.columns)
        self.assertIn("macd", features.columns)

    def test_technical_indicators_calculation(self):
        """高速テクニカル指標計算テスト"""
        feature_manager = self.mock_env["feature_manager"]

        price_data = pd.DataFrame({"Close": 100 + np.random.randn(100)})

        indicators = ["sma_20", "rsi", "macd", "bollinger_bands"]

        # 指標計算（高速）
        results = feature_manager.calculate_indicators(price_data, indicators)

        self.assertEqual(len(results), 4)
        for indicator in indicators:
            self.assertIn(indicator, results)
            self.assertEqual(len(results[indicator]), 100)

    def test_performance_benchmarking(self):
        """性能ベンチマークテスト"""
        ml_manager = self.mock_env["ml_manager"]

        # 各操作の性能測定
        performance_results = {}

        # 訓練性能
        train_perf = measure_mock_performance(
            ml_manager.train_model, "bench_model", self.X_medium, self.y_medium
        )
        performance_results["training"] = train_perf

        # 予測性能
        pred_perf = measure_mock_performance(
            ml_manager.predict, "bench_model", self.X_medium
        )
        performance_results["prediction"] = pred_perf

        # バッチ予測性能
        batch_perf = measure_mock_performance(
            ml_manager.batch_predict, ["model_1", "model_2", "model_3"], self.X_small
        )
        performance_results["batch_prediction"] = batch_perf

        # 性能アサーション（全操作が10ms以下）
        for operation, perf in performance_results.items():
            self.assertLess(
                perf.execution_time_ms,
                10.0,
                f"{operation}の実行時間が制限を超過: {perf.execution_time_ms:.2f}ms",
            )
            self.assertEqual(perf.success_rate, 1.0, f"{operation}が失敗")

        print("    性能サマリー:")
        for operation, perf in performance_results.items():
            print(f"      {operation}: {perf.execution_time_ms:.2f}ms")

    @unittest.skipUnless(ML_MODELS_AVAILABLE, "ML models not available")
    def test_integration_with_real_ml_models(self):
        """実MLモデルとの統合テスト（限定的）"""
        # 実際のMLModelManagerを使用するが、小規模データのみ
        models_dir = Path("test_models_fast")
        models_dir.mkdir(exist_ok=True)

        try:
            manager = MLModelManager(models_dir=str(models_dir))
            config = ModelConfig(model_type="linear", task_type="regression")

            # 最小要件（50サンプル）を満たすデータでテスト
            tiny_X = self.X_small.iloc[:50]  # 50サンプル
            tiny_y = self.y_small.iloc[:50]

            # 高速実行を確保するため小規模データのみ
            start_time = time.perf_counter()

            manager.create_model("integration_test", config)
            manager.train_model("integration_test", tiny_X, tiny_y)
            predictions = manager.predict("integration_test", tiny_X)

            execution_time = (time.perf_counter() - start_time) * 1000

            self.assertEqual(len(predictions), 50)
            self.assertLess(execution_time, 1000)  # 1秒以下

        finally:
            # クリーンアップ
            for f in models_dir.glob("*.joblib"):
                f.unlink()
            models_dir.rmdir()


class TestMLModelPerformanceComparison(unittest.TestCase):
    """ML モデル性能比較テスト"""

    def test_mock_vs_real_performance_simulation(self):
        """モック vs 実処理の性能シミュレーション"""

        # モック処理の測定
        mock_env = create_fast_test_environment(include_ml=True)
        ml_mock = mock_env["ml_manager"]

        test_data = pd.DataFrame(np.random.randn(1000, 10))
        test_target = pd.Series(np.random.randn(1000))

        # モック性能測定
        start_time = time.perf_counter()
        ml_mock.train_model("test", test_data, test_target)
        predictions = ml_mock.predict("test", test_data)
        mock_time = (time.perf_counter() - start_time) * 1000

        # 実処理シミュレーション（重い処理を模擬）
        start_time = time.perf_counter()

        # 実際の機械学習処理を模擬（重い計算）
        for _ in range(100):  # 重い計算をシミュレート
            _ = np.linalg.svd(np.random.randn(10, 10))

        simulated_real_time = (time.perf_counter() - start_time) * 1000

        # 改善倍率計算
        speedup_factor = simulated_real_time / mock_time if mock_time > 0 else 1

        print(f"    モック実行時間: {mock_time:.2f}ms")
        print(f"    実処理シミュレーション時間: {simulated_real_time:.2f}ms")
        print(f"    高速化倍率: {speedup_factor:.1f}x")

        # モックが大幅に高速であることを確認
        self.assertLess(mock_time, 50)  # 50ms以下
        self.assertGreater(speedup_factor, 10)  # 10倍以上高速


def run_fast_ml_tests():
    """高速MLテスト実行"""
    print("=== Issue #375 高速MLモデルテスト実行 ===")

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 高速テストを追加
    suite.addTests(loader.loadTestsFromTestCase(TestFastMLModels))
    suite.addTests(loader.loadTestsFromTestCase(TestMLModelPerformanceComparison))

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
    result = run_fast_ml_tests()

    print("\n=== Issue #375 MLモデルテスト高速化完了 ===")
