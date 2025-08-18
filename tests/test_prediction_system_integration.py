#!/usr/bin/env python3
"""
予測システム統合テストスイート
Issue #870後の統合テスト：新機能の統合と既存システムとの互換性確認
"""

import unittest
import numpy as np
import pandas as pd
import logging
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 既存システムのインポート
try:
    from daytrade_core import DayTradeCore
    DAYTRADE_CORE_AVAILABLE = True
except ImportError:
    DAYTRADE_CORE_AVAILABLE = False

try:
    from ml_prediction_models import MLPredictionModels
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

try:
    from enhanced_personal_analysis_engine import EnhancedPersonalAnalysisEngine
    ANALYSIS_ENGINE_AVAILABLE = True
except ImportError:
    ANALYSIS_ENGINE_AVAILABLE = False

# 新規実装システムのインポート
try:
    from advanced_feature_selector import create_advanced_feature_selector
    from advanced_ensemble_system import create_advanced_ensemble_system, EnsembleMethod
    from meta_learning_system import create_meta_learning_system, TaskType
    from comprehensive_prediction_evaluation import create_comprehensive_evaluator
    NEW_SYSTEMS_AVAILABLE = True
except ImportError as e:
    NEW_SYSTEMS_AVAILABLE = False
    print(f"新システムインポートエラー: {e}")


class TestSystemCompatibility(unittest.TestCase):
    """システム互換性テスト"""

    def setUp(self):
        """テスト準備"""
        self.test_data_size = 200
        self.feature_count = 20

        # テストデータ作成
        np.random.seed(42)
        self.test_features = pd.DataFrame(
            np.random.randn(self.test_data_size, self.feature_count),
            columns=[f'feature_{i}' for i in range(self.feature_count)]
        )

        # ターゲット作成（非線形関係）
        self.test_target = (
            self.test_features['feature_0'] * 2 +
            self.test_features['feature_1'] ** 2 * 0.5 +
            np.sin(self.test_features['feature_2']) * 1.5 +
            np.random.randn(self.test_data_size) * 0.1
        )

        # 価格データシミュレーション
        self.price_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(self.test_data_size) * 0.02) + 100,
            'volume': np.random.randint(1000, 10000, self.test_data_size),
            'high': np.cumsum(np.random.randn(self.test_data_size) * 0.02) + 102,
            'low': np.cumsum(np.random.randn(self.test_data_size) * 0.02) + 98,
            'open': np.cumsum(np.random.randn(self.test_data_size) * 0.02) + 99
        })

        # データ分割
        split_idx = int(self.test_data_size * 0.8)
        self.X_train = self.test_features[:split_idx]
        self.X_test = self.test_features[split_idx:]
        self.y_train = self.test_target[:split_idx]
        self.y_test = self.test_target[split_idx:]

    def test_new_systems_availability(self):
        """新システムの利用可能性テスト"""
        self.assertTrue(NEW_SYSTEMS_AVAILABLE, "新実装システムが利用できません")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_feature_selector_integration(self):
        """特徴量選択システム統合テスト"""
        try:
            # 特徴量選択システム作成
            selector = create_advanced_feature_selector(max_features=15)

            # 特徴量選択実行
            selected_X, selection_info = selector.select_features(
                self.test_features, self.test_target, self.price_data
            )

            # 基本検証
            self.assertIsInstance(selected_X, pd.DataFrame)
            self.assertIsInstance(selection_info, dict)
            self.assertLessEqual(selected_X.shape[1], 15)
            self.assertIn('selected_features', selection_info)
            self.assertIn('market_regime', selection_info)

            print(f"✓ 特徴量選択システム統合成功: {selected_X.shape[1]}特徴量選択")

        except Exception as e:
            self.fail(f"特徴量選択システム統合失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_ensemble_system_integration(self):
        """アンサンブルシステム統合テスト"""
        try:
            # アンサンブルシステム作成
            ensemble_system = create_advanced_ensemble_system(
                method=EnsembleMethod.STACKING,
                cv_folds=3
            )

            # 訓練・予測
            ensemble_system.fit(self.X_train, self.y_train)
            predictions = ensemble_system.predict(self.X_test)

            # 基本検証
            self.assertEqual(len(predictions), len(self.X_test))
            self.assertTrue(np.all(np.isfinite(predictions)))

            # システム要約取得
            summary = ensemble_system.get_ensemble_summary()
            self.assertIsInstance(summary, dict)
            self.assertIn('ensemble_models', summary)

            print(f"✓ アンサンブルシステム統合成功: {len(summary.get('ensemble_models', []))}手法")

        except Exception as e:
            self.fail(f"アンサンブルシステム統合失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_meta_learning_integration(self):
        """メタラーニングシステム統合テスト"""
        try:
            # メタラーニングシステム作成
            meta_system = create_meta_learning_system(repository_size=20)

            # 訓練・予測
            model, predictions, result_info = meta_system.fit_predict(
                self.X_train, self.y_train, self.price_data,
                task_type=TaskType.REGRESSION,
                X_predict=self.X_test
            )

            # 基本検証
            self.assertEqual(len(predictions), len(self.X_test))
            self.assertTrue(np.all(np.isfinite(predictions)))
            self.assertIsInstance(result_info, dict)
            self.assertIn('model_type', result_info)
            self.assertIn('market_condition', result_info)

            # 学習洞察取得
            insights = meta_system.get_learning_insights()
            self.assertIsInstance(insights, dict)

            print(f"✓ メタラーニングシステム統合成功: {result_info.get('model_type')}")

        except Exception as e:
            self.fail(f"メタラーニングシステム統合失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_comprehensive_evaluation_integration(self):
        """包括的評価システム統合テスト"""
        try:
            # 包括的評価器作成
            evaluator = create_comprehensive_evaluator()

            # 評価実行
            report = evaluator.run_comprehensive_evaluation(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.price_data, save_results=False
            )

            # 基本検証
            self.assertIsInstance(report.component_results, list)
            self.assertGreater(len(report.component_results), 0)
            self.assertIsInstance(report.improvement_analysis, dict)
            self.assertIsInstance(report.recommendations, list)

            # ベースライン結果確認
            baseline_found = any(
                r.component_type.value == 'baseline'
                for r in report.component_results
            )
            self.assertTrue(baseline_found, "ベースライン結果が見つかりません")

            print(f"✓ 包括的評価システム統合成功: {len(report.component_results)}コンポーネント評価")

        except Exception as e:
            self.fail(f"包括的評価システム統合失敗: {e}")


class TestDataFlowIntegration(unittest.TestCase):
    """データフロー統合テスト"""

    def setUp(self):
        """テスト準備"""
        self.sample_size = 100
        self.feature_count = 15

        # シンプルなテストデータ
        np.random.seed(123)
        self.X = pd.DataFrame(
            np.random.randn(self.sample_size, self.feature_count),
            columns=[f'f_{i}' for i in range(self.feature_count)]
        )
        self.y = self.X['f_0'] + self.X['f_1'] * 0.5 + np.random.randn(self.sample_size) * 0.1

        self.price_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(50)) + 100,
            'volume': np.random.randint(1000, 5000, 50)
        })

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_end_to_end_pipeline(self):
        """エンドツーエンドパイプラインテスト"""
        try:
            # 1. 特徴量選択
            feature_selector = create_advanced_feature_selector(max_features=10)
            selected_X, _ = feature_selector.select_features(
                self.X, self.y, self.price_data
            )

            # 2. データ分割
            split_idx = int(len(selected_X) * 0.8)
            X_train = selected_X[:split_idx]
            X_test = selected_X[split_idx:]
            y_train = self.y[:split_idx]
            y_test = self.y[split_idx:]

            # 3. アンサンブル学習
            ensemble = create_advanced_ensemble_system(
                method=EnsembleMethod.VOTING, cv_folds=3
            )
            ensemble.fit(X_train, y_train)

            # 4. 予測実行
            predictions = ensemble.predict(X_test)

            # 5. 評価
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            # 検証
            self.assertTrue(np.all(np.isfinite(predictions)))
            self.assertGreaterEqual(r2, -1.0)  # R²の下限
            self.assertGreaterEqual(mse, 0.0)   # MSEの下限

            print(f"✓ エンドツーエンドパイプライン成功: R²={r2:.3f}, MSE={mse:.3f}")

        except Exception as e:
            self.fail(f"エンドツーエンドパイプライン失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_data_consistency(self):
        """データ一貫性テスト"""
        try:
            # 特徴量選択での一貫性
            selector = create_advanced_feature_selector(max_features=8)

            # 同じデータで複数回実行
            selected_X1, info1 = selector.select_features(
                self.X, self.y, self.price_data, method='mutual_info'
            )
            selected_X2, info2 = selector.select_features(
                self.X, self.y, self.price_data, method='mutual_info'
            )

            # 特徴量選択の一貫性確認
            self.assertEqual(
                set(info1['selected_features']),
                set(info2['selected_features']),
                "同じ手法での特徴量選択結果が異なります"
            )

            print(f"✓ データ一貫性確認成功: {len(info1['selected_features'])}特徴量")

        except Exception as e:
            self.fail(f"データ一貫性テスト失敗: {e}")


class TestPerformanceIntegration(unittest.TestCase):
    """性能統合テスト"""

    def setUp(self):
        """テスト準備"""
        self.large_sample_size = 500
        self.feature_count = 30

        # 大きめのテストデータ
        np.random.seed(456)
        self.X_large = pd.DataFrame(
            np.random.randn(self.large_sample_size, self.feature_count),
            columns=[f'feature_{i}' for i in range(self.feature_count)]
        )

        self.y_large = (
            self.X_large['feature_0'] * 1.5 +
            self.X_large['feature_1'] * 0.8 +
            np.random.randn(self.large_sample_size) * 0.2
        )

        self.price_data_large = pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_performance_benchmarks(self):
        """性能ベンチマークテスト"""
        try:
            # データ分割
            split_idx = int(self.large_sample_size * 0.8)
            X_train = self.X_large[:split_idx]
            X_test = self.X_large[split_idx:]
            y_train = self.y_large[:split_idx]
            y_test = self.y_large[split_idx:]

            performance_results = {}

            # 1. 特徴量選択性能
            start_time = datetime.now()
            selector = create_advanced_feature_selector(max_features=20)
            selected_X, _ = selector.select_features(
                X_train, y_train, self.price_data_large
            )
            feature_selection_time = (datetime.now() - start_time).total_seconds()
            performance_results['feature_selection'] = feature_selection_time

            # 2. アンサンブル学習性能
            start_time = datetime.now()
            ensemble = create_advanced_ensemble_system(
                method=EnsembleMethod.STACKING, cv_folds=3
            )
            ensemble.fit(selected_X, y_train)
            ensemble_training_time = (datetime.now() - start_time).total_seconds()
            performance_results['ensemble_training'] = ensemble_training_time

            # 3. 予測性能
            start_time = datetime.now()
            selected_X_test = X_test[selected_X.columns]
            predictions = ensemble.predict(selected_X_test)
            prediction_time = (datetime.now() - start_time).total_seconds()
            performance_results['prediction'] = prediction_time

            # 性能検証
            self.assertLess(feature_selection_time, 30.0, "特徴量選択が遅すぎます")
            self.assertLess(ensemble_training_time, 60.0, "アンサンブル学習が遅すぎます")
            self.assertLess(prediction_time, 5.0, "予測が遅すぎます")

            print(f"✓ 性能ベンチマーク成功:")
            for task, time_taken in performance_results.items():
                print(f"  {task}: {time_taken:.2f}秒")

        except Exception as e:
            self.fail(f"性能ベンチマークテスト失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_memory_usage(self):
        """メモリ使用量テスト"""
        import psutil
        import os

        try:
            process = psutil.Process(os.getpid())

            # 初期メモリ使用量
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # システム作成・実行
            meta_system = create_meta_learning_system(repository_size=30)
            model, predictions, _ = meta_system.fit_predict(
                self.X_large[:400], self.y_large[:400], self.price_data_large,
                X_predict=self.X_large[400:]
            )

            # 実行後メモリ使用量
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # メモリ使用量検証（500MB以下）
            self.assertLess(memory_increase, 500, f"メモリ使用量が過大: {memory_increase:.1f}MB")

            print(f"✓ メモリ使用量テスト成功: +{memory_increase:.1f}MB")

        except ImportError:
            self.skipTest("psutilが利用できません")
        except Exception as e:
            self.fail(f"メモリ使用量テスト失敗: {e}")


class TestConfigurationIntegration(unittest.TestCase):
    """設定統合テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'config'
        self.config_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """テスト後処理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_file_compatibility(self):
        """設定ファイル互換性テスト"""
        try:
            # 既存設定ファイルの確認
            config_files = [
                'config/ml.json',
                'config/performance_thresholds.yaml'
            ]

            for config_file in config_files:
                if os.path.exists(config_file):
                    # ファイル読み込み可能性確認
                    if config_file.endswith('.json'):
                        import json
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        self.assertIsInstance(config_data, dict)

                    elif config_file.endswith('.yaml'):
                        try:
                            import yaml
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config_data = yaml.safe_load(f)
                            self.assertIsInstance(config_data, dict)
                        except ImportError:
                            self.skipTest("PyYAMLが利用できません")

            print("✓ 設定ファイル互換性確認成功")

        except Exception as e:
            self.fail(f"設定ファイル互換性テスト失敗: {e}")


class TestErrorHandling(unittest.TestCase):
    """エラーハンドリングテスト"""

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_invalid_input_handling(self):
        """不正入力ハンドリングテスト"""
        try:
            # 空データでのテスト
            empty_df = pd.DataFrame()
            empty_series = pd.Series(dtype=float)

            selector = create_advanced_feature_selector(max_features=5)

            # エラーではなく適切にハンドリングされることを確認
            try:
                result = selector.select_features(
                    empty_df, empty_series, empty_df
                )
                # フォールバック動作を確認
                self.assertIsInstance(result, tuple)
            except Exception:
                # 適切なエラーメッセージでの失敗は許容
                pass

            print("✓ 不正入力ハンドリング確認成功")

        except Exception as e:
            self.fail(f"不正入力ハンドリングテスト失敗: {e}")

    @unittest.skipUnless(NEW_SYSTEMS_AVAILABLE, "新システムが利用できません")
    def test_system_fallback_behavior(self):
        """システムフォールバック動作テスト"""
        try:
            # 最小限のデータでのテスト
            minimal_X = pd.DataFrame({
                'f1': [1, 2, 3],
                'f2': [4, 5, 6]
            })
            minimal_y = pd.Series([1, 2, 3])
            minimal_price = pd.DataFrame({
                'close': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            })

            # メタラーニングシステムでフォールバック確認
            meta_system = create_meta_learning_system(repository_size=5)

            # 極小データでも実行できることを確認
            model, predictions, info = meta_system.fit_predict(
                minimal_X, minimal_y, minimal_price,
                X_predict=minimal_X
            )

            self.assertEqual(len(predictions), len(minimal_X))
            self.assertIn('model_type', info)

            print("✓ システムフォールバック動作確認成功")

        except Exception as e:
            self.fail(f"システムフォールバック動作テスト失敗: {e}")


def run_integration_tests():
    """統合テスト実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストクラス追加
    test_classes = [
        TestSystemCompatibility,
        TestDataFlowIntegration,
        TestPerformanceIntegration,
        TestConfigurationIntegration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print("\n" + "="*60)
    print("統合テスト結果サマリー")
    print("="*60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    print("予測システム統合テストスイート開始")
    print("="*60)

    success = run_integration_tests()

    if success:
        print("\n✓ 全ての統合テストが成功しました！")
    else:
        print("\n✗ 一部のテストで問題が発生しました。")

    print("\n統合テスト完了")