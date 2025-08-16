#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Hyperparameter Optimizer テストスイート
Issue #856対応：改善版ハイパーパラメータ最適化システムの包括的テスト

テスト項目：
1. 設定管理機能のテスト
2. ベースラインスコア計算のテスト
3. 最適化手法選択ロジックのテスト
4. パラメータ重要度計算のテスト
5. データベース統合のテスト
6. 統合シナリオのテスト
"""

import unittest
import tempfile
import shutil
import sqlite3
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Windows環境対応
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from hyperparameter_optimizer import (
        EnhancedHyperparameterOptimizer,
        ModelType,
        PredictionTask,
        OptimizationMethod,
        OptimizationResult,
        HyperparameterSpaceConfig,
        create_enhanced_hyperparameter_optimizer
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestEnhancedHyperparameterConfigManager(unittest.TestCase):
    """改善版ハイパーパラメータ設定管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_hyperparameter_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_default_config_creation(self):
        """デフォルト設定ファイル作成テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # 設定が読み込まれること
        self.assertIsNotNone(optimizer.config)
        self.assertIsInstance(optimizer.hyperparameter_spaces, HyperparameterSpaceConfig)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_config_loading(self):
        """カスタム設定読み込みテスト"""
        custom_config = {
            'random_forest_classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            },
            'optimization_settings': {
                'random_search': {
                    'n_iter': 30,
                    'cv_folds': 5
                }
            },
            'baseline_scores': {
                'classification': {'accuracy': 0.6},
                'regression': {'r2': -0.1}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # カスタム設定が読み込まれること
        self.assertEqual(
            optimizer.hyperparameter_spaces.random_forest_classifier['n_estimators'],
            [50, 100, 200]
        )
        self.assertEqual(
            optimizer.optimization_configs[OptimizationMethod.RANDOM].n_iter,
            30
        )
        self.assertEqual(
            optimizer.baseline_configs['classification']['accuracy'],
            0.6
        )

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_hyperparameter_space_loading(self):
        """ハイパーパラメータ空間読み込みテスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # ハイパーパラメータ空間が正しく設定されること
        spaces = optimizer.hyperparameter_spaces
        self.assertIsInstance(spaces.random_forest_classifier, dict)
        self.assertIsInstance(spaces.random_forest_regressor, dict)
        self.assertIsInstance(spaces.xgboost_classifier, dict)
        self.assertIsInstance(spaces.xgboost_regressor, dict)


class TestBaselineScoreCalculation(unittest.TestCase):
    """ベースラインスコア計算テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_baseline_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_classification_baseline_score(self):
        """分類タスクベースラインスコア計算テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # テストデータ作成（不均衡データ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0] * 80 + [1] * 20)  # 80% vs 20%

        baseline_score = optimizer.get_appropriate_baseline_score(
            PredictionTask.PRICE_DIRECTION, X, y
        )

        # 最頻値予測（80%）に近い値になることを確認
        self.assertGreater(baseline_score, 0.7)
        self.assertLess(baseline_score, 0.9)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_regression_baseline_score(self):
        """回帰タスクベースラインスコア計算テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # テストデータ作成
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randn(100))

        baseline_score = optimizer.get_appropriate_baseline_score(
            PredictionTask.PRICE_REGRESSION, X, y
        )

        # 平均値予測のR2スコア（通常0.0）になることを確認
        self.assertAlmostEqual(baseline_score, 0.0, places=5)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_baseline_score_fallback(self):
        """ベースラインスコアフォールバックテスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # 不正なデータでエラーが発生した場合のフォールバック
        X = pd.DataFrame()  # 空のデータフレーム
        y = pd.Series()     # 空のシリーズ

        baseline_score_cls = optimizer.get_appropriate_baseline_score(
            PredictionTask.PRICE_DIRECTION, X, y
        )
        baseline_score_reg = optimizer.get_appropriate_baseline_score(
            PredictionTask.PRICE_REGRESSION, X, y
        )

        # 設定ファイルのフォールバック値が使用されること
        self.assertEqual(baseline_score_cls, 0.5)
        self.assertEqual(baseline_score_reg, 0.0)


class TestOptimizationMethodSelection(unittest.TestCase):
    """最適化手法選択ロジックテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_optimization_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_grid_search_creation(self):
        """Grid Search作成テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        param_space = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        config = optimizer.optimization_configs[OptimizationMethod.GRID]

        # Grid Searchオプティマイザーが作成されること
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

        search_optimizer = optimizer._create_optimizer(
            OptimizationMethod.GRID, config, model, param_space, 'accuracy'
        )

        self.assertEqual(search_optimizer.__class__.__name__, 'GridSearchCV')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('hyperparameter_optimizer.SCIPY_AVAILABLE', True)
    def test_random_search_creation(self):
        """Random Search作成テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        param_space = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
        config = optimizer.optimization_configs[OptimizationMethod.RANDOM]

        # Random Searchオプティマイザーが作成されること
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

        search_optimizer = optimizer._create_optimizer(
            OptimizationMethod.RANDOM, config, model, param_space, 'accuracy'
        )

        self.assertEqual(search_optimizer.__class__.__name__, 'RandomizedSearchCV')
        self.assertEqual(search_optimizer.n_iter, config.n_iter)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('hyperparameter_optimizer.SCIPY_AVAILABLE', False)
    def test_random_search_fallback(self):
        """Random Searchフォールバックテスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        param_space = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
        config = optimizer.optimization_configs[OptimizationMethod.RANDOM]

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

        # scipyが利用できない場合、Grid Searchにフォールバック
        search_optimizer = optimizer._create_optimizer(
            OptimizationMethod.RANDOM, config, model, param_space, 'accuracy'
        )

        self.assertEqual(search_optimizer.__class__.__name__, 'GridSearchCV')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_param_space_limitation(self):
        """パラメータ空間制限テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        large_param_space = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [3, 5, 7, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20]
        }

        limited_space = optimizer._limit_param_space_for_grid(large_param_space, 20)

        # パラメータ空間が制限されること
        total_combinations = 1
        for values in limited_space.values():
            total_combinations *= len(values)

        self.assertLessEqual(total_combinations, 20)


class TestParameterImportanceCalculation(unittest.TestCase):
    """パラメータ重要度計算テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_importance_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cv_variance_importance(self):
        """CV分散重要度計算テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # モックオプティマイザー作成
        mock_optimizer = Mock()
        mock_optimizer.cv_results_ = {
            'param_n_estimators': [100, 200, 100, 200],
            'param_max_depth': [5, 5, 10, 10],
            'mean_test_score': [0.8, 0.85, 0.82, 0.87]
        }

        importance = optimizer._calculate_cv_variance_importance(mock_optimizer)

        # パラメータ重要度が計算されること
        self.assertIn('n_estimators', importance)
        self.assertIn('max_depth', importance)
        self.assertIsInstance(importance['n_estimators'], float)
        self.assertIsInstance(importance['max_depth'], float)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_correlation_importance(self):
        """相関重要度計算テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # 数値パラメータの相関テスト
        mock_optimizer = Mock()
        mock_optimizer.cv_results_ = {
            'param_n_estimators': [50, 100, 150, 200, 250, 300],
            'mean_test_score': [0.7, 0.75, 0.8, 0.85, 0.82, 0.79]  # 相関あり
        }

        importance = optimizer._calculate_correlation_importance(mock_optimizer)

        # 相関がある場合、重要度が計算されること
        self.assertIn('n_estimators', importance)
        self.assertGreater(importance['n_estimators'], 0.0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_deviation_importance(self):
        """偏差重要度計算テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        mock_optimizer = Mock()
        mock_optimizer.cv_results_ = {
            'param_n_estimators': [50, 100, 200, 300],
            'param_max_depth': ['auto', 'sqrt', 'log2', None]
        }

        best_params = {'n_estimators': 300, 'max_depth': 'sqrt'}

        importance = optimizer._calculate_deviation_importance(mock_optimizer, best_params)

        # 偏差重要度が計算されること
        self.assertIn('n_estimators', importance)
        self.assertIn('max_depth', importance)


class TestDatabaseIntegration(unittest.TestCase):
    """データベース統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_db_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_database_initialization(self):
        """データベース初期化テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # データベースファイルが作成されること
        self.assertTrue(optimizer.db_path.exists())

        # テーブルが作成されることを確認
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.cursor()

            # enhanced_optimization_resultsテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='enhanced_optimization_results'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # enhanced_parameter_importanceテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='enhanced_parameter_importance'"
            )
            self.assertIsNotNone(cursor.fetchone())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    async def test_optimization_result_recording(self):
        """最適化結果記録テスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # テスト用最適化結果
        result = OptimizationResult(
            model_type='random_forest',
            task='price_direction',
            optimization_method='random',
            best_params={'n_estimators': 100, 'max_depth': 10},
            best_score=0.85,
            cv_scores=[0.8, 0.85, 0.87],
            optimization_time=60.0,
            improvement=10.0,
            param_importance={'n_estimators': 0.8, 'max_depth': 0.6},
            baseline_score=0.75,
            total_combinations=50
        )

        # 結果を記録
        await optimizer._record_optimization_result(result)

        # データベースに記録されることを確認
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM enhanced_optimization_results"
            )
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute(
                "SELECT COUNT(*) FROM enhanced_parameter_importance"
            )
            self.assertEqual(cursor.fetchone()[0], 2)  # 2つのパラメータ


class TestIntegrationScenarios(unittest.TestCase):
    """統合シナリオテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "integration_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_full_optimization_workflow_mock(self):
        """完全最適化ワークフローテスト（モック版）"""
        # テスト用設定ファイル作成
        test_config = {
            'random_forest_classifier': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            },
            'optimization_settings': {
                'random_search': {'n_iter': 2, 'cv_folds': 2}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)

        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        # モックデータ作成
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.choice([0, 1], 50))

        # 最適化の各ステップをテスト
        baseline_score = optimizer.get_appropriate_baseline_score(
            PredictionTask.PRICE_DIRECTION, X, y
        )
        self.assertIsInstance(baseline_score, float)

        # モデルとパラメータ空間の取得
        model, param_space, scoring = optimizer._get_model_and_params(
            ModelType.RANDOM_FOREST, PredictionTask.PRICE_DIRECTION
        )
        self.assertIsNotNone(model)
        self.assertIsInstance(param_space, dict)
        self.assertEqual(scoring, 'accuracy')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_config_file_reload(self):
        """設定ファイル再読み込みテスト"""
        # 初期設定
        initial_config = {
            'random_forest_classifier': {
                'n_estimators': [100, 200]
            },
            'optimization_settings': {
                'random_search': {'n_iter': 50}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(initial_config, f)

        optimizer = EnhancedHyperparameterOptimizer(self.config_path)
        initial_n_iter = optimizer.optimization_configs[OptimizationMethod.RANDOM].n_iter

        # 設定変更
        updated_config = {
            'random_forest_classifier': {
                'n_estimators': [50, 100, 150, 200]
            },
            'optimization_settings': {
                'random_search': {'n_iter': 30}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f)

        # 新しいインスタンスで設定が変更されることを確認
        new_optimizer = EnhancedHyperparameterOptimizer(self.config_path)
        updated_n_iter = new_optimizer.optimization_configs[OptimizationMethod.RANDOM].n_iter

        self.assertNotEqual(initial_n_iter, updated_n_iter)
        self.assertEqual(updated_n_iter, 30)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_optimization_summary(self):
        """最適化サマリーテスト"""
        optimizer = EnhancedHyperparameterOptimizer(self.config_path)

        summary = optimizer.get_optimization_summary()

        # サマリーの基本項目が含まれることを確認
        self.assertIn('config_path', summary)
        self.assertIn('available_methods', summary)
        self.assertIn('integrations', summary)
        self.assertIn('recent_optimizations', summary)
        self.assertIn('parameter_importance_stats', summary)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_factory_function(self):
        """ファクトリー関数テスト"""
        optimizer = create_enhanced_hyperparameter_optimizer(str(self.config_path))

        self.assertIsInstance(optimizer, EnhancedHyperparameterOptimizer)
        self.assertEqual(str(optimizer.config_path), str(self.config_path))


def run_enhanced_hyperparameter_optimizer_tests():
    """改善版ハイパーパラメータ最適化システムテスト実行"""
    print("=== Enhanced Hyperparameter Optimizer テスト開始 ===")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestEnhancedHyperparameterConfigManager,
        TestBaselineScoreCalculation,
        TestOptimizationMethodSelection,
        TestParameterImportanceCalculation,
        TestDatabaseIntegration,
        TestIntegrationScenarios
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"スキップ: {len(result.skipped)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Windows環境対応
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    success = run_enhanced_hyperparameter_optimizer_tests()
    sys.exit(0 if success else 1)
