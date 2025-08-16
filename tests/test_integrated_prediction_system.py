#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Prediction System テストスイート
Issue #855対応：統合予測システムの包括的テスト

テスト項目：
1. 統合予測システムの初期化
2. 設定ファイル管理
3. システム選択とルーティング
4. 予測戦略の実行
5. キャッシュ機能
6. エラーハンドリング
7. パフォーマンス監視
8. 統合シナリオ
"""

import unittest
import tempfile
import shutil
import sqlite3
import yaml
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Windows環境対応
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from integrated_prediction_system import (
        IntegratedPredictionSystem,
        ModelCache,
        SystemConfig,
        PredictionResult,
        SystemMetrics,
        create_integrated_prediction_system
    )
    from prediction_strategies import (
        strategy_manager,
        create_prediction_candidate,
        SingleBestStrategy,
        WeightedEnsembleStrategy
    )
    from prediction_error_handler import (
        prediction_error_handler,
        ErrorContext,
        PredictionError,
        ErrorCategory,
        ErrorSeverity,
        safe_predict
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestModelCache(unittest.TestCase):
    """モデルキャッシュテスト"""

    def setUp(self):
        """テスト前準備"""
        self.cache = ModelCache(max_size=3, ttl_minutes=1)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cache_put_and_get(self):
        """キャッシュの格納と取得テスト"""
        test_model = {"type": "test_model", "data": [1, 2, 3]}
        self.cache.put("test_key", test_model)

        retrieved = self.cache.get("test_key")
        self.assertEqual(retrieved, test_model)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cache_size_limit(self):
        """キャッシュサイズ制限テスト"""
        # キャッシュサイズ制限を超えて格納
        for i in range(5):
            self.cache.put(f"key_{i}", f"model_{i}")

        # 最大サイズ（3）まで削減されることを確認
        self.assertLessEqual(len(self.cache.cache), 3)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cache_ttl_expiration(self):
        """TTL期限切れテスト"""
        test_model = {"expired": True}
        self.cache.put("expire_test", test_model)

        # TTLを強制的に期限切れにする
        self.cache.cache["expire_test"]["created_at"] = datetime.now() - timedelta(minutes=2)

        # 期限切れで取得できないことを確認
        self.assertIsNone(self.cache.get("expire_test"))

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cache_clear(self):
        """キャッシュクリアテスト"""
        self.cache.put("test1", "model1")
        self.cache.put("test2", "model2")

        self.cache.clear()

        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(len(self.cache.access_times), 0)


class TestIntegratedPredictionSystemConfig(unittest.TestCase):
    """統合予測システム設定テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        self.db_path = self.temp_dir / "test_integration.db"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_default_config_creation(self):
        """デフォルト設定ファイル作成テスト"""
        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        # 設定ファイルが作成されること
        self.assertTrue(self.config_path.exists())

        # 基本設定が含まれること
        self.assertIn('integration', system.config)
        self.assertIn('systems', system.config)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_config_loading(self):
        """カスタム設定読み込みテスト"""
        custom_config = {
            'integration': {
                'default_system': 'advanced_system',
                'fallback_enabled': True
            },
            'systems': {
                'test_system': {
                    'name': 'Test System',
                    'module': 'test_module',
                    'class': 'TestClass',
                    'expected_accuracy': 95.0
                }
            },
            'symbol_preferences': {
                'TEST1': {'preferred_system': 'test_system'}
            },
            'cache': {
                'enabled': True,
                'max_models_in_memory': 5
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        # カスタム設定が読み込まれること
        self.assertEqual(system.config['integration']['default_system'], 'advanced_system')
        self.assertIn('test_system', system.system_configs)
        self.assertEqual(system.system_configs['test_system'].expected_accuracy, 95.0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_database_initialization(self):
        """データベース初期化テスト"""
        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        # テーブルが作成されること
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # system_metricsテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='system_metrics'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # prediction_historyテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='prediction_history'"
            )
            self.assertIsNotNone(cursor.fetchone())


class TestSystemSelection(unittest.TestCase):
    """システム選択テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"

        # テスト用設定
        test_config = {
            'integration': {
                'default_system': 'original_system',
                'performance_based_selection': False
            },
            'symbol_preferences': {
                'SYMBOL_A': {'preferred_system': 'advanced_system'},
                'SYMBOL_B': {'preferred_system': 'original_system'}
            },
            'systems': {
                'original_system': {
                    'name': 'Original System',
                    'module': 'ml_prediction_models',
                    'class': 'MLPredictionModels'
                },
                'advanced_system': {
                    'name': 'Advanced System',
                    'module': 'advanced_ml_prediction_system',
                    'class': 'AdvancedMLPredictionSystem'
                }
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_symbol_specific_system_selection(self):
        """銘柄別システム選択テスト"""
        system = IntegratedPredictionSystem(config_path=self.config_path)

        # 銘柄別設定による選択
        selected_a = system._select_optimal_system('SYMBOL_A')
        self.assertEqual(selected_a, 'advanced_system')

        selected_b = system._select_optimal_system('SYMBOL_B')
        self.assertEqual(selected_b, 'original_system')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_default_system_selection(self):
        """デフォルトシステム選択テスト"""
        system = IntegratedPredictionSystem(config_path=self.config_path)

        # 設定のない銘柄はデフォルトシステム
        selected = system._select_optimal_system('UNKNOWN_SYMBOL')
        self.assertEqual(selected, 'original_system')


class TestPredictionStrategies(unittest.TestCase):
    """予測戦略テスト"""

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_single_best_strategy(self):
        """最高システム単体戦略テスト"""
        strategy = SingleBestStrategy("confidence")

        candidates = [
            create_prediction_candidate(100.5, "system_a", 0.7, 1.0),
            create_prediction_candidate(101.2, "system_b", 0.9, 0.5),
            create_prediction_candidate(99.8, "system_c", 0.6, 1.5),
        ]

        async def run_test():
            result = await strategy.execute(candidates)
            self.assertEqual(result.system_id, "system_b")  # 最高信頼度
            self.assertEqual(result.confidence, 0.9)

        asyncio.run(run_test())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_weighted_ensemble_strategy(self):
        """重み付きアンサンブル戦略テスト"""
        strategy = WeightedEnsembleStrategy("confidence_based")

        candidates = [
            create_prediction_candidate(100.0, "system_a", 0.8, 1.0),
            create_prediction_candidate(102.0, "system_b", 0.6, 0.5),
            create_prediction_candidate(98.0, "system_c", 0.7, 1.5),
        ]

        async def run_test():
            result = await strategy.execute(candidates)
            self.assertEqual(result.system_id, "ensemble")
            # アンサンブル結果は重み付き平均になる
            self.assertIsInstance(result.prediction, float)

        asyncio.run(run_test())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_strategy_manager(self):
        """戦略マネージャーテスト"""
        self.assertIn("single_best_confidence", strategy_manager.list_strategies())
        self.assertIn("ensemble_confidence", strategy_manager.list_strategies())

        strategy = strategy_manager.get_strategy("single_best_confidence")
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, SingleBestStrategy)


class TestErrorHandling(unittest.TestCase):
    """エラーハンドリングテスト"""

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_error_context_creation(self):
        """エラーコンテキスト作成テスト"""
        from prediction_error_handler import create_error_context

        context = create_error_context(
            symbol="TEST",
            system_id="test_system",
            operation="test_operation"
        )

        self.assertEqual(context.symbol, "TEST")
        self.assertEqual(context.system_id, "test_system")
        self.assertEqual(context.operation, "test_operation")

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_prediction_error_creation(self):
        """予測エラー作成テスト"""
        context = ErrorContext(symbol="TEST", system_id="test_system")

        error = PredictionError(
            error_id="test_error",
            category=ErrorCategory.PREDICTION_ERROR,
            severity=ErrorSeverity.WARNING,
            message="Test error message",
            context=context
        )

        # 辞書変換テスト
        error_dict = error.to_dict()
        self.assertEqual(error_dict['error_id'], "test_error")
        self.assertEqual(error_dict['category'], "prediction_error")
        self.assertEqual(error_dict['severity'], "warning")

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_safe_predict_function(self):
        """安全な予測実行テスト"""
        async def failing_operation():
            raise ValueError("Test error")

        async def run_test():
            context = ErrorContext(symbol="TEST", system_id="test_system")
            result, error = await safe_predict(failing_operation, context)

            self.assertIsNone(result)
            self.assertIsNotNone(error)
            self.assertEqual(error.context.symbol, "TEST")

        asyncio.run(run_test())


class TestIntegrationScenarios(unittest.TestCase):
    """統合シナリオテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "integration_config.yaml"
        self.db_path = self.temp_dir / "integration.db"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_factory_function(self):
        """ファクトリー関数テスト"""
        system = create_integrated_prediction_system(str(self.config_path))

        self.assertIsInstance(system, IntegratedPredictionSystem)
        self.assertIsNotNone(system.config)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('integrated_prediction_system.IntegratedPredictionSystem._get_system_instance')
    async def test_single_prediction_mock(self, mock_get_instance):
        """単一システム予測テスト（モック使用）"""
        # モックシステムインスタンス
        mock_system = AsyncMock()
        mock_system.predict.return_value = {"prediction": 100.5, "confidence": 0.85}
        mock_get_instance.return_value = mock_system

        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        # 単一システム予測
        result = await system.predict("TEST_SYMBOL", use_ensemble=False)

        self.assertIsNotNone(result.prediction)
        self.assertGreater(result.confidence, 0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('integrated_prediction_system.IntegratedPredictionSystem._get_system_instance')
    async def test_ensemble_prediction_mock(self, mock_get_instance):
        """アンサンブル予測テスト（モック使用）"""
        # モックシステムインスタンス
        mock_system = AsyncMock()
        mock_system.predict.return_value = {"prediction": 100.0, "confidence": 0.8}
        mock_get_instance.return_value = mock_system

        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        # アンサンブル予測
        result = await system.predict(
            "TEST_SYMBOL",
            use_ensemble=True,
            strategy_name="ensemble_confidence"
        )

        self.assertIsNotNone(result.prediction)
        self.assertIn("ensemble", result.system_used)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_system_status_summary(self):
        """システム状態サマリーテスト"""
        system = IntegratedPredictionSystem(
            config_path=self.config_path,
            db_path=self.db_path
        )

        status = system.get_system_status()

        # 基本項目が含まれること
        self.assertIn('timestamp', status)
        self.assertIn('config_path', status)
        self.assertIn('systems', status)
        self.assertIn('cache_status', status)
        self.assertIn('integration_settings', status)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_configuration_reload(self):
        """設定再読み込みテスト"""
        # 初期設定
        initial_config = {
            'integration': {'default_system': 'original_system'},
            'systems': {}
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(initial_config, f)

        system = IntegratedPredictionSystem(config_path=self.config_path)
        initial_default = system.config['integration']['default_system']

        # 設定変更
        updated_config = {
            'integration': {'default_system': 'advanced_system'},
            'systems': {}
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f)

        # 設定再読み込み
        system.reload_configuration()

        # 変更が反映されること
        self.assertNotEqual(initial_default, system.config['integration']['default_system'])
        self.assertEqual(system.config['integration']['default_system'], 'advanced_system')


def run_integrated_prediction_system_tests():
    """統合予測システムテスト実行"""
    print("=== Integrated Prediction System テスト開始 ===")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestModelCache,
        TestIntegratedPredictionSystemConfig,
        TestSystemSelection,
        TestPredictionStrategies,
        TestErrorHandling,
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

    success = run_integrated_prediction_system_tests()
    sys.exit(0 if success else 1)