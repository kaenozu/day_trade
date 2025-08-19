#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Model Performance Monitor テストスイート
Issue #857対応：改善版モデル性能監視システムの包括的テスト

テスト項目：
1. 設定管理機能のテスト
2. 動的監視対象銘柄管理のテスト
3. 段階的再学習トリガーのテスト
4. 性能メトリクス記録のテスト
5. アラートシステムのテスト
6. 統合シナリオのテスト
"""

import unittest
import tempfile
import shutil
import sqlite3
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Windows環境対応
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from src.day_trade.monitoring.model_performance_monitor import (
        EnhancedModelPerformanceMonitor,
        EnhancedPerformanceConfigManager,
        DynamicSymbolManager,
        GranularRetrainingManager,
        PerformanceMetrics,
        RetrainingResult,
        PerformanceAlert,
        AlertLevel,
        RetrainingScope,
        create_enhanced_performance_monitor
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestEnhancedPerformanceConfigManager(unittest.TestCase):
    """改善版性能設定管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_performance_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_default_config_creation(self):
        """デフォルト設定ファイル作成テスト"""
        config_manager = EnhancedPerformanceConfigManager(self.config_path)

        # 設定ファイルが作成されること
        self.assertTrue(self.config_path.exists())

        # 基本設定が含まれること
        self.assertIn('performance_thresholds', config_manager.config)
        self.assertIn('monitoring', config_manager.config)
        self.assertIn('retraining', config_manager.config)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_config_loading(self):
        """カスタム設定読み込みテスト"""
        custom_config = {
            'performance_thresholds': {
                'accuracy': 95.0,
                'symbol_specific': {
                    'high_volume_stocks': {'accuracy': 92.0},
                    'mid_volume_stocks': {'accuracy': 88.0}
                }
            },
            'monitoring': {
                'default_symbols': ['TEST1', 'TEST2'],
                'dynamic_monitoring': {'enabled': True, 'max_symbols': 15},
                'validation_hours': 72
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 12,
                'global_threshold': 85.0
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        config_manager = EnhancedPerformanceConfigManager(self.config_path)

        # カスタム設定が読み込まれること
        self.assertEqual(config_manager.get_threshold('accuracy'), 95.0)
        self.assertEqual(
            config_manager.get_threshold('accuracy', '7203'), 92.0
        )  # high_volume
        self.assertEqual(
            config_manager.get_monitoring_config()['default_symbols'],
            ['TEST1', 'TEST2']
        )

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_symbol_categorization(self):
        """銘柄分類テスト"""
        config_manager = EnhancedPerformanceConfigManager(self.config_path)

        # 高ボリューム銘柄
        self.assertEqual(
            config_manager._categorize_symbol('7203'), 'high_volume_stocks'
        )
        self.assertEqual(
            config_manager._categorize_symbol('8306'), 'high_volume_stocks'
        )

        # 中ボリューム銘柄
        self.assertEqual(
            config_manager._categorize_symbol('OTHER'), 'mid_volume_stocks'
        )


class TestDynamicSymbolManager(unittest.TestCase):
    """動的監視対象銘柄管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_symbol_config.yaml"
        self.config_manager = EnhancedPerformanceConfigManager(self.config_path)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_basic_symbol_management(self):
        """基本的な銘柄管理テスト"""
        symbol_manager = DynamicSymbolManager(self.config_manager)

        # デフォルト監視銘柄の取得
        symbols = symbol_manager.get_monitoring_symbols()
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_dynamic_monitoring_disabled(self):
        """動的監視無効時のテスト"""
        # 動的監視を無効にした設定
        config = {
            'monitoring': {
                'default_symbols': ['TEST1', 'TEST2', 'TEST3'],
                'dynamic_monitoring': {'enabled': False}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        config_manager = EnhancedPerformanceConfigManager(self.config_path)
        symbol_manager = DynamicSymbolManager(config_manager)

        symbols = symbol_manager.get_monitoring_symbols()
        self.assertEqual(set(symbols), {'TEST1', 'TEST2', 'TEST3'})

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.SYMBOL_SELECTOR_AVAILABLE', True)
    def test_dynamic_monitoring_with_mock_selector(self):
        """モック版SymbolSelectorでの動的監視テスト"""
        # 動的監視を有効にした設定
        config = {
            'monitoring': {
                'default_symbols': ['BASE1', 'BASE2'],
                'dynamic_monitoring': {'enabled': True, 'max_symbols': 8}
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        config_manager = EnhancedPerformanceConfigManager(self.config_path)
        symbol_manager = DynamicSymbolManager(config_manager)

        # モックのSymbolSelectorを設定
        mock_selector = Mock()
        mock_selector.get_recommended_symbols.return_value = ['DYN1', 'DYN2', 'DYN3']
        mock_selector.get_high_volatility_symbols.return_value = ['VOL1', 'VOL2']
        symbol_manager.symbol_selector = mock_selector

        symbols = symbol_manager.get_monitoring_symbols()

        # ベース銘柄と動的銘柄が含まれることを確認
        self.assertIn('BASE1', symbols)
        self.assertIn('BASE2', symbols)
        self.assertTrue(any(s.startswith('DYN') for s in symbols))

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_update_interval_check(self):
        """更新間隔チェックテスト"""
        symbol_manager = DynamicSymbolManager(self.config_manager)

        # 初回は更新が必要
        self.assertTrue(symbol_manager.should_update_symbols())

        # 更新後は更新不要
        symbol_manager.last_update = datetime.now()
        self.assertFalse(symbol_manager.should_update_symbols())

        # 十分時間が経過したら更新必要
        symbol_manager.last_update = datetime.now() - timedelta(hours=25)
        self.assertTrue(symbol_manager.should_update_symbols())


class TestGranularRetrainingManager(unittest.TestCase):
    """段階的再学習管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_retraining_config.yaml"
        self.config_manager = EnhancedPerformanceConfigManager(self.config_path)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_retraining_scope_determination(self):
        """再学習スコープ決定テスト"""
        retraining_manager = GranularRetrainingManager(self.config_manager)

        # 高性能（再学習不要）
        high_performance = {
            'SYM1': PerformanceMetrics(symbol='SYM1', accuracy=95.0),
            'SYM2': PerformanceMetrics(symbol='SYM2', accuracy=92.0),
            'SYM3': PerformanceMetrics(symbol='SYM3', accuracy=91.0)
        }
        scope = retraining_manager.determine_retraining_scope(high_performance)
        self.assertEqual(scope, RetrainingScope.NONE)

        # 全体的に低性能（グローバル再学習）
        low_performance = {
            'SYM1': PerformanceMetrics(symbol='SYM1', accuracy=75.0),
            'SYM2': PerformanceMetrics(symbol='SYM2', accuracy=70.0),
            'SYM3': PerformanceMetrics(symbol='SYM3', accuracy=72.0)
        }
        scope = retraining_manager.determine_retraining_scope(low_performance)
        self.assertEqual(scope, RetrainingScope.GLOBAL)

        # 一部低性能（銘柄別再学習）
        mixed_performance = {
            'SYM1': PerformanceMetrics(symbol='SYM1', accuracy=95.0),
            'SYM2': PerformanceMetrics(symbol='SYM2', accuracy=92.0),
            'SYM3': PerformanceMetrics(symbol='SYM3', accuracy=82.0)  # 低性能
        }
        scope = retraining_manager.determine_retraining_scope(mixed_performance)
        self.assertEqual(scope, RetrainingScope.SYMBOL)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cooldown_period_check(self):
        """冷却期間チェックテスト"""
        retraining_manager = GranularRetrainingManager(self.config_manager)

        # 初回は実行可能
        self.assertTrue(
            retraining_manager.check_cooldown_period(RetrainingScope.GLOBAL)
        )

        # 実行後は冷却期間
        retraining_manager.update_retraining_time(RetrainingScope.GLOBAL)
        self.assertFalse(
            retraining_manager.check_cooldown_period(RetrainingScope.GLOBAL)
        )

        # 増分学習は短い冷却期間
        retraining_manager.update_retraining_time(RetrainingScope.INCREMENTAL)
        # 増分学習は6時間冷却なので、グローバルよりも早く実行可能になる
        self.assertFalse(
            retraining_manager.check_cooldown_period(RetrainingScope.INCREMENTAL)
        )

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_incremental_update_detection(self):
        """増分学習必要性検出テスト"""
        retraining_manager = GranularRetrainingManager(self.config_manager)

        # 微小な劣化で増分学習が提案される
        slight_degradation = {
            'SYM1': PerformanceMetrics(symbol='SYM1', accuracy=89.0),  # 軽微な劣化
            'SYM2': PerformanceMetrics(symbol='SYM2', accuracy=92.0),
            'SYM3': PerformanceMetrics(symbol='SYM3', accuracy=91.0)
        }
        self.assertTrue(
            retraining_manager._should_incremental_update(slight_degradation)
        )

        # 良好な性能では増分学習不要
        good_performance = {
            'SYM1': PerformanceMetrics(symbol='SYM1', accuracy=95.0),
            'SYM2': PerformanceMetrics(symbol='SYM2', accuracy=92.0),
            'SYM3': PerformanceMetrics(symbol='SYM3', accuracy=91.0)
        }
        self.assertFalse(
            retraining_manager._should_incremental_update(good_performance)
        )


class TestPerformanceMetricsAndAlerts(unittest.TestCase):
    """性能メトリクスとアラートテスト"""

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_performance_metrics_creation(self):
        """性能メトリクス作成テスト"""
        metrics = PerformanceMetrics(
            symbol='TEST',
            accuracy=85.5,
            prediction_accuracy=80.0,
            return_prediction=75.0,
            sample_count=100
        )

        self.assertEqual(metrics.symbol, 'TEST')
        self.assertEqual(metrics.accuracy, 85.5)
        self.assertEqual(metrics.sample_count, 100)
        self.assertIsNotNone(metrics.timestamp)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_performance_alert_creation(self):
        """性能アラート作成テスト"""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            symbol='TEST',
            metric='accuracy',
            current_value=85.0,
            threshold=90.0,
            message='Performance below threshold',
            recommended_action='Monitor closely'
        )

        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.symbol, 'TEST')
        self.assertEqual(alert.current_value, 85.0)
        self.assertEqual(alert.threshold, 90.0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_retraining_result_creation(self):
        """再学習結果作成テスト"""
        result = RetrainingResult(
            triggered=True,
            scope=RetrainingScope.PARTIAL,
            affected_symbols=['SYM1', 'SYM2'],
            improvement=5.2,
            duration=1800.0,
            estimated_time=1500.0,
            actual_benefit=4.8
        )

        self.assertTrue(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.PARTIAL)
        self.assertEqual(len(result.affected_symbols), 2)
        self.assertEqual(result.improvement, 5.2)


class TestEnhancedModelPerformanceMonitor(unittest.TestCase):
    """改善版モデル性能監視システム統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_monitor_config.yaml"
        self.db_path = self.temp_dir / "test_upgrade_system.db"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_monitor_initialization(self):
        """監視システム初期化テスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        self.assertIsNotNone(monitor.config_manager)
        self.assertIsNotNone(monitor.symbol_manager)
        self.assertIsNotNone(monitor.retraining_manager)
        self.assertTrue(self.db_path.exists())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_database_initialization(self):
        """データベース初期化テスト"""
        EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # テーブルが作成されることを確認
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # enhanced_performance_historyテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='enhanced_performance_history'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # enhanced_retraining_historyテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='enhanced_retraining_history'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # performance_alertsテーブル
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='performance_alerts'"
            )
            self.assertIsNotNone(cursor.fetchone())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_performance_monitoring_with_mock(self, mock_validator_class):
        """モック版性能監視テスト"""
        # モックの設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 87.5
        mock_metrics.prediction_accuracy = 85.0
        mock_metrics.sample_count = 50
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics
        mock_validator_class.return_value = mock_validator

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 特定銘柄での性能取得
        performance_data = await monitor.get_latest_model_performance(['TEST_SYMBOL'])

        self.assertIn('TEST_SYMBOL', performance_data)
        self.assertEqual(performance_data['TEST_SYMBOL'].accuracy, 87.5)
        self.assertEqual(performance_data['TEST_SYMBOL'].sample_count, 50)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.ml_upgrade_system')
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_retraining_trigger_with_mock(
        self, mock_validator_class, mock_upgrade
    ):
        """モック版再学習トリガーテスト"""
        # モックの設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 75.0  # 低性能でグローバル再学習をトリガー
        mock_metrics.prediction_accuracy = 70.0
        mock_metrics.sample_count = 30
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics
        mock_validator_class.return_value = mock_validator

        # 再学習システムのモック
        mock_report = Mock()
        mock_report.overall_improvement = 8.5
        mock_upgrade.run_complete_system_upgrade.return_value = mock_report
        mock_upgrade.integrate_best_models.return_value = None

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 再学習チェック実行
        result = await monitor.check_and_trigger_enhanced_retraining()

        self.assertTrue(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.GLOBAL)
        self.assertEqual(result.improvement, 8.5)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_performance_summary(self):
        """性能サマリー取得テスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        summary = monitor.get_enhanced_performance_summary()

        # 基本項目が含まれることを確認
        self.assertIn('timestamp', summary)
        self.assertIn('config_path', summary)
        self.assertIn('monitoring_symbols', summary)
        self.assertIn('symbol_count', summary)
        self.assertIn('integrations', summary)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_factory_function(self):
        """ファクトリー関数テスト"""
        monitor = create_enhanced_performance_monitor(str(self.config_path))

        self.assertIsInstance(monitor, EnhancedModelPerformanceMonitor)
        self.assertEqual(str(monitor.config_manager.config_path), str(self.config_path))


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
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_full_monitoring_cycle(self, mock_validator_class):
        """完全な監視サイクルテスト"""
        # 段階的に性能が悪化するシナリオ
        performance_sequence = [
            92.0,  # 良好
            89.0,  # 軽微な劣化（増分学習）
            86.0,  # 中程度の劣化（銘柄別再学習）
            78.0   # 重大な劣化（グローバル再学習）
        ]

        mock_validator = AsyncMock()
        mock_validator_class.return_value = mock_validator

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        results = []
        for accuracy in performance_sequence:
            # モックの性能データを設定
            mock_metrics = Mock()
            mock_metrics.overall_accuracy = accuracy
            mock_metrics.prediction_accuracy = accuracy - 2.0
            mock_metrics.sample_count = 25
            mock_validator.validate_current_system_accuracy.return_value = mock_metrics

            # 性能取得と再学習チェック
            await monitor.get_latest_model_performance(['TEST'])
            retraining_result = await monitor.check_and_trigger_enhanced_retraining()

            results.append({
                'accuracy': accuracy,
                'scope': retraining_result.scope,
                'triggered': retraining_result.triggered
            })

        # 結果の検証
        self.assertEqual(results[0]['scope'], RetrainingScope.NONE)  # 良好
        self.assertEqual(results[1]['scope'], RetrainingScope.INCREMENTAL)  # 軽微
        self.assertEqual(results[2]['scope'], RetrainingScope.SYMBOL)  # 中程度
        self.assertEqual(results[3]['scope'], RetrainingScope.GLOBAL)  # 重大

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_config_file_reload(self):
        """設定ファイル再読み込みテスト"""
        # 初期設定
        initial_config = {
            'performance_thresholds': {'accuracy': 90.0},
            'monitoring': {'default_symbols': ['INIT1', 'INIT2']}
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(initial_config, f)

        monitor = EnhancedModelPerformanceMonitor(config_path=self.config_path)
        initial_symbols = monitor.symbol_manager.get_monitoring_symbols()

        # 設定変更
        updated_config = {
            'performance_thresholds': {'accuracy': 85.0},
            'monitoring': {'default_symbols': ['UPDATE1', 'UPDATE2', 'UPDATE3']}
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f)

        # 設定再読み込み
        monitor.config_manager.reload_if_modified()
        updated_symbols = monitor.symbol_manager.get_monitoring_symbols()

        # 変更が反映されることを確認
        self.assertNotEqual(set(initial_symbols), set(updated_symbols))
        self.assertIn('UPDATE1', updated_symbols)
        self.assertEqual(monitor.config_manager.get_threshold('accuracy'), 85.0)


def run_enhanced_performance_monitor_tests():
    """改善版性能監視システムテスト実行"""
    print("=== Enhanced Model Performance Monitor テスト開始 ===")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestEnhancedPerformanceConfigManager,
        TestDynamicSymbolManager,
        TestGranularRetrainingManager,
        TestPerformanceMetricsAndAlerts,
        TestEnhancedModelPerformanceMonitor,
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

    success = run_enhanced_performance_monitor_tests()
    sys.exit(0 if success else 1)
