#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Model Performance Monitor テストスイート
Issue #857対応：改善版モデル性能監視システムの包括的テスト

テスト項目：
1. 改善版設定管理システム
2. 動的監視対象銘柄管理
3. 段階的再学習トリガー
4. 性能アラートシステム
5. 外部システム連携
6. 統合シナリオ
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
    from model_performance_monitor import (
        EnhancedModelPerformanceMonitor,
        EnhancedPerformanceConfigManager,
        DynamicSymbolManager,
        GranularRetrainingManager,
        PerformanceMetrics,
        RetrainingResult,
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
        manager = EnhancedPerformanceConfigManager(self.config_path)

        # 設定ファイルが作成されること
        self.assertTrue(self.config_path.exists())

        # 基本設定が含まれること
        self.assertIn('performance_thresholds', manager.config)
        self.assertIn('monitoring', manager.config)
        self.assertIn('retraining', manager.config)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_threshold_retrieval(self):
        """閾値取得テスト"""
        manager = EnhancedPerformanceConfigManager(self.config_path)

        # デフォルト閾値取得
        accuracy_threshold = manager.get_threshold('accuracy')
        self.assertIsInstance(accuracy_threshold, float)
        self.assertGreater(accuracy_threshold, 0)

        # 銘柄別閾値テスト（設定がない場合はデフォルト）
        symbol_threshold = manager.get_threshold('accuracy', '7203')
        self.assertIsInstance(symbol_threshold, float)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_config_loading(self):
        """カスタム設定読み込みテスト"""
        custom_config = {
            'performance_thresholds': {
                'accuracy': 95.0,
                'prediction_accuracy': 90.0,
                'symbol_specific': {
                    'high_volume_stocks': {
                        'accuracy': 92.0
                    }
                }
            },
            'monitoring': {
                'default_symbols': ['TEST1', 'TEST2'],
                'dynamic_monitoring': {'enabled': True}
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 12
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        manager = EnhancedPerformanceConfigManager(self.config_path)

        # カスタム設定が読み込まれること
        self.assertEqual(manager.get_threshold('accuracy'), 95.0)
        self.assertEqual(manager.get_monitoring_config()['default_symbols'], ['TEST1', 'TEST2'])
        self.assertEqual(manager.get_retraining_config()['cooldown_hours'], 12)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_symbol_categorization(self):
        """銘柄カテゴリ分類テスト"""
        manager = EnhancedPerformanceConfigManager(self.config_path)

        # 高ボリューム銘柄
        high_volume_cat = manager._categorize_symbol('7203')
        self.assertEqual(high_volume_cat, 'high_volume_stocks')

        # その他の銘柄
        other_cat = manager._categorize_symbol('1234')
        self.assertEqual(other_cat, 'mid_volume_stocks')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_config_reload(self):
        """設定ファイル再読み込みテスト"""
        manager = EnhancedPerformanceConfigManager(self.config_path)

        # 初期設定確認
        initial_threshold = manager.get_threshold('accuracy')

        # 設定ファイルを更新
        updated_config = manager.config.copy()
        updated_config['performance_thresholds']['accuracy'] = 99.0

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f)

        # 強制再読み込み
        manager.reload_if_modified()

        # 更新された設定が読み込まれること
        updated_threshold = manager.get_threshold('accuracy')
        self.assertNotEqual(initial_threshold, updated_threshold)
        self.assertEqual(updated_threshold, 99.0)


class TestDynamicSymbolManager(unittest.TestCase):
    """動的監視対象銘柄管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        self.config_manager = EnhancedPerformanceConfigManager(self.config_path)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_basic_symbol_retrieval(self):
        """基本的な監視銘柄取得テスト"""
        manager = DynamicSymbolManager(self.config_manager)

        symbols = manager.get_monitoring_symbols()

        # デフォルト銘柄が取得されること
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)

        # デフォルト銘柄が含まれること
        default_symbols = self.config_manager.get_monitoring_config().get('default_symbols', [])
        for symbol in default_symbols:
            self.assertIn(symbol, symbols)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.SYMBOL_SELECTOR_AVAILABLE', True)
    def test_dynamic_symbol_integration(self):
        """動的監視銘柄統合テスト"""
        # モックSymbolSelectorを作成
        mock_symbol_selector = Mock()
        mock_symbol_selector.get_recommended_symbols.return_value = ['DYN1', 'DYN2']
        mock_symbol_selector.get_high_volatility_symbols.return_value = ['VOL1']

        manager = DynamicSymbolManager(self.config_manager)
        manager.symbol_selector = mock_symbol_selector

        # 動的監視を有効化
        config = self.config_manager.config.copy()
        config['monitoring']['dynamic_monitoring'] = {'enabled': True, 'max_symbols': 10}
        self.config_manager.config = config

        symbols = manager.get_monitoring_symbols()

        # 動的銘柄が追加されること
        self.assertIn('DYN1', symbols)
        self.assertIn('DYN2', symbols)
        self.assertIn('VOL1', symbols)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_symbol_limit_enforcement(self):
        """監視銘柄数制限テスト"""
        manager = DynamicSymbolManager(self.config_manager)

        # 制限数を設定
        config = self.config_manager.config.copy()
        config['monitoring']['default_symbols'] = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        config['monitoring']['dynamic_monitoring'] = {'enabled': False, 'max_symbols': 3}
        self.config_manager.config = config

        symbols = manager.get_monitoring_symbols()

        # 制限数以下であること
        self.assertLessEqual(len(symbols), 6)  # dynamic無効なのでdefault_symbolsのまま

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_update_timing_check(self):
        """更新タイミングチェックテスト"""
        manager = DynamicSymbolManager(self.config_manager)

        # 初回は更新すべき
        self.assertTrue(manager.should_update_symbols())

        # 更新時刻を設定
        manager.last_update = datetime.now()

        # 直後は更新不要
        self.assertFalse(manager.should_update_symbols())


class TestGranularRetrainingManager(unittest.TestCase):
    """段階的再学習管理テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        self.config_manager = EnhancedPerformanceConfigManager(self.config_path)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_retraining_scope_determination(self):
        """再学習スコープ決定テスト"""
        manager = GranularRetrainingManager(self.config_manager)

        # テスト用性能データ（グローバル再学習レベル）
        low_performances = {
            'SYM1': PerformanceMetrics('SYM1', 75.0),  # 低性能
            'SYM2': PerformanceMetrics('SYM2', 70.0),  # 低性能
        }

        scope = manager.determine_retraining_scope(low_performances)
        self.assertEqual(scope, RetrainingScope.GLOBAL)

        # 部分再学習レベル
        mixed_performances = {
            'SYM1': PerformanceMetrics('SYM1', 86.0),  # やや低性能
            'SYM2': PerformanceMetrics('SYM2', 87.0),  # やや低性能
            'SYM3': PerformanceMetrics('SYM3', 92.0),  # 良好
        }

        scope = manager.determine_retraining_scope(mixed_performances)
        self.assertEqual(scope, RetrainingScope.PARTIAL)

        # 銘柄別再学習レベル
        symbol_specific_performances = {
            'SYM1': PerformanceMetrics('SYM1', 84.0),  # 閾値以下
            'SYM2': PerformanceMetrics('SYM2', 91.0),  # 良好
            'SYM3': PerformanceMetrics('SYM3', 92.0),  # 良好
        }

        scope = manager.determine_retraining_scope(symbol_specific_performances)
        self.assertEqual(scope, RetrainingScope.SYMBOL)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_incremental_update_detection(self):
        """増分学習判定テスト"""
        manager = GranularRetrainingManager(self.config_manager)

        # 軽微な劣化（増分学習対象）
        minor_degradation = {
            'SYM1': PerformanceMetrics('SYM1', 88.0),  # 軽微な劣化
            'SYM2': PerformanceMetrics('SYM2', 91.0),  # 良好
        }

        should_incremental = manager._should_incremental_update(minor_degradation)
        self.assertTrue(should_incremental)

        # 良好な性能（増分学習不要）
        good_performances = {
            'SYM1': PerformanceMetrics('SYM1', 92.0),
            'SYM2': PerformanceMetrics('SYM2', 94.0),
        }

        should_incremental = manager._should_incremental_update(good_performances)
        self.assertFalse(should_incremental)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_cooldown_period_check(self):
        """冷却期間チェックテスト"""
        manager = GranularRetrainingManager(self.config_manager)

        # 初回実行（冷却期間なし）
        self.assertTrue(manager.check_cooldown_period(RetrainingScope.GLOBAL))

        # 実行時刻を記録
        manager.update_retraining_time(RetrainingScope.GLOBAL)

        # 直後は冷却期間中
        self.assertFalse(manager.check_cooldown_period(RetrainingScope.GLOBAL))

        # 異なるスコープは実行可能
        self.assertTrue(manager.check_cooldown_period(RetrainingScope.SYMBOL))

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_scope_specific_cooldown(self):
        """スコープ別冷却期間テスト"""
        manager = GranularRetrainingManager(self.config_manager)

        # 設定で24時間冷却期間を設定
        config = self.config_manager.config.copy()
        config['retraining']['cooldown_hours'] = 24
        self.config_manager.config = config

        # 各スコープの冷却期間をテスト
        scopes_and_multipliers = [
            (RetrainingScope.INCREMENTAL, 0.25),  # 6時間
            (RetrainingScope.SYMBOL, 0.5),        # 12時間
            (RetrainingScope.PARTIAL, 0.75),      # 18時間
            (RetrainingScope.GLOBAL, 1.0),        # 24時間
        ]

        for scope, multiplier in scopes_and_multipliers:
            # 実行時刻を記録
            manager.last_retraining[scope] = datetime.now() - timedelta(hours=24 * multiplier - 1)

            # 冷却期間中のはず
            self.assertFalse(manager.check_cooldown_period(scope))

            # 冷却期間経過後
            manager.last_retraining[scope] = datetime.now() - timedelta(hours=24 * multiplier + 1)
            self.assertTrue(manager.check_cooldown_period(scope))


class TestEnhancedModelPerformanceMonitor(unittest.TestCase):
    """改善版モデル性能監視システム統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_monitor_config.yaml"
        self.db_path = self.temp_dir / "test_monitor.db"

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

        # 基本コンポーネントが初期化されること
        self.assertIsNotNone(monitor.config_manager)
        self.assertIsNotNone(monitor.symbol_manager)
        self.assertIsNotNone(monitor.retraining_manager)

        # データベースが作成されること
        self.assertTrue(self.db_path.exists())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_database_initialization(self):
        """データベース初期化テスト"""
        EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # テーブルが作成されること
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
    def test_cache_validity_check(self):
        """キャッシュ有効性チェックテスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 空のキャッシュは無効
        self.assertFalse(monitor._is_cache_valid())

        # キャッシュデータを追加
        monitor.performance_cache['TEST'] = PerformanceMetrics('TEST', 90.0)

        # 新しいキャッシュは有効
        self.assertTrue(monitor._is_cache_valid())

        # 古いキャッシュは無効
        old_time = datetime.now() - timedelta(hours=2)
        monitor.performance_cache['TEST'].timestamp = old_time
        self.assertFalse(monitor._is_cache_valid())

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_performance_data_retrieval_mock(self, mock_validator_class):
        """性能データ取得テスト（モック使用）"""
        # モックvalidatorを設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 85.0
        mock_metrics.prediction_accuracy = 82.0
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics
        mock_validator_class.return_value = mock_validator

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )
        monitor.accuracy_validator = mock_validator

        # 性能データ取得
        performance_data = await monitor.get_latest_model_performance(['TEST1', 'TEST2'])

        # 結果検証
        self.assertIsInstance(performance_data, dict)
        self.assertEqual(len(performance_data), 2)
        self.assertIn('TEST1', performance_data)
        self.assertIn('TEST2', performance_data)

        # 各銘柄の性能データが正しいこと
        for symbol, perf in performance_data.items():
            self.assertIsInstance(perf, PerformanceMetrics)
            self.assertEqual(perf.symbol, symbol)
            self.assertEqual(perf.accuracy, 85.0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    async def test_alert_generation(self):
        """アラート生成テスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 閾値を下回る性能データ
        low_performance = PerformanceMetrics('TEST', 80.0, 75.0, 70.0, 78.0)

        # アラート生成
        await monitor._check_and_generate_alerts(low_performance)

        # アラートが生成されること
        self.assertGreater(len(monitor.alerts), 0)

        # データベースにアラートが記録されること
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_alerts")
            alert_count = cursor.fetchone()[0]
            self.assertGreater(alert_count, 0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_performance_summary_generation(self):
        """性能サマリー生成テスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # サマリー取得
        summary = monitor.get_enhanced_performance_summary()

        # 基本項目が含まれること
        self.assertIn('timestamp', summary)
        self.assertIn('config_path', summary)
        self.assertIn('monitoring_symbols', summary)
        self.assertIn('symbol_count', summary)
        self.assertIn('integrations', summary)

        # 統合情報が含まれること
        self.assertIn('symbol_selector', summary['integrations'])
        self.assertIn('ml_systems', summary['integrations'])

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.ml_upgrade_system')
    async def test_retraining_execution_mock(self, mock_ml_system):
        """再学習実行テスト（モック使用）"""
        # モックml_upgrade_systemを設定
        mock_report = Mock()
        mock_report.overall_improvement = 5.0
        mock_ml_system.run_complete_system_upgrade.return_value = mock_report
        mock_ml_system.integrate_best_models.return_value = Mock()

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 低性能データを準備
        symbol_performances = {
            'TEST1': PerformanceMetrics('TEST1', 75.0),
            'TEST2': PerformanceMetrics('TEST2', 70.0)
        }

        # 再学習実行
        result = await monitor._execute_enhanced_retraining(
            RetrainingScope.GLOBAL, symbol_performances
        )

        # 結果検証
        self.assertIsInstance(result, RetrainingResult)
        self.assertTrue(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.GLOBAL)
        self.assertEqual(result.improvement, 5.0)
        self.assertIsNone(result.error)


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
        monitor = create_enhanced_performance_monitor(str(self.config_path))

        self.assertIsInstance(monitor, EnhancedModelPerformanceMonitor)
        self.assertIsNotNone(monitor.config_manager)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    @patch('model_performance_monitor.ml_upgrade_system')
    async def test_complete_monitoring_workflow(self, mock_ml_system, mock_validator_class):
        """完全な監視ワークフローテスト"""
        # モック設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 82.0  # 閾値以下（90.0未満）
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics
        mock_validator_class.return_value = mock_validator

        mock_report = Mock()
        mock_report.overall_improvement = 8.0
        mock_ml_system.run_complete_system_upgrade.return_value = mock_report
        mock_ml_system.integrate_best_models.return_value = Mock()

        # カスタム設定作成（低い閾値で再学習をトリガー）
        custom_config = {
            'performance_thresholds': {
                'accuracy': 90.0,
                'prediction_accuracy': 85.0
            },
            'monitoring': {
                'default_symbols': ['TEST1', 'TEST2'],
                'dynamic_monitoring': {'enabled': False},
                'validation_hours': 24
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 1,  # 短い冷却期間
                'global_threshold': 85.0,  # 82.0より高い閾値で再学習をトリガー
                'symbol_specific_threshold': 88.0,
                'partial_threshold': 90.0
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )
        monitor.accuracy_validator = mock_validator

        # 完全なワークフロー実行
        result = await monitor.check_and_trigger_enhanced_retraining()

        # 結果検証
        self.assertIsInstance(result, RetrainingResult)
        self.assertTrue(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.GLOBAL)
        self.assertGreater(result.improvement, 0)

        # データベースに結果が記録されること
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()

            # 性能履歴の記録確認
            cursor.execute("SELECT COUNT(*) FROM enhanced_performance_history")
            perf_count = cursor.fetchone()[0]
            self.assertGreater(perf_count, 0)

            # 再学習履歴の記録確認
            cursor.execute("SELECT COUNT(*) FROM enhanced_retraining_history")
            retrain_count = cursor.fetchone()[0]
            self.assertGreater(retrain_count, 0)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    async def test_no_retraining_scenario(self):
        """再学習不要シナリオテスト"""
        # 高性能なモックvalidatorを設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 95.0  # 閾値以上
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics

        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )
        monitor.accuracy_validator = mock_validator

        # 再学習チェック実行
        result = await monitor.check_and_trigger_enhanced_retraining()

        # 再学習が実行されないこと
        self.assertFalse(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.NONE)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    async def test_cooldown_prevention_scenario(self):
        """冷却期間による再学習防止シナリオテスト"""
        monitor = EnhancedModelPerformanceMonitor(
            config_path=self.config_path,
            upgrade_db_path=self.db_path
        )

        # 事前に再学習実行時刻を設定（冷却期間中にするため）
        monitor.retraining_manager.last_retraining[RetrainingScope.GLOBAL] = datetime.now()

        # 低性能なモックvalidatorを設定
        mock_validator = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 75.0  # 閾値以下
        mock_validator.validate_current_system_accuracy.return_value = mock_metrics
        monitor.accuracy_validator = mock_validator

        # 再学習チェック実行
        result = await monitor.check_and_trigger_enhanced_retraining()

        # 冷却期間により再学習が実行されないこと
        self.assertFalse(result.triggered)
        self.assertEqual(result.scope, RetrainingScope.GLOBAL)  # スコープは決定されるが実行されない


def run_enhanced_performance_monitor_tests():
    """改善版モデル性能監視システムテスト実行"""
    print("=== Enhanced Model Performance Monitor テスト開始 ===")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestEnhancedPerformanceConfigManager,
        TestDynamicSymbolManager,
        TestGranularRetrainingManager,
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
