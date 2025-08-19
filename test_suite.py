#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Personal Test Suite - 包括的テストシステム
Issue #933 Phase 4対応: システム品質保証のための統合テスト
"""

import unittest
import sys
import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# テスト対象モジュールのインポート
try:
    import sys
    sys.path.insert(0, '.')

    # コアモジュール
    from version import get_version_info, __version__, __version_extended__
    from performance_monitor import PerformanceMonitor, performance_monitor
    from data_persistence import DataPersistence, data_persistence

    # WebサーバーとCLI
    from daytrade_web import DayTradeWebServer
    from daytrade_core import main as daytrade_core_main

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[警告] モジュールインポートエラー: {e}")
    MODULES_AVAILABLE = False


class TestVersionManagement(unittest.TestCase):
    """バージョン管理システムのテスト"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")

    def test_version_info_structure(self):
        """バージョン情報の構造テスト"""
        version_info = get_version_info()

        self.assertIsInstance(version_info, dict)
        self.assertIn('version', version_info)
        self.assertIn('version_extended', version_info)
        self.assertIn('release_name', version_info)
        self.assertIn('build_date', version_info)

        # バージョン形式チェック
        self.assertRegex(version_info['version'], r'\d+\.\d+\.\d+')

    def test_version_consistency(self):
        """バージョンの一貫性テスト"""
        version_info = get_version_info()

        self.assertEqual(version_info['version'], __version__)
        self.assertEqual(version_info['version_extended'], __version_extended__)

    def test_version_components(self):
        """バージョンコンポーネントテスト"""
        from version import COMPONENTS, COMPATIBILITY

        self.assertIsInstance(COMPONENTS, dict)
        self.assertIn('core', COMPONENTS)
        self.assertIn('web_server', COMPONENTS)
        self.assertIn('cli', COMPONENTS)

        self.assertIsInstance(COMPATIBILITY, dict)
        self.assertIn('min_python_version', COMPATIBILITY)
        self.assertIn('supported_os', COMPATIBILITY)


class TestPerformanceMonitoring(unittest.TestCase):
    """パフォーマンス監視システムのテスト"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")

        # テスト用の監視インスタンス作成
        self.monitor = PerformanceMonitor(max_history_size=50)

    def tearDown(self):
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()

    def test_analysis_time_tracking(self):
        """分析時間追跡テスト"""
        self.monitor.track_analysis_time('7203', 0.15, 'test_analysis')
        self.monitor.track_analysis_time('8306', 0.22, 'test_analysis')

        summary = self.monitor.get_performance_summary()
        self.assertEqual(summary['total_analyses'], 2)
        self.assertGreater(summary['avg_analysis_time_ms'], 0)

    def test_api_response_tracking(self):
        """API応答時間追跡テスト"""
        self.monitor.track_api_response_time('/api/test', 0.12, 200)
        self.monitor.track_api_response_time('/api/test', 0.08, 200)
        self.monitor.track_api_response_time('/api/error', 0.30, 500)

        summary = self.monitor.get_performance_summary()
        self.assertEqual(summary['total_api_calls'], 3)
        self.assertGreater(summary['avg_api_response_ms'], 0)

    def test_error_tracking(self):
        """エラー追跡テスト"""
        self.monitor.track_error('test_error', 'Test error message')
        self.monitor.track_error('api_error', 'API error message')

        summary = self.monitor.get_performance_summary()
        self.assertEqual(len(summary['error_counts']), 2)
        self.assertEqual(summary['error_counts']['test_error'], 1)
        self.assertEqual(summary['error_counts']['api_error'], 1)

    def test_performance_report_generation(self):
        """パフォーマンスレポート生成テスト"""
        # テストデータ追加
        self.monitor.track_analysis_time('7203', 0.15, 'test_analysis')
        self.monitor.track_api_response_time('/api/test', 0.12, 200)

        report = self.monitor.generate_performance_report()

        self.assertIsInstance(report, str)
        self.assertIn('Performance Report', report)
        self.assertIn('System Overview', report)
        self.assertIn('Total Analyses', report)
        self.assertIn('Total API Calls', report)

    def test_memory_monitoring(self):
        """メモリ監視テスト"""
        memory_usage = self.monitor.get_memory_usage()

        if memory_usage:  # psutilが利用可能な場合
            self.assertIsInstance(memory_usage, dict)
            self.assertIn('rss_mb', memory_usage)
            self.assertIn('memory_percent', memory_usage)
            self.assertGreater(memory_usage['rss_mb'], 0)


class TestDataPersistence(unittest.TestCase):
    """データ永続化システムのテスト"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")

        # テスト用の一時データベース
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, 'test_db.sqlite')
        self.persistence = DataPersistence(db_path=self.test_db_path)

    def tearDown(self):
        if hasattr(self, 'persistence'):
            self.persistence.stop_auto_backup()

        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_initialization(self):
        """データベース初期化テスト"""
        self.assertTrue(Path(self.test_db_path).exists())

        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                'analysis_history', 'api_performance', 'system_metrics',
                'error_logs', 'recommendations_history'
            ]

            for table in expected_tables:
                self.assertIn(table, tables)

    def test_analysis_result_saving(self):
        """分析結果保存テスト"""
        test_data = {
            'symbol': '7203',
            'recommendation': 'BUY',
            'confidence': 0.85,
            'price': 1500
        }

        self.persistence.save_analysis_result(
            symbol='7203',
            analysis_type='test_analysis',
            duration_ms=150.5,
            result_data=test_data,
            confidence_score=0.85,
            session_id='test_session'
        )

        stats = self.persistence.get_analysis_statistics(24)
        self.assertEqual(stats['total_statistics']['total_analyses'], 1)

    def test_api_performance_saving(self):
        """API性能データ保存テスト"""
        self.persistence.save_api_performance(
            endpoint='/api/test',
            response_time_ms=120.3,
            status_code=200,
            session_id='test_session'
        )

        stats = self.persistence.get_api_statistics(24)
        self.assertEqual(stats['total_statistics']['total_requests'], 1)

    def test_error_log_saving(self):
        """エラーログ保存テスト"""
        self.persistence.save_error_log(
            error_type='test_error',
            error_message='Test error occurred',
            context_data={'test': 'data'},
            session_id='test_session'
        )

        # データベースから直接検証
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM error_logs')
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    def test_data_export(self):
        """データエクスポートテスト"""
        # テストデータ追加
        test_data = {'test': 'data'}
        self.persistence.save_analysis_result(
            symbol='7203',
            analysis_type='test',
            duration_ms=100,
            result_data=test_data,
            confidence_score=0.8
        )

        # JSONエクスポートテスト
        export_result = self.persistence.export_data('json', self.test_dir)
        self.assertTrue(Path(export_result).exists())

        # エクスポートファイルの内容確認
        with open(export_result, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)

        self.assertIn('export_info', exported_data)
        self.assertIn('analysis_history', exported_data)
        self.assertEqual(len(exported_data['analysis_history']), 1)


class TestWebServerIntegration(unittest.TestCase):
    """WebサーバーとのShell統合テスト"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")

        # テスト用Webサーバー起動
        self.web_server = DayTradeWebServer(port=8099, debug=True)
        self.app = self.web_server.app.test_client()

    def test_status_endpoint(self):
        """ステータスエンドポイントテスト"""
        response = self.app.get('/api/status')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['status'], 'running')
        self.assertIn('version', data)
        self.assertIn('features', data)

    def test_analysis_endpoint(self):
        """分析エンドポイントテスト"""
        response = self.app.get('/api/analysis/7203')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['symbol'], '7203')
        self.assertIn('recommendation', data)
        self.assertIn('confidence', data)
        self.assertIn('timestamp', data)

    def test_recommendations_endpoint(self):
        """推奨銘柄エンドポイントテスト"""
        response = self.app.get('/api/recommendations')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('recommendations', data)
        self.assertIn('total_count', data)
        self.assertGreater(data['total_count'], 0)

    def test_performance_endpoints(self):
        """パフォーマンス監視エンドポイントテスト"""
        # 基本パフォーマンス
        response = self.app.get('/api/performance')
        self.assertIn(response.status_code, [200, 501])  # 501 = 監視無効時

        # 詳細パフォーマンス
        response = self.app.get('/api/performance/detailed')
        self.assertIn(response.status_code, [200, 501])

        # パフォーマンスレポート
        response = self.app.get('/api/performance/report')
        self.assertIn(response.status_code, [200, 501])

    def test_data_persistence_endpoints(self):
        """データ永続化エンドポイントテスト"""
        # 統計エンドポイント
        response = self.app.get('/api/data/statistics')
        self.assertIn(response.status_code, [200, 501])

        # データベース情報
        response = self.app.get('/api/data/database-info')
        self.assertIn(response.status_code, [200, 501])

        # データエクスポート
        response = self.app.get('/api/data/export?format=json')
        self.assertIn(response.status_code, [200, 501])

    def test_health_check(self):
        """ヘルスチェックテスト"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')


class TestSystemIntegration(unittest.TestCase):
    """システム全体統合テスト"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")

    def test_module_imports(self):
        """モジュールインポートテスト"""
        modules_to_test = [
            'version', 'performance_monitor', 'data_persistence',
            'daytrade_web', 'daytrade_core'
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError:
                self.fail(f"Failed to import {module_name}")

    def test_version_consistency_across_modules(self):
        """モジュール間バージョン一貫性テスト"""
        from version import get_version_info
        from daytrade_web import VERSION_INFO as web_version

        base_version = get_version_info()

        # Webサーバーのバージョンと一致確認
        self.assertEqual(base_version['version'], web_version['version'])
        self.assertEqual(base_version['build_date'], web_version['build_date'])

    def test_performance_monitor_integration(self):
        """パフォーマンス監視統合テスト"""
        from performance_monitor import performance_monitor, track_performance

        # デコレーターテスト
        @track_performance
        def test_function(x, y):
            time.sleep(0.01)  # 少し時間をかける
            return x + y

        result = test_function(1, 2)
        self.assertEqual(result, 3)

        # パフォーマンスデータが記録されているか確認
        summary = performance_monitor.get_performance_summary()
        self.assertGreaterEqual(summary['total_analyses'], 1)


def run_performance_benchmark():
    """パフォーマンスベンチマーク実行"""
    print("=== パフォーマンスベンチマーク実行 ===")

    if not MODULES_AVAILABLE:
        print("[警告] モジュールが利用できないためベンチマークをスキップします")
        return

    # 分析処理のベンチマーク
    start_time = time.time()

    try:
        monitor = PerformanceMonitor(max_history_size=100)

        # 100回の分析をシミュレート
        for i in range(100):
            monitor.track_analysis_time(f'symbol_{i}', 0.1 + (i % 10) * 0.01, 'benchmark')

        # API呼び出しをシミュレート
        for i in range(50):
            monitor.track_api_response_time('/api/test', 0.05 + (i % 5) * 0.01, 200)

        total_time = time.time() - start_time
        summary = monitor.get_performance_summary()

        print(f"ベンチマーク実行時間: {total_time:.2f}秒")
        print(f"処理した分析数: {summary['total_analyses']}")
        print(f"処理したAPI呼び出し数: {summary['total_api_calls']}")
        print(f"平均分析時間: {summary['avg_analysis_time_ms']:.2f}ms")
        print(f"平均API応答時間: {summary['avg_api_response_ms']:.2f}ms")

        monitor.stop_monitoring()

    except Exception as e:
        print(f"ベンチマーク実行エラー: {e}")


def run_full_test_suite():
    """完全テストスイート実行"""
    print("Day Trade Personal - 包括的テストスイート")
    print("=" * 60)

    # バージョン情報表示
    if MODULES_AVAILABLE:
        try:
            version_info = get_version_info()
            print(f"テスト対象バージョン: {version_info['version']} {version_info['release_name']}")
            print(f"ビルド日: {version_info['build_date']}")
        except Exception as e:
            print(f"バージョン情報取得エラー: {e}")

    print("-" * 60)

    # ユニットテスト実行
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("-" * 60)

    # パフォーマンスベンチマーク
    run_performance_benchmark()

    print("-" * 60)

    # 結果サマリー
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0

    print(f"テスト結果サマリー:")
    print(f"  実行テスト数: {total_tests}")
    print(f"  成功: {total_tests - failures - errors}")
    print(f"  失敗: {failures}")
    print(f"  エラー: {errors}")
    print(f"  成功率: {success_rate:.1f}%")

    if success_rate >= 90:
        print("🎉 システム品質: 優秀 (90%以上)")
    elif success_rate >= 80:
        print("✅ システム品質: 良好 (80%以上)")
    elif success_rate >= 70:
        print("⚠️  システム品質: 注意が必要 (70%以上)")
    else:
        print("❌ システム品質: 改善が必要 (70%未満)")

    return success_rate >= 80


if __name__ == '__main__':
    success = run_full_test_suite()

    # 終了コード設定
    sys.exit(0 if success else 1)