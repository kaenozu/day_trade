#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test Suite - 統合テストスイート
本番運用前の包括的なシステムテスト
"""

import asyncio
import unittest
import tempfile
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# テスト対象モジュール
from daytrade_core import DayTradeCore
from daytrade_web import DayTradeWebServer
from enhanced_data_provider import get_data_provider
from enhanced_personal_analysis_engine import get_analysis_engine
from ml_accuracy_improvement_system import get_accuracy_system
from fallback_notification_system import get_notification_system
from market_time_manager import get_market_manager


class IntegrationTestSuite(unittest.TestCase):
    """統合テストスイート"""
    
    @classmethod
    def setUpClass(cls):
        """テストクラス初期化"""
        cls.test_symbols = ["7203", "8306", "9984", "6758"]
        cls.temp_dir = tempfile.mkdtemp()
        print(f"🧪 Integration Test Suite 開始")
        print(f"📁 テスト用一時ディレクトリ: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """テストクラス終了処理"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        print("✅ Integration Test Suite 完了")
    
    def setUp(self):
        """各テスト前の初期化"""
        self.start_time = datetime.now()
        
    def tearDown(self):
        """各テスト後の処理"""
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"  ⏱  実行時間: {duration:.2f}秒")


class DataProviderIntegrationTest(IntegrationTestSuite):
    """データプロバイダー統合テスト"""
    
    def test_data_provider_basic_functionality(self):
        """データプロバイダー基本機能テスト"""
        print("\n📊 データプロバイダー基本機能テスト")
        
        async def test_data_operations():
            provider = get_data_provider()
            
            # 株価データ取得テスト
            for symbol in self.test_symbols:
                result = await provider.get_stock_data(symbol)
                
                self.assertIsNotNone(result)
                self.assertIsNotNone(result.data)
                self.assertIsNotNone(result.quality)
                self.assertIn('symbol', result.data)
                self.assertIn('price', result.data)
                
                print(f"  ✓ {symbol}: 品質={result.quality.value}")
            
            # 市場データ取得テスト
            market_result = await provider.get_market_data()
            self.assertIsNotNone(market_result)
            self.assertIn('market_status', market_result.data)
            
            print("  ✓ 市場データ取得成功")
        
        asyncio.run(test_data_operations())
    
    def test_fallback_behavior(self):
        """フォールバック動作テスト"""
        print("\n🔄 フォールバック動作テスト")
        
        async def test_fallback():
            provider = get_data_provider()
            
            # プロバイダー失敗をシミュレーション
            original_providers = provider.providers
            provider.providers = []  # 全プロバイダーを無効化
            
            try:
                result = await provider.get_stock_data("7203")
                
                # フォールバックまたはダミーデータが返されることを確認
                self.assertIsNotNone(result)
                self.assertTrue(
                    result.quality.value in ['fallback', 'dummy']
                )
                
                print(f"  ✓ フォールバック動作確認: {result.quality.value}")
                
            finally:
                # プロバイダーを復元
                provider.providers = original_providers
        
        asyncio.run(test_fallback())


class AnalysisEngineIntegrationTest(IntegrationTestSuite):
    """分析エンジン統合テスト"""
    
    def test_analysis_engine_functionality(self):
        """分析エンジン機能テスト"""
        print("\n🔍 分析エンジン機能テスト")
        
        async def test_analysis():
            engine = get_analysis_engine()
            
            # 各分析モードのテスト
            from enhanced_personal_analysis_engine import AnalysisMode
            
            for mode in AnalysisMode:
                result = await engine.analyze_symbol("7203", mode)
                
                self.assertIsNotNone(result)
                self.assertIsNotNone(result.signal)
                self.assertIsNotNone(result.confidence)
                self.assertIsInstance(result.confidence, float)
                self.assertTrue(0.0 <= result.confidence <= 1.0)
                
                print(f"  ✓ {mode.value}: シグナル={result.signal.value}, 信頼度={result.confidence:.2f}")
            
            # ポートフォリオ分析テスト
            portfolio_results = await engine.analyze_portfolio(self.test_symbols)
            
            self.assertEqual(len(portfolio_results), len(self.test_symbols))
            
            for result in portfolio_results:
                self.assertIsNotNone(result.symbol)
                self.assertIsNotNone(result.signal)
            
            print(f"  ✓ ポートフォリオ分析: {len(portfolio_results)}銘柄")
        
        asyncio.run(test_analysis())
    
    def test_ml_accuracy_system(self):
        """ML精度向上システムテスト"""
        print("\n🎯 ML精度向上システムテスト")
        
        async def test_ml_accuracy():
            system = get_accuracy_system()
            
            # 精度評価テスト
            report = await system.evaluate_current_accuracy()
            
            self.assertIsNotNone(report)
            self.assertIsNotNone(report.model_name)
            self.assertIsInstance(report.accuracy_metrics, dict)
            self.assertGreater(len(report.accuracy_metrics), 0)
            
            print(f"  ✓ 精度評価: モデル={report.model_name}")
            
            # 改善戦略実装テスト
            strategies = ["feature_engineering", "hyperparameter_tuning"]
            improvements = await system.implement_accuracy_improvements(strategies)
            
            self.assertIsNotNone(improvements)
            self.assertIn('implemented', improvements)
            self.assertIn('failed', improvements)
            
            print(f"  ✓ 改善戦略: 成功={len(improvements['implemented'])}")
        
        asyncio.run(test_ml_accuracy())


class WebServerIntegrationTest(IntegrationTestSuite):
    """Webサーバー統合テスト"""
    
    def test_web_server_initialization(self):
        """Webサーバー初期化テスト"""
        print("\n🌐 Webサーバー初期化テスト")
        
        try:
            server = DayTradeWebServer(port=8001, debug=False)
            self.assertIsNotNone(server.app)
            
            # アプリケーションのルート確認
            with server.app.test_client() as client:
                # メインダッシュボード
                response = client.get('/')
                self.assertEqual(response.status_code, 200)
                print("  ✓ メインダッシュボード応答正常")
                
                # 分析API
                response = client.get('/api/analysis')
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertIn('status', data)
                print("  ✓ 分析API応答正常")
                
                # ML詳細API
                response = client.get('/api/ml-details')
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertIn('status', data)
                print("  ✓ ML詳細API応答正常")
                
                # データ品質API
                response = client.get('/api/data-quality')
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertIn('status', data)
                print("  ✓ データ品質API応答正常")
                
                # リスク監視API
                response = client.get('/api/risk-monitoring')
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertIn('status', data)
                print("  ✓ リスク監視API応答正常")
        
        except ImportError:
            self.skipTest("Web機能の依存関係が不足")


class NotificationSystemIntegrationTest(IntegrationTestSuite):
    """通知システム統合テスト"""
    
    def test_notification_system(self):
        """通知システムテスト"""
        print("\n📢 通知システムテスト")
        
        notification_system = get_notification_system()
        
        # 通知の初期状態確認
        initial_summary = notification_system.get_session_summary()
        initial_count = initial_summary['total_notifications']
        
        # ダミーデータ使用通知
        from fallback_notification_system import DataSource
        notification_system.notify_fallback_usage(
            "test_component", 
            "test_data", 
            "Integration test", 
            DataSource.DUMMY_DATA
        )
        
        # 通知が追加されたことを確認
        updated_summary = notification_system.get_session_summary()
        self.assertGreater(
            updated_summary['total_notifications'], 
            initial_count
        )
        
        # ダッシュボード状態の確認
        dashboard_status = notification_system.get_dashboard_status()
        self.assertIsNotNone(dashboard_status)
        self.assertIn("ダミーデータ", dashboard_status)
        
        print("  ✓ 通知システム正常動作")


class MarketTimeManagerIntegrationTest(IntegrationTestSuite):
    """市場時間管理統合テスト"""
    
    def test_market_time_manager(self):
        """市場時間管理システムテスト"""
        print("\n⏰ 市場時間管理システムテスト")
        
        manager = get_market_manager()
        
        # 市場状況の取得
        market_status = manager.get_market_status()
        self.assertIsNotNone(market_status)
        print(f"  ✓ 市場状況: {market_status.value}")
        
        # 市場オープン状況の確認
        is_open = manager.is_market_open()
        self.assertIsInstance(is_open, bool)
        print(f"  ✓ 市場オープン: {is_open}")
        
        # 自動更新判定
        should_update = manager.should_auto_update()
        self.assertIsInstance(should_update, bool)
        print(f"  ✓ 自動更新必要: {should_update}")


class EndToEndIntegrationTest(IntegrationTestSuite):
    """エンドツーエンド統合テスト"""
    
    def test_complete_workflow(self):
        """完全ワークフローテスト"""
        print("\n🔄 完全ワークフローテスト")
        
        async def test_e2e_workflow():
            # 1. データ取得
            provider = get_data_provider()
            stock_data = await provider.get_stock_data("7203")
            self.assertIsNotNone(stock_data)
            print("  ✓ データ取得")
            
            # 2. 分析実行
            engine = get_analysis_engine()
            analysis_result = await engine.analyze_symbol("7203")
            self.assertIsNotNone(analysis_result)
            print(f"  ✓ 分析実行: {analysis_result.signal.value}")
            
            # 3. ML精度評価
            accuracy_system = get_accuracy_system()
            accuracy_report = await accuracy_system.evaluate_current_accuracy()
            self.assertIsNotNone(accuracy_report)
            print(f"  ✓ 精度評価: {accuracy_report.model_name}")
            
            # 4. 通知システム確認
            notification_system = get_notification_system()
            summary = notification_system.get_session_summary()
            self.assertIsNotNone(summary)
            print(f"  ✓ 通知システム: {summary['total_notifications']}件")
            
            # 5. 市場時間管理確認
            market_manager = get_market_manager()
            market_status = market_manager.get_market_status()
            self.assertIsNotNone(market_status)
            print(f"  ✓ 市場時間管理: {market_status.value}")
        
        asyncio.run(test_e2e_workflow())


def run_integration_tests():
    """統合テストの実行"""
    print("🧪 Day Trade AI システム統合テスト開始")
    print("=" * 60)
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # 各テストクラスを追加
    test_classes = [
        DataProviderIntegrationTest,
        AnalysisEngineIntegrationTest,
        WebServerIntegrationTest,
        NotificationSystemIntegrationTest,
        MarketTimeManagerIntegrationTest,
        EndToEndIntegrationTest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 統合テスト結果サマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失敗したテスト:")
        for test, trace in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n💥 エラーが発生したテスト:")
        for test, trace in result.errors:
            print(f"  - {test}")
    
    # 成功率計算
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📊 成功率: {success_rate:.1f}%")
    
    # 結果判定
    if result.wasSuccessful():
        print("\n✅ 全統合テストが成功しました - 本番運用準備完了！")
        return True
    else:
        print("\n❌ 一部のテストが失敗しました - 修正が必要です")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)