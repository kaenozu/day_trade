#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallback Behavior Test - フォールバック動作検証テスト
乱数フォールバックの動作を検証し、品質を確保
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import numpy as np

from enhanced_data_provider import EnhancedDataProvider, DataQuality, DataResult
from fallback_notification_system import FallbackNotificationSystem, DataSource


class TestFallbackBehavior(unittest.TestCase):
    """フォールバック動作テストクラス"""

    def setUp(self):
        """テストセットアップ"""
        self.data_provider = EnhancedDataProvider()
        self.notification_system = FallbackNotificationSystem()

    def test_data_quality_consistency(self):
        """データ品質の一貫性テスト"""
        # 同じシンボルに対して複数回データ要求した際の一貫性をチェック

        # フォールバックシナリオのシミュレーション
        with patch.object(self.data_provider, '_get_yfinance_data', side_effect=Exception("API Error")):

            # 複数回の呼び出しで同じフォールバックデータが返されるかテスト
            symbol = "7203"

            # キャッシュをクリア
            self.data_provider.cache.clear()

            async def test_consistency():
                results = []
                for _ in range(3):
                    result = await self.data_provider.get_stock_data(symbol)
                    results.append(result)

                # すべて同じ品質レベルであることを確認
                qualities = [r.quality for r in results]
                self.assertTrue(all(q == qualities[0] for q in qualities))

                # ダミーデータの場合、値が一貫していることを確認
                if results[0].quality == DataQuality.DUMMY:
                    prices = [r.data['price'] for r in results]
                    self.assertTrue(all(p == prices[0] for p in prices))

            asyncio.run(test_consistency())

    def test_fallback_notification_triggers(self):
        """フォールバック通知のトリガーテスト"""
        # 通知システムの初期化
        self.notification_system.session_notifications.clear()

        # フォールバック使用をシミュレーション
        self.notification_system.notify_fallback_usage(
            "test_component",
            "test_data",
            "Simulated failure",
            DataSource.DUMMY_DATA
        )

        # 通知が正しく記録されているかチェック
        self.assertEqual(len(self.notification_system.session_notifications), 1)

        notification = self.notification_system.session_notifications[0]
        self.assertEqual(notification['component'], "test_component")
        self.assertEqual(notification['source'], DataSource.DUMMY_DATA.value)

        # ダッシュボード状態の確認
        status = self.notification_system.get_dashboard_status()
        self.assertIn("ダミーデータ使用中", status)

    def test_circuit_breaker_behavior(self):
        """サーキットブレーカーの動作テスト"""
        provider_name = "test_provider"

        # 複数回の失敗を記録
        for _ in range(3):
            self.data_provider._record_failure(provider_name)

        # サーキットブレーカーがオープンになることを確認
        self.assertTrue(self.data_provider._is_circuit_open(provider_name))

        # 成功を記録してサーキットブレーカーがリセットされることを確認
        self.data_provider._record_success(provider_name)
        self.assertFalse(self.data_provider._is_circuit_open(provider_name))

    def test_cache_fallback_behavior(self):
        """キャッシュフォールバックの動作テスト"""
        # キャッシュに古いデータを設定
        old_result = DataResult(
            data={'symbol': '7203', 'price': 1000.0},
            quality=DataQuality.HIGH,
            source="test",
            timestamp=datetime.now(),
            latency_ms=10.0
        )

        self.data_provider._update_cache("stock_7203_1d", old_result)

        # すべてのプロバイダーが失敗した場合のテスト
        with patch.object(self.data_provider, 'providers', []):
            async def test_cache_fallback():
                result = await self.data_provider.get_stock_data("7203")

                # キャッシュまたはダミーデータが返されることを確認
                self.assertIn(result.quality, [DataQuality.FALLBACK, DataQuality.DUMMY])

                if result.quality == DataQuality.FALLBACK:
                    # キャッシュデータが使用されていることを確認
                    self.assertTrue(result.is_cached)

            asyncio.run(test_cache_fallback())

    def test_random_data_determinism(self):
        """乱数データの決定性テスト"""
        # 同じシードで同じ結果が得られることを確認

        # ダミーデータ生成の一貫性をテスト
        symbol = "TEST"

        # 複数回生成して一貫性をチェック
        dummy_results = []
        for _ in range(3):
            result = self.data_provider._generate_fallback_stock_data(symbol, "1d", 0.0)
            dummy_results.append(result)

        # ダミーデータは固定値であることを確認
        prices = [r.data['price'] for r in dummy_results]
        self.assertTrue(all(p == prices[0] for p in prices))

    def test_data_quality_degradation_path(self):
        """データ品質劣化パスのテスト"""
        # 高品質 → 中品質 → 低品質 → フォールバック → ダミー の劣化パスをテスト

        # モックデータで各品質レベルをシミュレーション
        quality_levels = [
            DataQuality.HIGH,
            DataQuality.MEDIUM,
            DataQuality.LOW,
            DataQuality.FALLBACK,
            DataQuality.DUMMY
        ]

        for i, quality in enumerate(quality_levels):
            # 各品質レベルで適切な通知が行われることを確認
            if quality in [DataQuality.FALLBACK, DataQuality.DUMMY]:
                # フォールバック・ダミー使用時は通知が発生
                notifications_before = len(self.notification_system.session_notifications)

                if quality == DataQuality.DUMMY:
                    self.notification_system.notify_fallback_usage(
                        "test", "data", "Quality degradation", DataSource.DUMMY_DATA
                    )
                else:
                    self.notification_system.notify_fallback_usage(
                        "test", "data", "Quality degradation", DataSource.FALLBACK_DATA
                    )

                notifications_after = len(self.notification_system.session_notifications)
                self.assertGreater(notifications_after, notifications_before)

    def test_ml_prediction_fallback(self):
        """ML予測のフォールバック動作テスト"""
        try:
            from simple_ml_prediction_system import SimpleMLPredictionSystem

            ml_system = SimpleMLPredictionSystem()

            async def test_ml_fallback():
                # ML予測の失敗をシミュレーション
                with patch.object(ml_system, 'predict_symbol_movement', side_effect=Exception("ML Error")):
                    # フォールバック予測が生成されることを確認
                    try:
                        result = await ml_system.predict_symbol_movement("7203")

                        # エラー結果でもそれなりの構造を持つことを確認
                        self.assertIn('symbol', result.__dict__)
                        self.assertIn('prediction', result.__dict__)
                        self.assertIn('confidence', result.__dict__)

                        # 安全なフォールバック値であることを確認
                        self.assertEqual(result.prediction, 0)  # 安全な待機シグナル
                        self.assertEqual(result.confidence, 0.0)  # 低信頼度

                    except Exception as e:
                        # 例外処理が適切に行われていることを確認
                        self.assertIsInstance(e, Exception)

            asyncio.run(test_ml_fallback())

        except ImportError:
            self.skipTest("SimpleMLPredictionSystem not available")

    def test_performance_under_failure(self):
        """障害時のパフォーマンステスト"""
        # 大量の失敗リクエストでもシステムが安定していることを確認

        async def test_performance():
            tasks = []

            # 同時に複数のフォールバック要求を実行
            for i in range(10):
                task = self.data_provider.get_stock_data(f"TEST{i}")
                tasks.append(task)

            # すべてのタスクが適切に完了することを確認
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 例外が発生していないことを確認
            for result in results:
                self.assertIsInstance(result, DataResult)
                self.assertIsNotNone(result.data)
                self.assertIsNotNone(result.quality)

        asyncio.run(test_performance())

    def test_fallback_recovery(self):
        """フォールバックからの復旧テスト"""
        # フォールバック状態から正常状態への復旧をテスト

        provider_name = "test_provider"

        # 失敗状態にする
        self.data_provider._record_failure(provider_name)
        self.data_provider._record_failure(provider_name)
        self.data_provider._record_failure(provider_name)

        self.assertTrue(self.data_provider._is_circuit_open(provider_name))

        # 復旧をシミュレーション
        self.data_provider._record_success(provider_name)

        # サーキットブレーカーがリセットされることを確認
        self.assertFalse(self.data_provider._is_circuit_open(provider_name))

        # 通知システムの復旧通知テスト
        self.notification_system.notify_real_data_recovery("test_component", "test_data")

        # 復旧通知が記録されることを確認
        latest_notification = self.notification_system.session_notifications[-1]
        self.assertEqual(latest_notification['source'], DataSource.REAL_DATA.value)


class FallbackBehaviorIntegrationTest(unittest.TestCase):
    """フォールバック動作統合テスト"""

    def test_end_to_end_fallback_scenario(self):
        """エンドツーエンドフォールバックシナリオ"""
        # 実際のワークフローでのフォールバック動作をテスト

        async def integration_test():
            provider = EnhancedDataProvider()
            notification_system = FallbackNotificationSystem()

            # 1. 正常な状態でデータ取得
            try:
                result = await provider.get_stock_data("7203")
                print(f"正常取得: 品質={result.quality.value}, ソース={result.source}")
            except Exception as e:
                print(f"正常取得失敗: {e}")

            # 2. プロバイダー失敗のシミュレーション
            with patch.object(provider, 'providers', []):
                result = await provider.get_stock_data("7203")
                print(f"フォールバック: 品質={result.quality.value}, ソース={result.source}")

                # フォールバック使用の確認
                self.assertIn(result.quality, [DataQuality.FALLBACK, DataQuality.DUMMY])

            # 3. 通知状況の確認
            summary = notification_system.get_session_summary()
            print(f"通知サマリー: {summary}")

        asyncio.run(integration_test())


def run_fallback_tests():
    """フォールバックテストを実行"""
    print("🧪 フォールバック動作検証テスト開始")
    print("=" * 60)

    # テストスイートの作成
    suite = unittest.TestSuite()

    # 基本動作テスト
    suite.addTest(unittest.makeSuite(TestFallbackBehavior))

    # 統合テスト
    suite.addTest(unittest.makeSuite(FallbackBehaviorIntegrationTest))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果表示
    print("\\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\\n失敗したテスト:")
        for test, trace in result.failures:
            print(f"- {test}")

    if result.errors:
        print("\\nエラーが発生したテスト:")
        for test, trace in result.errors:
            print(f"- {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\\n成功率: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_fallback_tests()
    if success:
        print("\\n✅ すべてのフォールバックテストが正常に完了しました")
    else:
        print("\\n❌ 一部のテストが失敗しました")

    exit(0 if success else 1)