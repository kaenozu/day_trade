#!/usr/bin/env python3
"""
強化版性能監視システムテスト
Issue #857対応: 93%精度維持保証システム検証
"""

import asyncio
import unittest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import logging

from enhanced_performance_monitor import (
    EnhancedPerformanceMonitorV2,
    AccuracyGuaranteeSystem,
    AccuracyGuaranteeConfig,
    AccuracyGuaranteeLevel,
    MonitoringIntensity,
    ContinuousPerformanceMonitor,
    AccuracyTrendAnalyzer,
    EmergencyDetector,
    IntelligentRetrainingController,
    ContinuousMonitoringMetrics,
    PredictionQualityStatus
)

try:
    from src.day_trade.monitoring.model_performance_monitor import PerformanceMetrics, RetrainingResult, RetrainingScope
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # モック用クラス定義
    class PerformanceMetrics:
        def __init__(self, symbol, accuracy, **kwargs):
            self.symbol = symbol
            self.accuracy = accuracy
            self.timestamp = datetime.now()


class TestAccuracyGuaranteeSystem(unittest.TestCase):
    """精度保証システムテスト"""

    def setUp(self):
        """テスト前準備"""
        self.config = AccuracyGuaranteeConfig(
            guarantee_level=AccuracyGuaranteeLevel.STANDARD_93,
            min_accuracy=93.0,
            monitoring_intensity=MonitoringIntensity.HIGH
        )
        self.guarantee_system = AccuracyGuaranteeSystem(self.config)

    def test_accuracy_guarantee_validation_success(self):
        """精度保証検証（成功ケース）"""
        # 高精度データ
        performances = {
            "7203": PerformanceMetrics("7203", 95.5),
            "8306": PerformanceMetrics("8306", 94.2),
            "4751": PerformanceMetrics("4751", 93.8)
        }

        async def test():
            met, violations, overall = await self.guarantee_system.validate_accuracy_guarantee(performances)
            self.assertTrue(met)
            self.assertEqual(len(violations), 0)
            self.assertGreaterEqual(overall, 93.0)

        asyncio.run(test())

    def test_accuracy_guarantee_validation_failure(self):
        """精度保証検証（失敗ケース）"""
        # 低精度データ
        performances = {
            "7203": PerformanceMetrics("7203", 89.5),  # 違反
            "8306": PerformanceMetrics("8306", 94.2),
            "4751": PerformanceMetrics("4751", 91.8)   # 違反
        }

        async def test():
            met, violations, overall = await self.guarantee_system.validate_accuracy_guarantee(performances)
            self.assertFalse(met)
            self.assertEqual(len(violations), 2)
            self.assertIn("7203", violations)
            self.assertIn("4751", violations)

        asyncio.run(test())

    def test_violation_severity_assessment(self):
        """違反重要度評価テスト"""
        # 重大違反
        critical_severity = self.guarantee_system._assess_violation_severity(80.0)
        self.assertEqual(critical_severity, "critical")

        # 高重要度違反
        high_severity = self.guarantee_system._assess_violation_severity(87.0)
        self.assertEqual(high_severity, "high")

        # 中重要度違反
        medium_severity = self.guarantee_system._assess_violation_severity(92.0)
        self.assertEqual(medium_severity, "medium")

    def test_recovery_strategy_selection(self):
        """回復戦略選択テスト"""
        # 重大違反用戦略
        critical_strategy = self.guarantee_system._select_recovery_strategy("critical", ["7203"])
        self.assertEqual(critical_strategy.strategy_name, "emergency_global")
        self.assertEqual(critical_strategy.cooldown_hours, 0)

        # 高重要度違反用戦略
        high_strategy = self.guarantee_system._select_recovery_strategy("high", ["8306"])
        self.assertEqual(high_strategy.strategy_name, "priority_symbols")
        self.assertEqual(high_strategy.cooldown_hours, 6)


class TestAccuracyTrendAnalyzer(unittest.TestCase):
    """精度トレンド分析テスト"""

    def setUp(self):
        """テスト前準備"""
        self.analyzer = AccuracyTrendAnalyzer()

    def test_upward_trend_analysis(self):
        """上昇トレンド分析テスト"""
        # 上昇データ作成
        history = []
        base_time = datetime.now()
        for i in range(24):
            timestamp = base_time - timedelta(hours=24-i)
            accuracy = 90.0 + i * 0.2  # 徐々に上昇
            history.append((timestamp, accuracy))

        result = self.analyzer.analyze_accuracy_trend("7203", history)

        self.assertGreater(result["trend"], 0)  # 正のトレンド
        self.assertGreater(result["stability"], 0.5)  # 安定性確認
        self.assertGreater(result["prediction"], result["current"])  # 予測上昇

    def test_downward_trend_analysis(self):
        """下降トレンド分析テスト"""
        # 下降データ作成
        history = []
        base_time = datetime.now()
        for i in range(24):
            timestamp = base_time - timedelta(hours=24-i)
            accuracy = 95.0 - i * 0.1  # 徐々に下降
            history.append((timestamp, accuracy))

        result = self.analyzer.analyze_accuracy_trend("8306", history)

        self.assertLess(result["trend"], 0)  # 負のトレンド
        self.assertLess(result["prediction"], result["current"])  # 予測下降

    def test_stable_trend_analysis(self):
        """安定トレンド分析テスト"""
        # 安定データ作成
        history = []
        base_time = datetime.now()
        for i in range(24):
            timestamp = base_time - timedelta(hours=24-i)
            accuracy = 94.0 + (i % 2 - 0.5) * 0.1  # 微小変動
            history.append((timestamp, accuracy))

        result = self.analyzer.analyze_accuracy_trend("4751", history)

        self.assertAlmostEqual(result["trend"], 0.0, delta=1.0)  # ほぼゼロトレンド
        self.assertGreater(result["stability"], 0.8)  # 高安定性


class TestEmergencyDetector(unittest.TestCase):
    """緊急事態検出テスト"""

    def setUp(self):
        """テスト前準備"""
        self.detector = EmergencyDetector()

    def test_critical_accuracy_detection(self):
        """危険精度検出テスト"""
        metrics = ContinuousMonitoringMetrics(
            symbol="7203",
            timestamp=datetime.now(),
            accuracy_current=75.0,  # 危険レベル
            accuracy_trend=-3.0,
            prediction_confidence=0.4,
            sample_count=100,
            moving_average_24h=80.0,
            moving_average_7d=85.0,
            quality_status=PredictionQualityStatus.CRITICAL,
            degradation_rate=8.0,  # 急激な劣化
            stability_score=0.2  # 不安定
        )

        is_emergency, flags = self.detector.detect_emergency_conditions(metrics)

        self.assertTrue(is_emergency)
        self.assertIn("critical_accuracy", flags)
        self.assertIn("rapid_degradation", flags)
        self.assertIn("unstable_predictions", flags)
        self.assertIn("low_confidence", flags)

    def test_normal_conditions_detection(self):
        """正常状態検出テスト"""
        metrics = ContinuousMonitoringMetrics(
            symbol="8306",
            timestamp=datetime.now(),
            accuracy_current=95.0,  # 良好
            accuracy_trend=0.5,
            prediction_confidence=0.9,
            sample_count=150,
            moving_average_24h=94.5,
            moving_average_7d=94.0,
            quality_status=PredictionQualityStatus.EXCELLENT,
            degradation_rate=0.0,
            stability_score=0.9
        )

        is_emergency, flags = self.detector.detect_emergency_conditions(metrics)

        self.assertFalse(is_emergency)
        self.assertEqual(len(flags), 0)


class TestIntelligentRetrainingController(unittest.TestCase):
    """インテリジェント再学習制御テスト"""

    def setUp(self):
        """テスト前準備"""
        self.controller = IntelligentRetrainingController()

    def test_strategy_selection_critical(self):
        """重大状況での戦略選択テスト"""
        async def test():
            strategy = await self.controller.select_optimal_strategy(
                conditions=["critical_accuracy", "system_failure"],
                current_accuracy=75.0,
                available_resources=10
            )

            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.strategy_name, "emergency_global")
            self.assertEqual(strategy.priority, 1)

        asyncio.run(test())

    def test_strategy_selection_resource_constrained(self):
        """リソース制約下での戦略選択テスト"""
        async def test():
            strategy = await self.controller.select_optimal_strategy(
                conditions=["minor_degradation"],
                current_accuracy=91.0,
                available_resources=2  # 限定リソース
            )

            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.strategy_name, "incremental_update")
            self.assertLessEqual(strategy.resource_cost, 2)

        asyncio.run(test())

    def test_no_suitable_strategy(self):
        """適切な戦略なしテスト"""
        async def test():
            strategy = await self.controller.select_optimal_strategy(
                conditions=["unknown_condition"],
                current_accuracy=95.0,
                available_resources=1
            )

            self.assertIsNone(strategy)

        asyncio.run(test())


class TestContinuousPerformanceMonitor(unittest.TestCase):
    """連続性能監視テスト"""

    def setUp(self):
        """テスト前準備"""
        self.config = AccuracyGuaranteeConfig()
        self.monitor = ContinuousPerformanceMonitor(self.config)

    def test_monitoring_intervals(self):
        """監視間隔設定テスト"""
        # 高頻度監視設定
        self.config.monitoring_intensity = MonitoringIntensity.HIGH
        interval = self.monitor.monitoring_intervals[MonitoringIntensity.HIGH]
        self.assertEqual(interval, 300)  # 5分

        # 連続監視設定
        self.config.monitoring_intensity = MonitoringIntensity.CONTINUOUS
        interval = self.monitor.monitoring_intervals[MonitoringIntensity.CONTINUOUS]
        self.assertEqual(interval, 60)  # 1分

    @patch('enhanced_performance_monitor.ContinuousPerformanceMonitor._calculate_symbol_metrics')
    def test_metrics_collection(self, mock_calculate):
        """メトリクス収集テスト"""
        # モックメトリクス
        mock_metrics = ContinuousMonitoringMetrics(
            symbol="7203",
            timestamp=datetime.now(),
            accuracy_current=94.5,
            accuracy_trend=0.5,
            prediction_confidence=0.85,
            sample_count=120,
            moving_average_24h=94.0,
            moving_average_7d=93.5,
            quality_status=PredictionQualityStatus.GOOD
        )
        mock_calculate.return_value = mock_metrics

        async def test():
            symbols = ["7203", "8306"]
            metrics = await self.monitor._collect_continuous_metrics(symbols)

            self.assertEqual(len(metrics), 2)
            self.assertIn("7203", metrics)
            self.assertEqual(mock_calculate.call_count, 2)

        asyncio.run(test())

    def test_quality_status_determination(self):
        """品質状態判定テスト"""
        # 優秀レベル
        excellent = self.monitor._determine_quality_status(96.0)
        self.assertEqual(excellent, PredictionQualityStatus.EXCELLENT)

        # 良好レベル
        good = self.monitor._determine_quality_status(94.0)
        self.assertEqual(good, PredictionQualityStatus.GOOD)

        # 許容レベル
        acceptable = self.monitor._determine_quality_status(91.0)
        self.assertEqual(acceptable, PredictionQualityStatus.ACCEPTABLE)

        # 警告レベル
        warning = self.monitor._determine_quality_status(87.0)
        self.assertEqual(warning, PredictionQualityStatus.WARNING)

        # 危険レベル
        critical = self.monitor._determine_quality_status(82.0)
        self.assertEqual(critical, PredictionQualityStatus.CRITICAL)


class TestEnhancedPerformanceMonitorV2(unittest.TestCase):
    """強化版性能監視システム統合テスト"""

    def setUp(self):
        """テスト前準備"""
        # 一時設定ファイル作成
        self.temp_config = self._create_temp_config()
        self.monitor = EnhancedPerformanceMonitorV2(self.temp_config)

    def tearDown(self):
        """テスト後クリーンアップ"""
        if hasattr(self, 'temp_config') and Path(self.temp_config).exists():
            Path(self.temp_config).unlink()

    def _create_temp_config(self) -> str:
        """テスト用設定ファイル作成"""
        config = {
            "accuracy_guarantee": {
                "min_accuracy": 93.0,
                "target_accuracy": 95.0,
                "emergency_threshold": 85.0
            },
            "continuous_monitoring": {
                "intensity": "high",
                "interval_minutes": 5,
                "trend_analysis": True
            },
            "intelligent_retraining": {
                "auto_trigger": True,
                "resource_optimization": True,
                "strategy_selection": "adaptive"
            }
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        import yaml
        yaml.dump(config, temp_file, default_flow_style=False)
        temp_file.close()

        return temp_file.name

    def test_system_initialization(self):
        """システム初期化テスト"""
        self.assertIsNotNone(self.monitor.continuous_monitor)
        self.assertIsNotNone(self.monitor.retraining_controller)
        self.assertIsNotNone(self.monitor.config)

        # 設定値確認
        self.assertEqual(
            self.monitor.config["accuracy_guarantee"]["min_accuracy"],
            93.0
        )

    def test_configuration_loading(self):
        """設定読み込みテスト"""
        config = self.monitor.config

        # 精度保証設定
        self.assertIn("accuracy_guarantee", config)
        self.assertEqual(config["accuracy_guarantee"]["min_accuracy"], 93.0)

        # 連続監視設定
        self.assertIn("continuous_monitoring", config)
        self.assertEqual(config["continuous_monitoring"]["intensity"], "high")

        # インテリジェント再学習設定
        self.assertIn("intelligent_retraining", config)
        self.assertTrue(config["intelligent_retraining"]["auto_trigger"])

    @patch('enhanced_performance_monitor.ContinuousPerformanceMonitor.start_continuous_monitoring')
    def test_enhanced_monitoring_start(self, mock_start):
        """強化監視開始テスト"""
        mock_start.return_value = AsyncMock()

        async def test():
            symbols = ["7203", "8306", "4751"]

            # 短時間のテスト実行
            task = asyncio.create_task(
                self.monitor.start_enhanced_monitoring(symbols)
            )

            # 少し待機してキャンセル
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # 呼び出し確認
            mock_start.assert_called_once_with(symbols)

        asyncio.run(test())

    def test_comprehensive_report_generation(self):
        """包括レポート生成テスト"""
        async def test():
            report = await self.monitor.generate_comprehensive_report()

            # 必須フィールド確認
            self.assertIn("timestamp", report)
            self.assertIn("monitoring_status", report)
            self.assertIn("accuracy_guarantee", report)
            self.assertIn("continuous_metrics", report)
            self.assertIn("resource_usage", report)

            # 精度保証設定確認
            guarantee = report["accuracy_guarantee"]
            self.assertIn("level", guarantee)
            self.assertIn("min_accuracy", guarantee)
            self.assertEqual(guarantee["min_accuracy"], 93.0)

        asyncio.run(test())


class TestPerformanceIntegration(unittest.TestCase):
    """性能監視統合テスト"""

    def test_end_to_end_accuracy_monitoring(self):
        """エンドツーエンド精度監視テスト"""
        config = AccuracyGuaranteeConfig(
            min_accuracy=93.0,
            monitoring_intensity=MonitoringIntensity.HIGH
        )

        # システム初期化
        guarantee_system = AccuracyGuaranteeSystem(config)
        monitor = ContinuousPerformanceMonitor(config)

        # テストシナリオ: 精度低下→検出→回復
        async def test_scenario():
            # 1. 初期状態（良好）
            good_performances = {
                "7203": PerformanceMetrics("7203", 95.0),
                "8306": PerformanceMetrics("8306", 94.5)
            }

            met, violations, overall = await guarantee_system.validate_accuracy_guarantee(good_performances)
            self.assertTrue(met)
            self.assertEqual(len(violations), 0)

            # 2. 精度低下状態
            poor_performances = {
                "7203": PerformanceMetrics("7203", 89.0),  # 違反
                "8306": PerformanceMetrics("8306", 91.0)   # 境界
            }

            met, violations, overall = await guarantee_system.validate_accuracy_guarantee(poor_performances)
            self.assertFalse(met)
            self.assertGreater(len(violations), 0)

            # 3. 緊急回復トリガー
            recovery_result = await guarantee_system.trigger_guarantee_recovery(violations, overall)
            self.assertIsNotNone(recovery_result)

        asyncio.run(test_scenario())

    def test_monitoring_stress_test(self):
        """監視ストレステスト"""
        config = AccuracyGuaranteeConfig(
            monitoring_intensity=MonitoringIntensity.CONTINUOUS
        )
        monitor = ContinuousPerformanceMonitor(config)

        async def stress_test():
            # 大量銘柄での短時間監視
            symbols = [f"SYM{i:04d}" for i in range(20)]

            # モック化して高速実行
            with patch.object(monitor, '_calculate_symbol_metrics') as mock_calc:
                mock_calc.return_value = ContinuousMonitoringMetrics(
                    symbol="MOCK",
                    timestamp=datetime.now(),
                    accuracy_current=94.0,
                    accuracy_trend=0.0,
                    prediction_confidence=0.85,
                    sample_count=100,
                    moving_average_24h=94.0,
                    moving_average_7d=94.0,
                    quality_status=PredictionQualityStatus.GOOD
                )

                # 短時間監視実行
                task = asyncio.create_task(monitor.start_continuous_monitoring(symbols))
                await asyncio.sleep(0.5)  # 0.5秒監視
                monitor.stop_monitoring()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # 呼び出し数確認
                self.assertGreater(mock_calc.call_count, 0)

        asyncio.run(stress_test())


def run_enhanced_monitor_tests():
    """強化監視テスト実行"""
    print("=== 強化版性能監視システムテスト ===")

    # テストスイート作成
    suite = unittest.TestSuite()

    # 精度保証システムテスト
    suite.addTest(TestAccuracyGuaranteeSystem('test_accuracy_guarantee_validation_success'))
    suite.addTest(TestAccuracyGuaranteeSystem('test_accuracy_guarantee_validation_failure'))
    suite.addTest(TestAccuracyGuaranteeSystem('test_violation_severity_assessment'))
    suite.addTest(TestAccuracyGuaranteeSystem('test_recovery_strategy_selection'))

    # トレンド分析テスト
    suite.addTest(TestAccuracyTrendAnalyzer('test_upward_trend_analysis'))
    suite.addTest(TestAccuracyTrendAnalyzer('test_downward_trend_analysis'))
    suite.addTest(TestAccuracyTrendAnalyzer('test_stable_trend_analysis'))

    # 緊急検出テスト
    suite.addTest(TestEmergencyDetector('test_critical_accuracy_detection'))
    suite.addTest(TestEmergencyDetector('test_normal_conditions_detection'))

    # 再学習制御テスト
    suite.addTest(TestIntelligentRetrainingController('test_strategy_selection_critical'))
    suite.addTest(TestIntelligentRetrainingController('test_strategy_selection_resource_constrained'))
    suite.addTest(TestIntelligentRetrainingController('test_no_suitable_strategy'))

    # 連続監視テスト
    suite.addTest(TestContinuousPerformanceMonitor('test_monitoring_intervals'))
    suite.addTest(TestContinuousPerformanceMonitor('test_metrics_collection'))
    suite.addTest(TestContinuousPerformanceMonitor('test_quality_status_determination'))

    # 統合システムテスト
    suite.addTest(TestEnhancedPerformanceMonitorV2('test_system_initialization'))
    suite.addTest(TestEnhancedPerformanceMonitorV2('test_configuration_loading'))
    suite.addTest(TestEnhancedPerformanceMonitorV2('test_enhanced_monitoring_start'))
    suite.addTest(TestEnhancedPerformanceMonitorV2('test_comprehensive_report_generation'))

    # 統合テスト
    suite.addTest(TestPerformanceIntegration('test_end_to_end_accuracy_monitoring'))
    suite.addTest(TestPerformanceIntegration('test_monitoring_stress_test'))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print(f"\n=== 失敗したテスト ===")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\n=== エラーが発生したテスト ===")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # ログレベル設定
    logging.basicConfig(level=logging.WARNING)  # テスト時は警告以上のみ

    success = run_enhanced_monitor_tests()
    if success:
        print("\n✅ 全テストが正常に完了しました")
        print("93%精度維持保証システムが正常に動作しています")
    else:
        print("\n❌ 一部のテストが失敗しました")
        exit(1)