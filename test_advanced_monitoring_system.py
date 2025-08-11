#!/usr/bin/env python3
"""
高度監視システム テストスイート
Phase G: 本番運用最適化フェーズ

監視・アラート・メトリクス収集システムの包括的テスト
"""

import json
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.day_trade.monitoring.advanced_monitoring_system import (
        AdvancedMonitoringSystem,
        AlertLevel,
        AlertManager,
        MetricsCollector,
        MetricType,
        MonitoringRule,
        RuleEngine,
    )
    from src.day_trade.monitoring.prometheus_integration import (
        ApplicationMetricsCollector,
        PrometheusExporter,
    )
except ImportError as e:
    print(f"[ERROR] インポートエラー: {e}")
    print("依存関係が不足している可能性があります。基本テストのみ実行します。")
    # 基本的なテストのみ実行
    FULL_TEST = False
else:
    FULL_TEST = True


class MockNotificationChannel:
    """モック通知チャンネル"""

    def __init__(self):
        self.sent_alerts = []

    def send_alert(self, alert):
        self.sent_alerts.append(alert)
        print(f"[MOCK] アラート送信: {alert.title} ({alert.level.value})")


class MonitoringSystemTester:
    """監視システムテスター"""

    def __init__(self):
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }

        print("=" * 80)
        print("[TEST] 高度監視システム テストスイート")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)

    def run_test(self, test_name: str, test_func):
        """テスト実行"""
        print(f"\n[TEST] {test_name} 実行中...")
        self.test_results['total_tests'] += 1

        try:
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time

            if result:
                print(f"[OK] {test_name} 成功 ({execution_time:.3f}秒)")
                self.test_results['passed_tests'] += 1
                status = "PASSED"
            else:
                print(f"[FAIL] {test_name} 失敗 ({execution_time:.3f}秒)")
                self.test_results['failed_tests'] += 1
                status = "FAILED"

            self.test_results['test_details'].append({
                'name': test_name,
                'status': status,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"[ERROR] {test_name} エラー: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'name': test_name,
                'status': "ERROR",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    def test_metrics_collector(self) -> bool:
        """メトリクス収集テスト"""
        collector = MetricsCollector()

        # メトリクス記録テスト
        collector.record_metric("test_counter", 10.0, MetricType.COUNTER, {"app": "test"})
        collector.record_metric("test_gauge", 50.5, MetricType.GAUGE, {"type": "cpu"})

        # データ確認
        if "test_counter" not in collector.metrics:
            return False

        if len(collector.metrics["test_counter"]) != 1:
            return False

        point = collector.metrics["test_counter"][0]
        if point.value != 10.0 or point.labels.get("app") != "test":
            return False

        # システムメトリクス収集テスト
        collector.collect_system_metrics()

        # 基本システムメトリクスの存在確認
        expected_metrics = [
            "system.cpu_usage_percent",
            "system.memory_usage_percent",
            "system.disk_usage_percent"
        ]

        for metric in expected_metrics:
            if metric not in collector.metrics or len(collector.metrics[metric]) == 0:
                return False

        # メトリクスサマリーテスト
        summary = collector.get_metric_summary("test_counter")
        return not (not summary or summary.get('count') != 1)

    def test_alert_manager(self) -> bool:
        """アラート管理テスト"""
        alert_manager = AlertManager()
        mock_channel = MockNotificationChannel()
        alert_manager.add_notification_channel(mock_channel)

        # アラート作成テスト
        alert_id = alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="テストアラート",
            message="これはテスト用のアラートです",
            source_component="TestModule",
            metadata={"test_key": "test_value"}
        )

        if not alert_id:
            return False

        # 通知送信確認
        if len(mock_channel.sent_alerts) != 1:
            return False

        sent_alert = mock_channel.sent_alerts[0]
        if (sent_alert.title != "テストアラート" or
            sent_alert.level != AlertLevel.WARNING):
            return False

        # アクティブアラート取得テスト
        active_alerts = alert_manager.get_active_alerts()
        if len(active_alerts) != 1:
            return False

        # アラート解決テスト
        alert_manager.resolve_alert(alert_id)
        active_alerts = alert_manager.get_active_alerts()
        return len(active_alerts) == 0

    def test_rule_engine(self) -> bool:
        """ルールエンジンテスト"""
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        mock_channel = MockNotificationChannel()
        alert_manager.add_notification_channel(mock_channel)

        rule_engine = RuleEngine(metrics_collector, alert_manager)

        # テストルール追加
        test_rule = MonitoringRule(
            rule_id="test_high_cpu",
            name="テスト高CPU使用率",
            condition="cpu_usage > 75",
            alert_level=AlertLevel.WARNING,
            cooldown_seconds=1
        )
        rule_engine.add_rule(test_rule)

        # 高いCPU使用率をシミュレート
        metrics_collector.record_metric("system.cpu_usage_percent", 80.0)

        # ルール評価
        rule_engine.evaluate_rules()

        # アラートが作成されたか確認
        if len(mock_channel.sent_alerts) != 1:
            return False

        alert = mock_channel.sent_alerts[0]
        if alert.level != AlertLevel.WARNING or "高CPU使用率" not in alert.title:
            return False

        # クールダウンテスト
        rule_engine.evaluate_rules()  # 再評価（クールダウン中）
        if len(mock_channel.sent_alerts) != 1:  # 追加アラートなし
            return False

        return True

    def test_monitoring_system_integration(self) -> bool:
        """監視システム統合テスト"""
        system = AdvancedMonitoringSystem()

        # 監視開始
        system.start_monitoring()
        time.sleep(2)  # 短時間監視実行

        # システムダッシュボード確認
        dashboard = system.get_system_dashboard()

        if not dashboard.get('monitoring_status', {}).get('running'):
            return False

        if dashboard.get('monitoring_status', {}).get('rules_count', 0) == 0:
            return False

        # 手動アラートテスト
        alert_id = system.create_manual_alert(
            AlertLevel.INFO,
            "統合テストアラート",
            "統合テスト実行中"
        )

        if not alert_id:
            return False

        # アプリケーションメトリクス記録テスト
        system.record_application_metric(
            "test_api_requests", 100.0, {"endpoint": "/api/test"}
        )

        # 監視停止
        system.stop_monitoring()

        return True

    def test_prometheus_integration(self) -> bool:
        """Prometheus統合テスト"""
        exporter = PrometheusExporter(port=9091)  # テスト用ポート
        collector = ApplicationMetricsCollector(exporter)

        try:
            # サーバー開始
            exporter.start_server()
            time.sleep(1)  # サーバー起動待機

            # メトリクス記録
            collector.record_api_call("GET", "/test", 200, 0.1)
            exporter.record_ml_prediction("TestModel", 0.05, 0.9)
            exporter.update_system_metrics(60.0, 1024.0, 10)

            # メトリクステキスト生成テスト
            metrics_text = exporter.registry.generate_metrics_text()

            if "daytrade_requests_total" not in metrics_text:
                return False

            if "daytrade_cpu_usage_percent" not in metrics_text:
                return False

            # 操作テスト
            op_id = collector.start_operation("test_op")
            time.sleep(0.01)
            collector.end_operation(op_id, "test", "TEST", True)

            return True

        except Exception as e:
            print(f"Prometheusテストエラー: {e}")
            return False
        finally:
            exporter.stop_server()

    def test_performance_under_load(self) -> bool:
        """負荷テスト"""
        system = AdvancedMonitoringSystem()

        start_time = time.time()

        # 大量メトリクス記録
        for i in range(1000):
            system.record_application_metric(
                f"load_test_metric_{i % 10}",
                float(i),
                {"batch": str(i // 100)}
            )

        # 大量アラート作成
        for i in range(50):
            system.create_manual_alert(
                AlertLevel.INFO,
                f"負荷テストアラート {i}",
                f"負荷テスト用アラート番号 {i}"
            )

        execution_time = time.time() - start_time

        # 5秒以内で完了することを確認
        if execution_time > 5.0:
            print(f"負荷テスト時間超過: {execution_time:.2f}秒")
            return False

        # ダッシュボード取得テスト
        dashboard = system.get_system_dashboard()
        return dashboard

    def test_configuration_validation(self) -> bool:
        """設定検証テスト"""
        # 無効なルール設定テスト
        try:
            invalid_rule = MonitoringRule(
                rule_id="invalid",
                name="無効ルール",
                condition="invalid_syntax >>>",  # 無効な構文
                alert_level=AlertLevel.ERROR
            )

            metrics_collector = MetricsCollector()
            alert_manager = AlertManager()
            rule_engine = RuleEngine(metrics_collector, alert_manager)
            rule_engine.add_rule(invalid_rule)

            # 評価実行（エラーが発生するが継続すること）
            rule_engine.evaluate_rules()

            return True  # エラーハンドリングが正常に動作

        except Exception as e:
            print(f"設定検証テストエラー: {e}")
            return False

    def run_all_tests(self):
        """全テスト実行"""
        print("\n[START] 監視システム包括テスト開始")

        if not FULL_TEST:
            print("[WARNING] 依存関係不足のため、制限されたテストを実行します")
            return False

        # テスト実行
        test_methods = [
            ("メトリクス収集システム", self.test_metrics_collector),
            ("アラート管理システム", self.test_alert_manager),
            ("ルールエンジン", self.test_rule_engine),
            ("監視システム統合", self.test_monitoring_system_integration),
            ("Prometheus統合", self.test_prometheus_integration),
            ("負荷テスト", self.test_performance_under_load),
            ("設定検証", self.test_configuration_validation)
        ]

        for test_name, test_method in test_methods:
            self.run_test(test_name, test_method)
            time.sleep(0.5)  # テスト間の待機

        # 結果サマリー
        self._print_test_summary()

        return self.test_results['failed_tests'] == 0

    def _print_test_summary(self):
        """テスト結果サマリー表示"""
        print("\n" + "=" * 80)
        print("[SUMMARY] テスト結果サマリー")
        print("=" * 80)
        print(f"実行テスト数: {self.test_results['total_tests']}")
        print(f"成功: {self.test_results['passed_tests']}")
        print(f"失敗: {self.test_results['failed_tests']}")

        success_rate = (self.test_results['passed_tests'] /
                       self.test_results['total_tests']) * 100
        print(f"成功率: {success_rate:.1f}%")

        if self.test_results['failed_tests'] == 0:
            print("\n[OK] 全テスト成功")
        else:
            print(f"\n[WARNING] {self.test_results['failed_tests']} テスト失敗")

        # 詳細結果をJSONで保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"monitoring_test_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        print(f"\n[REPORT] 詳細結果保存: {results_file}")


def main():
    """メインテスト実行"""
    try:
        tester = MonitoringSystemTester()
        success = tester.run_all_tests()

        exit_code = 0 if success else 1
        print(f"\n[COMPLETE] 監視システムテスト完了 (終了コード: {exit_code})")

        return exit_code

    except Exception as e:
        print(f"\n[ERROR] テスト実行エラー: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
