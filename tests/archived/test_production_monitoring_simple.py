#!/usr/bin/env python3
"""
本番運用監視システム簡単テスト
Issue #436: 本番運用監視システム完成
"""

import asyncio
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from day_trade.monitoring.production_monitoring_system import (
    AlertSeverity,
    MonitoringScope,
    ProductionMonitoringSystem,
)


def test_basic_functionality():
    """基本機能テスト"""
    print("=== Basic Functionality Test ===")

    # システム初期化
    monitoring = ProductionMonitoringSystem()
    print("OK: ProductionMonitoringSystem initialized")

    # 監視開始
    monitoring.start_monitoring()
    print("OK: Monitoring system started")

    # メトリクス記録テスト
    monitoring.record_metric("test.cpu_usage", 65.5, {"host": "server1"})
    monitoring.record_metric("test.memory_usage", 78.2, {"host": "server1"})
    monitoring.record_metric("test.response_time", 25.3, {"endpoint": "/api/test"})
    print("OK: Metrics recorded")

    # API呼び出し記録テスト
    monitoring.record_api_call("/api/trades", "GET", 23.5, 200, True)
    monitoring.record_api_call("/api/trades", "POST", 45.2, 201, True)
    monitoring.record_api_call("/api/orders", "GET", 89.1, 500, False)
    print("OK: API calls recorded")

    # 少し待機
    time.sleep(2)

    # ダッシュボードデータ取得
    dashboard = monitoring.get_dashboard_data()
    print("OK: Dashboard data retrieved")

    monitoring.stop_monitoring()
    print("OK: Monitoring system stopped")

    return dashboard


def test_tracing():
    """トレーシングテスト"""
    print("\n=== Distributed Tracing Test ===")

    monitoring = ProductionMonitoringSystem()
    tracer = monitoring.tracer

    # 分散トレース実行
    main_span = tracer.start_trace("user_login_flow")
    tracer.add_span_tag(main_span, "user_id", "user123")

    # 認証フェーズ
    auth_span = tracer.start_trace("authentication", main_span)
    tracer.add_span_log(auth_span, "Authentication started")
    time.sleep(0.1)
    tracer.add_span_log(auth_span, "Authentication completed")
    tracer.finish_span(auth_span, "ok")

    # メインスパン終了
    tracer.finish_span(main_span, "ok")

    print("OK: Distributed trace executed")

    dashboard = monitoring.get_dashboard_data()
    trace_info = dashboard["traces"]

    print(f"Active traces: {trace_info['active_traces']}")
    print(f"Completed traces: {trace_info['completed_traces']}")

    return trace_info


def test_slo_monitoring():
    """SLO監視テスト"""
    print("\n=== SLO Monitoring Test ===")

    monitoring = ProductionMonitoringSystem()
    slo_manager = monitoring.slo_manager

    # SLOメトリクス記録
    for i in range(50):
        response_time = 20 + (i % 10)
        success = response_time < 50
        slo_manager.add_slo_metric("api_latency", response_time, success)

    print("OK: SLO metrics recorded")

    # SLO状態確認
    api_latency_status = slo_manager.get_slo_status("api_latency")

    if api_latency_status:
        print(
            f"API Latency: {api_latency_status.current_percentage:.2f}% "
            f"(target: {api_latency_status.config.target_percentage:.2f}%) "
            f"status: {api_latency_status.status}"
        )

    return api_latency_status


async def main():
    """メインテスト実行"""
    print("Production Monitoring System Test")
    print("=" * 50)

    try:
        # 基本機能テスト
        dashboard = test_basic_functionality()

        # トレーシングテスト
        trace_info = test_tracing()

        # SLO監視テスト
        slo_status = test_slo_monitoring()

        # 結果サマリー
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)

        print("System Health:")
        print(f"  Status: {dashboard['system_health']['status']}")
        print(f"  Active: {dashboard['system_health']['monitoring_active']}")

        print("Statistics:")
        print(f"  Total requests: {dashboard['statistics']['total_requests']}")
        print(
            f"  Success rate: {dashboard['statistics']['successful_requests']/max(1,dashboard['statistics']['total_requests']):.1%}"
        )
        print(f"  Alerts triggered: {dashboard['statistics']['alerts_triggered']}")
        print(f"  Anomalies detected: {dashboard['statistics']['anomalies_detected']}")

        print("Traces:")
        print(f"  Active traces: {trace_info['active_traces']}")
        print(f"  Completed traces: {trace_info['completed_traces']}")

        if slo_status:
            print("SLO Status:")
            print(
                f"  API Latency: {slo_status.current_percentage:.1f}% ({slo_status.status})"
            )

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
