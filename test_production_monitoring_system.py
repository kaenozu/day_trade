#!/usr/bin/env python3
"""
本番運用監視システムテスト
Issue #436: 本番運用監視システム完成

24/7安定運用のためのAPM・オブザーバビリティ統合基盤のテスト
"""

import asyncio
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from day_trade.monitoring.production_monitoring_system import (
    ProductionMonitoringSystem,
    AlertSeverity,
    MonitoringScope,
)


def test_basic_functionality():
    """基本機能テスト"""
    print("=== 基本機能テスト ===")

    # システム初期化
    monitoring = ProductionMonitoringSystem()
    print("✓ ProductionMonitoringSystemの初期化")

    # 監視開始
    monitoring.start_monitoring()
    print("✓ 監視システム開始")

    # メトリクス記録テスト
    monitoring.record_metric("test.cpu_usage", 65.5, {"host": "server1"})
    monitoring.record_metric("test.memory_usage", 78.2, {"host": "server1"})
    monitoring.record_metric("test.response_time", 25.3, {"endpoint": "/api/test"})
    print("✓ メトリクス記録")

    # API呼び出し記録テスト
    monitoring.record_api_call("/api/trades", "GET", 23.5, 200, True)
    monitoring.record_api_call("/api/trades", "POST", 45.2, 201, True)
    monitoring.record_api_call("/api/orders", "GET", 89.1, 500, False)  # エラー
    print("✓ API呼び出し記録")

    # 構造化ログ記録テスト
    monitoring.log_structured("INFO", "システム正常稼働", "SystemMonitor",
                             cpu_usage=65.5, memory_usage=78.2)
    monitoring.log_structured("WARNING", "レスポンス時間増加", "APIGateway",
                             endpoint="/api/orders", response_time=89.1)
    monitoring.log_structured("ERROR", "データベース接続エラー", "DatabaseManager",
                             error_code="DB001", retry_count=3)
    print("✓ 構造化ログ記録")

    # 少し待機してから結果確認
    time.sleep(2)

    # ダッシュボードデータ取得
    dashboard = monitoring.get_dashboard_data()
    print("✓ ダッシュボードデータ取得")

    # メトリクスサマリー取得
    cpu_summary = monitoring.get_metrics_summary("test.cpu_usage")
    response_time_summary = monitoring.get_metrics_summary("test.response_time")
    print("✓ メトリクスサマリー取得")

    # 最近のログ取得
    recent_logs = monitoring.get_recent_logs(limit=10)
    error_logs = monitoring.get_recent_logs(level="ERROR", limit=5)
    print("✓ ログ検索機能")

    monitoring.stop_monitoring()
    print("✓ 監視システム停止")

    return {
        'dashboard': dashboard,
        'cpu_summary': cpu_summary,
        'response_time_summary': response_time_summary,
        'recent_logs_count': len(recent_logs),
        'error_logs_count': len(error_logs)
    }


async def test_distributed_tracing():
    """分散トレーシングテスト"""
    print("\n=== 分散トレーシングテスト ===")

    monitoring = ProductionMonitoringSystem()

    # 分散トレース実行
    async with monitoring.trace_operation("user_login_flow") as main_span:
        monitoring.tracer.add_span_tag(main_span, "user_id", "user123")
        monitoring.tracer.add_span_tag(main_span, "client_ip", "192.168.1.100")

        # 認証フェーズ
        async with monitoring.trace_operation("authentication", main_span) as auth_span:
            monitoring.tracer.add_span_log(auth_span, "認証処理開始")
            await asyncio.sleep(0.1)  # 模擬処理時間
            monitoring.tracer.add_span_tag(auth_span, "auth_method", "password")
            monitoring.tracer.add_span_log(auth_span, "認証処理完了")

        # セッション作成フェーズ
        async with monitoring.trace_operation("session_creation", main_span) as session_span:
            monitoring.tracer.add_span_log(session_span, "セッション作成開始")
            await asyncio.sleep(0.05)
            monitoring.tracer.add_span_tag(session_span, "session_id", "sess_456")
            monitoring.tracer.add_span_log(session_span, "セッション作成完了")

        # ユーザーデータ取得フェーズ
        async with monitoring.trace_operation("user_data_fetch", main_span) as data_span:
            monitoring.tracer.add_span_log(data_span, "ユーザーデータ取得開始")
            await asyncio.sleep(0.08)
            monitoring.tracer.add_span_tag(data_span, "database", "user_db")
            monitoring.tracer.add_span_log(data_span, "ユーザーデータ取得完了")

    print("✓ 分散トレース実行完了")

    # トレース結果確認
    dashboard = monitoring.get_dashboard_data()
    trace_info = dashboard['traces']

    print(f"✓ アクティブトレース: {trace_info['active_traces']}")
    print(f"✓ 完了トレース: {trace_info['completed_traces']}")

    return trace_info


def test_slo_monitoring():
    """SLO監視テスト"""
    print("\n=== SLO監視テスト ===")

    monitoring = ProductionMonitoringSystem()
    slo_manager = monitoring.slo_manager

    # SLOメトリクス記録 - 成功ケース
    for i in range(50):
        response_time = 20 + (i % 10)  # 20-30ms
        success = response_time < 50  # 50ms以下を成功とする
        slo_manager.add_slo_metric("api_latency", response_time, success)

    # SLOメトリクス記録 - 異常ケース
    for i in range(5):
        response_time = 80 + (i * 10)  # 80-120ms（異常値）
        success = False
        slo_manager.add_slo_metric("api_latency", response_time, success)

    print("✓ SLOメトリクス記録")

    # SLO状態確認
    api_latency_status = slo_manager.get_slo_status("api_latency")
    system_availability_status = slo_manager.get_slo_status("system_availability")
    all_slo_status = slo_manager.get_all_slo_status()

    print("✓ SLO状態取得")

    if api_latency_status:
        print(f"  - API レイテンシー: {api_latency_status.current_percentage:.2f}% "
              f"(目標: {api_latency_status.config.target_percentage:.2f}%) "
              f"状態: {api_latency_status.status}")
        print(f"  - エラーバジェット消費: {api_latency_status.error_budget_consumed:.1f}%")

    return {
        'api_latency_status': api_latency_status,
        'system_availability_status': system_availability_status,
        'total_slos': len(all_slo_status)
    }


def test_anomaly_detection():
    """異常検知テスト"""
    print("\n=== 異常検知テスト ===")

    monitoring = ProductionMonitoringSystem()
    anomaly_detector = monitoring.anomaly_detector

    # 正常データを大量に追加（学習用）
    print("正常データ学習中...")
    for i in range(100):
        # CPU使用率: 40-60%の正常範囲
        normal_cpu = 50 + (i % 20) - 10
        anomaly_detector.add_metric_point("cpu_usage", normal_cpu)

        # メモリ使用率: 60-80%の正常範囲
        normal_memory = 70 + (i % 20) - 10
        anomaly_detector.add_metric_point("memory_usage", normal_memory)

    print("✓ 正常データ学習完了")

    # 異常値テスト
    test_cases = [
        ("cpu_usage", 95.0, "高CPU使用率"),
        ("cpu_usage", 5.0, "低CPU使用率"),
        ("memory_usage", 98.0, "高メモリ使用率"),
        ("memory_usage", 25.0, "低メモリ使用率"),
        ("cpu_usage", 55.0, "正常CPU使用率"),
        ("memory_usage", 75.0, "正常メモリ使用率"),
    ]

    anomaly_results = []
    for metric_name, value, description in test_cases:
        is_anomaly, score = anomaly_detector.detect_anomaly(metric_name, value)
        anomaly_results.append({
            'metric': metric_name,
            'value': value,
            'description': description,
            'is_anomaly': is_anomaly,
            'score': score
        })
        print(f"  - {description}: 値={value}, 異常={is_anomaly}, スコア={score:.3f}")

    print("✓ 異常検知テスト完了")

    return anomaly_results


def test_comprehensive_monitoring():
    """包括的監視テスト"""
    print("\n=== 包括的監視テスト ===")

    monitoring = ProductionMonitoringSystem()
    monitoring.start_monitoring()

    # 複数の監視項目を同時にテスト
    test_scenarios = [
        # 正常ケース
        {"api": "/api/users", "response_time": 25.0, "status": 200, "success": True},
        {"api": "/api/trades", "response_time": 35.0, "status": 200, "success": True},
        {"api": "/api/orders", "response_time": 42.0, "status": 201, "success": True},

        # 警告ケース
        {"api": "/api/analytics", "response_time": 65.0, "status": 200, "success": True},
        {"api": "/api/reports", "response_time": 78.0, "status": 200, "success": True},

        # エラーケース
        {"api": "/api/trades", "response_time": 120.0, "status": 500, "success": False},
        {"api": "/api/orders", "response_time": 95.0, "status": 503, "success": False},
    ]

    for scenario in test_scenarios:
        monitoring.record_api_call(
            scenario["api"], "GET",
            scenario["response_time"],
            scenario["status"],
            scenario["success"]
        )

        # カスタムメトリクス
        monitoring.record_metric("custom.trade_volume", 1000 + (scenario["response_time"] * 100))
        monitoring.record_metric("custom.active_users", 500 + int(scenario["response_time"]))

        # ログ記録
        log_level = "ERROR" if not scenario["success"] else "WARNING" if scenario["response_time"] > 60 else "INFO"
        monitoring.log_structured(
            log_level,
            f"API処理: {scenario['api']}",
            "APIGateway",
            response_time=scenario["response_time"],
            status_code=scenario["status"],
            endpoint=scenario["api"]
        )

    # 少し待機して処理完了を待つ
    time.sleep(3)

    # 結果確認
    dashboard = monitoring.get_dashboard_data()

    print("✓ 包括的監視テスト完了")
    print(f"  - システムヘルス: {dashboard['system_health']['status']}")
    print(f"  - 総リクエスト: {dashboard['statistics']['total_requests']}")
    print(f"  - 成功率: {dashboard['statistics']['successful_requests'] / max(1, dashboard['statistics']['total_requests']):.1%}")
    print(f"  - 平均応答時間: {dashboard['statistics']['avg_response_time']:.2f}ms")
    print(f"  - アラート数: {dashboard['alerts']['total']}")
    print(f"  - 異常検知: {dashboard['statistics']['anomalies_detected']}")

    monitoring.stop_monitoring()

    return dashboard


async def main():
    """メインテスト実行"""
    print("本番運用監視システム 統合テスト開始")
    print("=" * 60)

    try:
        # 基本機能テスト
        basic_results = test_basic_functionality()

        # 分散トレーシングテスト
        trace_results = await test_distributed_tracing()

        # SLO監視テスト
        slo_results = test_slo_monitoring()

        # 異常検知テスト
        anomaly_results = test_anomaly_detection()

        # 包括的監視テスト
        comprehensive_results = test_comprehensive_monitoring()

        # 結果サマリー
        print("\n" + "=" * 60)
        print("テスト結果サマリー")
        print("=" * 60)

        print("✅ 基本機能:")
        print(f"   - ダッシュボード: {basic_results['dashboard']['system_health']['status']}")
        print(f"   - CPU監視: {len(basic_results.get('cpu_summary', {}))}項目")
        print(f"   - ログ収集: {basic_results['recent_logs_count']}件")

        print("✅ 分散トレーシング:")
        print(f"   - アクティブトレース: {trace_results['active_traces']}")
        print(f"   - 完了トレース: {trace_results['completed_traces']}")

        print("✅ SLO監視:")
        if slo_results['api_latency_status']:
            print(f"   - APIレイテンシー: {slo_results['api_latency_status'].current_percentage:.1f}%")
            print(f"   - SLO状態: {slo_results['api_latency_status'].status}")
        print(f"   - 監視SLO数: {slo_results['total_slos']}")

        print("✅ 異常検知:")
        anomaly_detected = sum(1 for r in anomaly_results if r['is_anomaly'])
        print(f"   - 異常検知数: {anomaly_detected}/{len(anomaly_results)}")

        print("✅ 包括的監視:")
        print(f"   - システムヘルス: {comprehensive_results['system_health']['status']}")
        print(f"   - 総処理リクエスト: {comprehensive_results['statistics']['total_requests']}")
        print(f"   - 発生アラート: {comprehensive_results['alerts']['total']}")

        print("\n🎉 すべてのテストが正常に完了しました!")
        return True

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)