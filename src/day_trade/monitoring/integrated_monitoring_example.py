"""
統合監視システム使用例

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード
ELK Stack風ログ集約、Prometheus/Grafana風メトリクス・ダッシュボード、
アラートシステム、パフォーマンス最適化推奨システムの使用例
"""

import asyncio
from datetime import datetime, timedelta

from .alert_system import create_default_alert_rules, get_alert_manager
from .log_aggregation_system import create_log_aggregation_system
from .metrics_collection_system import get_metrics_system
from .performance_dashboard import DashboardTemplates, get_dashboard_manager
from .performance_optimization_system import get_optimization_manager


async def initialize_integrated_monitoring():
    """統合監視システムの初期化"""
    print("統合監視システムを初期化中...")

    # 1. ログ集約システム
    log_system = create_log_aggregation_system()
    await log_system.start_processing()
    print("[OK] ログ集約システム開始")

    # 2. メトリクス収集システム
    metrics_system = get_metrics_system()
    metrics_system.start()
    print("[OK] メトリクス収集システム開始")

    # 3. ダッシュボードシステム
    dashboard_manager = get_dashboard_manager()
    templates = DashboardTemplates()

    # デフォルトダッシュボード作成
    system_dashboard = templates.create_system_overview_dashboard()
    app_dashboard = templates.create_application_performance_dashboard()
    ml_dashboard = templates.create_ml_model_dashboard()

    dashboard_manager.save_dashboard(system_dashboard)
    dashboard_manager.save_dashboard(app_dashboard)
    dashboard_manager.save_dashboard(ml_dashboard)
    print("[OK] パフォーマンスダッシュボード設定完了")

    # 4. アラートシステム
    alert_manager = get_alert_manager()

    # デフォルトアラートルール設定
    default_rules = create_default_alert_rules()
    for rule in default_rules:
        alert_manager.add_alert_rule(rule)

    await alert_manager.start()
    print("[OK] アラートシステム開始")

    # 5. パフォーマンス最適化システム
    optimization_manager = get_optimization_manager()
    print("[OK] パフォーマンス最適化システム準備完了")

    return {
        "log_system": log_system,
        "metrics_system": metrics_system,
        "dashboard_manager": dashboard_manager,
        "alert_manager": alert_manager,
        "optimization_manager": optimization_manager
    }


async def demo_log_analysis():
    """ログ分析のデモ"""
    print("\n[INFO] ログ分析デモ")

    log_system = create_log_aggregation_system()

    # サンプルログの送信
    sample_logs = [
        "INFO: システム起動完了",
        "WARNING: 高CPU使用率検出 85%",
        "ERROR: データベース接続失敗",
        "INFO: ユーザーログイン: user123",
        "CRITICAL: メモリ不足エラー",
        "ERROR: SQLインジェクション試行検出"
    ]

    for log_msg in sample_logs:
        await log_system.ingest_log(log_msg)

    # ログ分析実行
    from .log_aggregation_system import LogSearchQuery

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)

    query = LogSearchQuery(
        start_time=start_time,
        end_time=end_time,
        limit=100
    )
    logs = await log_system.search_logs(query)

    print(f"  - 分析対象ログ: {len(logs)}件")

    if logs:
        latest_log = logs[0]
        print(f"  - 最新ログサンプル: {getattr(latest_log, 'message', 'N/A')}")
    else:
        print("  - ログが見つかりませんでした")


async def demo_metrics_collection():
    """メトリクス収集のデモ"""
    print("\n[INFO] メトリクス収集デモ")

    metrics_system = get_metrics_system()

    # HTTPリクエストの記録
    from .metrics_collection_system import record_http_request, record_ml_prediction

    record_http_request("/api/trades", "GET", 0.123, 200)
    record_http_request("/api/users", "POST", 0.456, 201)
    record_http_request("/api/orders", "GET", 2.1, 500)

    # ML予測の記録
    record_ml_prediction("price_predictor", 0.087, 0.95)
    record_ml_prediction("risk_analyzer", 0.234, 0.87)

    print("  - HTTPリクエストとML予測メトリクス記録完了")

    # Prometheusフォーマットでエクスポート
    prometheus_output = metrics_system.get_metrics_prometheus()
    print(f"  - Prometheusメトリクス: {len(prometheus_output.split('\\n'))}行")


async def demo_dashboard_data():
    """ダッシュボードデータの取得デモ"""
    print("\n[INFO] ダッシュボードデータ取得デモ")

    dashboard_manager = get_dashboard_manager()

    # システム概要ダッシュボードのデータ取得
    try:
        system_data = await dashboard_manager.get_dashboard_data("system_overview")
        print(f"  - システム概要ダッシュボード: {len(system_data['widgets'])}個のウィジェット")

        for widget in system_data['widgets']:
            series_count = len(widget['data']['series'])
            print(f"    * {widget['title']}: {series_count}系列のデータ")

    except Exception as e:
        print(f"  - ダッシュボードデータ取得エラー: {e}")


async def demo_alert_monitoring():
    """アラート監視のデモ"""
    print("\n[INFO] アラート監視デモ")

    alert_manager = get_alert_manager()

    # アクティブアラート確認
    active_alerts = alert_manager.get_active_alerts()
    print(f"  - アクティブアラート: {len(active_alerts)}件")

    # アラート履歴
    alert_history = alert_manager.get_alert_history(limit=5)
    print(f"  - アラート履歴: {len(alert_history)}件")

    if alert_history:
        latest_alert = alert_history[0]
        print(f"    * 最新: {latest_alert.rule_name} [{latest_alert.severity.value}]")


async def demo_performance_optimization():
    """パフォーマンス最適化のデモ"""
    print("\n[INFO] パフォーマンス最適化デモ")

    optimization_manager = get_optimization_manager()

    # 分析と推奨事項生成
    recommendations = await optimization_manager.analyze_and_recommend()
    print(f"  - 最適化推奨事項: {len(recommendations)}件")

    for recommendation in recommendations[:3]:
        print(f"    * {recommendation.title} [{recommendation.priority.value}]")
        print(f"      実装工数: {recommendation.implementation_effort}/10")
        improvement_summary = ", ".join([f"{k}: {v}%" for k, v in recommendation.estimated_improvement.items()])
        if improvement_summary:
            print(f"      期待効果: {improvement_summary}")

    # 最適化レポート
    report = optimization_manager.get_optimization_report()
    print(f"  - 総推奨事項: {report['total_recommendations']}件")
    print(f"  - 実装済み: {report['implemented_recommendations']}件")
    print(f"  - 実装率: {report['implementation_rate']:.1f}%")


async def demo_integrated_monitoring():
    """統合監視システム全体のデモ"""
    print("=" * 60)
    print("統合監視システム デモンストレーション")
    print("Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード")
    print("=" * 60)

    try:
        # システム初期化
        systems = await initialize_integrated_monitoring()
        print("\n[OK] 全システム初期化完了")

        # 各機能のデモ実行
        await demo_log_analysis()
        await demo_metrics_collection()
        await demo_dashboard_data()
        await demo_alert_monitoring()
        await demo_performance_optimization()

        print("\n" + "=" * 60)
        print("[SUCCESS] 統合監視システム デモ完了")
        print("ELK Stack風ログ集約、Prometheus/Grafana風メトリクス・ダッシュボード、")
        print("アラートシステム、AI駆動パフォーマンス最適化が連携動作中")
        print("=" * 60)

        return systems

    except Exception as e:
        print(f"\n[ERROR] デモ実行エラー: {e}")
        raise
    finally:
        # クリーンアップ
        try:
            if 'systems' in locals():
                await systems['log_system'].stop_processing()
                systems['metrics_system'].stop()
                await systems['alert_manager'].stop()
                print("\n[INFO] システム停止完了")
        except:
            pass


async def main():
    """メイン実行関数"""
    await demo_integrated_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
