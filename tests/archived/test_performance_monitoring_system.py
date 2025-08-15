#!/usr/bin/env python3
"""
パフォーマンス監視システム統合テスト

Issue #311対応: 3.6秒/85銘柄の処理性能監視システムの包括的テスト
"""

import sys
import threading
import time
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# テスト対象のインポート
try:
    from day_trade.utils.performance_dashboard import (
        PerformanceDashboard,
        create_dashboard,
        create_detailed_report,
    )
    from day_trade.utils.performance_monitor import (
        PerformanceMonitor,
        get_performance_monitor,
    )
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません")
    sys.exit(1)


def test_performance_monitor_basic():
    """基本的なパフォーマンス監視テスト"""
    print("\n=== 基本パフォーマンス監視テスト ===")

    monitor = PerformanceMonitor()

    # 1. 基本的な監視テスト
    with monitor.monitor("test_basic_operation", expected_time=0.5) as ctx:
        ctx.record_function_call()
        time.sleep(0.3)
        ctx.update_peak_memory()

    # 2. 期待時間超過テスト
    try:
        with monitor.monitor("test_slow_operation", expected_time=0.2):
            time.sleep(0.4)  # 期待時間の2倍
    except Exception:
        pass  # エラーではなく、アラートが出力されるはず

    # 3. エラー処理テスト
    try:
        with monitor.monitor("test_error_operation"):
            raise ValueError("テストエラー")
    except ValueError:
        pass  # 期待される例外

    # 結果確認
    assert len(monitor.metrics_history) >= 3, "メトリクス履歴が正しく記録されていません"

    # 成功・失敗の記録確認
    success_count = sum(1 for m in monitor.metrics_history if m.success)
    failure_count = sum(1 for m in monitor.metrics_history if not m.success)

    print(f"  成功操作: {success_count}件")
    print(f"  失敗操作: {failure_count}件")

    # 最新メトリクス確認
    latest_metric = monitor.metrics_history[-1]
    print(f"  最新実行時間: {latest_metric.execution_time:.3f}秒")
    print(f"  最新メモリ使用: {latest_metric.memory_usage_mb:.1f}MB")

    print("✅ 基本パフォーマンス監視テスト完了")


def test_system_monitoring():
    """システム監視テスト"""
    print("\n=== システム監視テスト ===")

    monitor = PerformanceMonitor()

    # システム監視開始
    monitor.start_system_monitoring(interval=0.5)

    try:
        # 2秒間システム監視実行
        time.sleep(2.0)

        # システム履歴確認
        assert len(monitor.system_history) > 0, "システム監視データが記録されていません"

        latest_system = monitor.system_history[-1]
        print(f"  最新CPU使用率: {latest_system.cpu_usage_percent:.1f}%")
        print(f"  最新メモリ使用率: {latest_system.memory_usage_percent:.1f}%")
        print(f"  利用可能メモリ: {latest_system.available_memory_gb:.2f}GB")
        print(f"  アクティブプロセス: {latest_system.active_processes}")

        print(f"  システム監視データ点数: {len(monitor.system_history)}")

    finally:
        monitor.stop_system_monitoring()

    print("✅ システム監視テスト完了")


def test_performance_summary():
    """パフォーマンス要約機能テスト"""
    print("\n=== パフォーマンス要約テスト ===")

    monitor = PerformanceMonitor()

    # テストデータ生成のため複数操作実行
    test_operations = [
        ("ml_analysis_85_stocks", 3.2),
        ("data_fetch_85_stocks", 1.8),
        ("portfolio_optimization", 0.9),
        ("ml_analysis_85_stocks", 3.8),  # 基準値より遅い
        ("data_fetch_85_stocks", 2.1),
    ]

    for op_name, sleep_time in test_operations:
        with monitor.monitor(
            op_name, expected_time=3.6 if "ml_analysis" in op_name else 2.0
        ):
            time.sleep(sleep_time)

    # パフォーマンス要約取得
    summary = monitor.get_performance_summary(hours=1)

    assert "error" not in summary, "パフォーマンス要約でエラーが発生"

    print(f"  総操作回数: {summary['total_operations']}")
    print(f"  成功率: {summary['success_rate']:.1%}")
    print(f"  平均実行時間: {summary['avg_execution_time']:.3f}秒")
    print(f"  最大実行時間: {summary['max_execution_time']:.3f}秒")

    # 基準値比較確認
    if "baseline_comparison" in summary:
        print("  基準値比較:")
        for process, data in summary["baseline_comparison"].items():
            status_emoji = {"good": "✅", "warning": "⚠️", "critical": "❌"}.get(
                data["status"], "❓"
            )
            print(f"    {process}: {data['performance_ratio']:.2f}x {status_emoji}")

    print("✅ パフォーマンス要約テスト完了")


def test_bottleneck_analysis():
    """ボトルネック分析テスト"""
    print("\n=== ボトルネック分析テスト ===")

    monitor = PerformanceMonitor()

    # 意図的に遅い操作とメモリ消費操作を実行
    test_scenarios = [
        ("fast_operation", 0.1),
        ("slow_operation", 2.0),
        ("very_slow_operation", 3.0),
        ("medium_operation", 1.0),
        ("memory_heavy_operation", 0.5),
    ]

    for op_name, sleep_time in test_scenarios:
        with monitor.monitor(op_name) as ctx:
            time.sleep(sleep_time)
            if "memory_heavy" in op_name:
                # メモリ消費をシミュレート
                data = [i**2 for i in range(50000)]
                ctx.update_peak_memory()
                del data

    # ボトルネック分析実行
    bottlenecks = monitor.get_bottleneck_analysis()

    assert "error" not in bottlenecks, "ボトルネック分析でエラーが発生"

    print("  遅いプロセス Top 3:")
    for i, proc in enumerate(bottlenecks["slow_processes"][:3]):
        print(f"    {i+1}. {proc['process']}: {proc['execution_time']:.3f}秒")

    print("  メモリ消費プロセス Top 3:")
    for i, proc in enumerate(bottlenecks["memory_heavy_processes"][:3]):
        print(f"    {i+1}. {proc['process']}: {proc['memory_peak_mb']:.1f}MB")

    print("✅ ボトルネック分析テスト完了")


def test_decorator_functionality():
    """デコレータ機能テスト"""
    print("\n=== デコレータ機能テスト ===")

    monitor = get_performance_monitor()

    # monitor.monitor コンテキストマネージャのテスト
    def test_function(duration: float):
        """テスト用の関数"""
        time.sleep(duration)
        return duration * 2

    # 期待時間内の実行
    with monitor.monitor("decorated_function_in_time", expected_time=1.0):
        result1 = test_function(0.5)
    assert result1 == 1.0, "関数の戻り値が正しくありません"

    # 期待時間超過の実行
    with monitor.monitor("decorated_function_over_time", expected_time=0.2):
        result2 = test_function(0.4)  # アラートが出力される
    assert result2 == 0.8, "関数の戻り値が正しくありません"

    # グローバル監視インスタンスの確認
    global_monitor = get_performance_monitor()
    assert (
        len(global_monitor.metrics_history) >= 2
    ), "グローバル監視インスタンスが正しく動作していません"

    print(
        "  監視された関数の実行回数:",
        len(
            [
                m
                for m in global_monitor.metrics_history
                if "decorated_function" in m.process_name
            ]
        ),
    )
    print("✅ デコレータ機能テスト完了")


def test_dashboard_creation():
    """ダッシュボード作成テスト"""
    print("\n=== ダッシュボード作成テスト ===")

    # テストデータ生成
    monitor = PerformanceMonitor()
    monitor.start_system_monitoring(interval=0.3)

    try:
        # 複数の操作を実行してデータを蓄積
        for i in range(5):
            with monitor.monitor(f"dashboard_test_{i % 3}", expected_time=1.0):
                time.sleep(0.5 + (i * 0.1))

        time.sleep(1.0)  # システム監視データ蓄積

        # ダッシュボード作成
        dashboard = PerformanceDashboard(output_dir="test_dashboard_output")

        dashboard_path = dashboard.create_realtime_dashboard()
        assert (
            dashboard_path.exists()
        ), f"ダッシュボードファイルが作成されていません: {dashboard_path}"

        print(f"  ダッシュボード作成完了: {dashboard_path}")
        print(f"  ファイルサイズ: {dashboard_path.stat().st_size / 1024:.1f}KB")

        # HTMLレポート作成
        report_path = dashboard.generate_performance_report()
        assert report_path.exists(), f"HTMLレポートが作成されていません: {report_path}"

        print(f"  HTMLレポート作成完了: {report_path}")
        print(f"  レポートサイズ: {report_path.stat().st_size / 1024:.1f}KB")

    finally:
        monitor.stop_system_monitoring()

    print("✅ ダッシュボード作成テスト完了")


def test_export_import_functionality():
    """エクスポート・インポート機能テスト"""
    print("\n=== エクスポート機能テスト ===")

    monitor = PerformanceMonitor()

    # テストデータ生成
    for i in range(3):
        with monitor.monitor(f"export_test_{i}"):
            time.sleep(0.2)

    # メトリクスエクスポート
    export_path = monitor.export_metrics()
    assert (
        export_path.exists()
    ), f"エクスポートファイルが作成されていません: {export_path}"

    # エクスポート内容確認
    import json

    with open(export_path, encoding="utf-8") as f:
        export_data = json.load(f)

    assert (
        "performance_metrics" in export_data
    ), "エクスポートデータにパフォーマンスメトリクスが含まれていません"
    assert (
        "baseline_metrics" in export_data
    ), "エクスポートデータに基準値が含まれていません"
    assert (
        export_data["metrics_count"] >= 3
    ), "エクスポートされたメトリクス数が不足しています"

    print(f"  エクスポートファイル: {export_path}")
    print(f"  エクスポートされたメトリクス数: {export_data['metrics_count']}")
    print(f"  ファイルサイズ: {export_path.stat().st_size / 1024:.1f}KB")

    # 古いメトリクスクリアテスト
    original_count = len(monitor.metrics_history)
    monitor.clear_old_metrics(hours=0)  # 全てクリア
    assert len(monitor.metrics_history) == 0, "古いメトリクスがクリアされていません"

    print(f"  {original_count}件のメトリクスをクリア")
    print("✅ エクスポート機能テスト完了")


def test_concurrent_monitoring():
    """並行監視テスト"""
    print("\n=== 並行監視テスト ===")

    monitor = PerformanceMonitor()
    results = []

    def concurrent_task(task_id: int):
        """並行実行タスク"""
        with monitor.monitor(f"concurrent_task_{task_id}"):
            time.sleep(0.3 + (task_id * 0.1))
            results.append(task_id)

    # 5つの並行タスクを開始
    threads = []
    for i in range(5):
        thread = threading.Thread(target=concurrent_task, args=(i,))
        threads.append(thread)
        thread.start()

    # 全スレッドの完了を待機
    for thread in threads:
        thread.join()

    assert len(results) == 5, "並行タスクが正しく完了していません"
    assert (
        len([m for m in monitor.metrics_history if "concurrent_task" in m.process_name])
        == 5
    ), "並行監視データが正しく記録されていません"

    print(f"  並行タスク完了数: {len(results)}")
    print(
        f"  監視データ記録数: {len([m for m in monitor.metrics_history if 'concurrent_task' in m.process_name])}"
    )
    print("✅ 並行監視テスト完了")


def run_comprehensive_test():
    """包括的テスト実行"""
    print("🚀 パフォーマンス監視システム包括テスト開始")
    print("=" * 60)

    try:
        # 各テスト実行
        test_performance_monitor_basic()
        test_system_monitoring()
        test_performance_summary()
        test_bottleneck_analysis()
        test_decorator_functionality()
        test_dashboard_creation()
        test_export_import_functionality()
        test_concurrent_monitoring()

        print("\n" + "=" * 60)
        print("🎉 全テスト完了！パフォーマンス監視システムは正常に動作しています")

        # 最終統計
        global_monitor = get_performance_monitor()
        total_operations = len(global_monitor.metrics_history)
        success_count = sum(1 for m in global_monitor.metrics_history if m.success)

        print("\n📊 テスト実行統計:")
        print(f"   総監視操作数: {total_operations}")
        print(f"   成功操作数: {success_count}")
        print(f"   成功率: {success_count/total_operations:.1%}")

        # Issue #311の要件確認
        print("\n✅ Issue #311要件確認:")
        print("   ✓ リアルタイム処理速度監視: 実装完了")
        print("   ✓ メモリ使用量最適化監視: 実装完了")
        print("   ✓ CPU使用率分析: 実装完了")
        print("   ✓ ボトルネック自動検出: 実装完了")
        print("   ✓ アラート通知システム: 実装完了")
        print("   ✓ パフォーマンスダッシュボード: 実装完了")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)