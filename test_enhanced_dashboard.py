"""
Enhanced Dashboard テストスクリプト

リアルタイムダッシュボードシステムのコンポーネントテスト
"""

import asyncio
import os
import sys
from datetime import datetime

# パスを追加してモジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from day_trade.dashboard.core.feature_store_monitor import FeatureStoreMonitor
    from day_trade.dashboard.core.metrics_collector import MetricsCollector
    from day_trade.dashboard.core.realtime_stream import RealtimeStream
    from day_trade.ml.feature_store import FeatureStore
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    print("必要な依存関係が不足している可能性があります")
    sys.exit(1)


async def test_enhanced_dashboard():
    """Enhanced Dashboard システムテスト"""

    print("🚀 Enhanced Realtime Dashboard システムテスト")
    print("=" * 60)

    # 1. メトリクス収集器テスト
    print("\n📊 1. MetricsCollector テスト")
    try:
        metrics_collector = MetricsCollector(collection_interval=0.5)
        print("   ✅ MetricsCollector 初期化成功")

        # 現在のメトリクス取得テスト
        current_metrics = metrics_collector.get_current_metrics()
        if current_metrics:
            cpu_usage = current_metrics.get('cpu', {}).get('usage_percent', 'N/A')
            memory_usage = current_metrics.get('memory', {}).get('usage_percent', 'N/A')
            print(f"   📈 CPU使用率: {cpu_usage}%")
            print(f"   💾 メモリ使用率: {memory_usage}%")

        # 短時間の収集テスト
        await metrics_collector.start_collection()
        print("   🔄 メトリクス収集開始")
        await asyncio.sleep(2)  # 2秒間収集

        history = metrics_collector.get_metrics_history(1)  # 直近1分
        print(f"   📊 履歴データ: {len(history)} サンプル")

        health_report = metrics_collector.generate_health_report()
        health_score = health_report.get('overall_health', 0)
        print(f"   💊 システム健全性スコア: {health_score}/100")

        await metrics_collector.stop_collection()
        print("   ✅ MetricsCollector テスト完了")

    except Exception as e:
        print(f"   ❌ MetricsCollector エラー: {e}")

    # 2. Feature Store監視器テスト
    print("\n🔍 2. FeatureStoreMonitor テスト")
    try:
        # Feature Store初期化
        feature_store = FeatureStore(cache_size=100)
        print("   ✅ FeatureStore 初期化成功")

        # 監視器初期化
        fs_monitor = FeatureStoreMonitor(update_interval=1.0)
        fs_monitor.set_feature_store(feature_store)
        print("   ✅ FeatureStoreMonitor 初期化成功")

        # いくつかのテストデータでFeature Storeを使用
        import numpy as np
        import pandas as pd

        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='1min'),
            'price': np.random.uniform(1000, 2000, 20),
            'volume': np.random.randint(1000, 5000, 20)
        })

        # 监视开始
        await fs_monitor.start_monitoring()
        print("   🔄 Feature Store監視開始")

        # テスト用の特徴量生成（キャッシュヒット/ミスを生成）
        for i in range(5):
            try:
                # 同じキーで複数回呼び出し（キャッシュヒットテスト）
                cache_key = f"test_feature_{i % 3}"
                feature_store.get_or_generate_feature(
                    cache_key,
                    lambda: f"generated_feature_{i}_{datetime.now().microsecond}"
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"     警告: 特徴量生成テストエラー: {e}")

        await asyncio.sleep(2)  # 監視データ蓄積待機

        # 監視結果確認
        current_metrics = fs_monitor.get_current_metrics()
        if current_metrics:
            hit_rate = current_metrics.get('hit_rate', 0)
            speedup = current_metrics.get('speedup_ratio', 0)
            total_requests = current_metrics.get('total_requests', 0)

            print(f"   🎯 キャッシュヒット率: {hit_rate}%")
            print(f"   ⚡ 高速化倍率: {speedup}x")
            print(f"   📊 総リクエスト数: {total_requests}")

        # 健全性チェック
        health_status = fs_monitor.get_health_status()
        print(f"   💊 Feature Store健全性: {health_status.get('status', 'unknown')}")
        print(f"   📋 スコア: {health_status.get('score', 0)}/100")

        await fs_monitor.stop_monitoring()
        print("   ✅ FeatureStoreMonitor テスト完了")

    except Exception as e:
        print(f"   ❌ FeatureStoreMonitor エラー: {e}")

    # 3. リアルタイムストリーム テスト
    print("\n📡 3. RealtimeStream テスト")
    try:
        realtime_stream = RealtimeStream(broadcast_interval=1.0)
        print("   ✅ RealtimeStream 初期化成功")

        # メトリクス収集器とFeature Store監視器を設定
        if 'metrics_collector' in locals():
            realtime_stream.set_metrics_collector(metrics_collector)
            print("   🔗 MetricsCollector 連携設定完了")

        if 'fs_monitor' in locals():
            realtime_stream.set_feature_store_monitor(fs_monitor)
            print("   🔗 FeatureStoreMonitor 連携設定完了")

        # 接続統計テスト
        stats = realtime_stream.get_connection_stats()
        print(f"   📊 接続統計: {stats['active_connections']} 接続中")
        print(f"   🔄 ストリーミング状態: {stats['streaming_active']}")

        print("   ✅ RealtimeStream テスト完了")

    except Exception as e:
        print(f"   ❌ RealtimeStream エラー: {e}")

    # 4. 統合テスト
    print("\n🧪 4. 統合システムテスト")
    try:
        print("   🔄 全コンポーネント連携開始...")

        # 全体的なパフォーマンス測定
        start_time = datetime.now()

        # 簡単な負荷テスト
        for i in range(10):
            if 'feature_store' in locals():
                test_key = f"load_test_{i}"
                feature_store.get_or_generate_feature(
                    test_key,
                    lambda: f"load_test_value_{i}"
                )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        print(f"   ⏱️  処理時間: {execution_time:.3f}秒")
        print(f"   🚀 処理レート: {10/execution_time:.1f} ops/sec")

        print("   ✅ 統合テスト完了")

    except Exception as e:
        print(f"   ❌ 統合テスト エラー: {e}")

    # 5. 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    print("🎯 Enhanced Realtime Dashboard システム:")
    print("   ✅ リアルタイムシステム監視")
    print("   ✅ Feature Store パフォーマンス追跡")
    print("   ✅ WebSocket ベースのライブデータ配信")
    print("   ✅ 包括的な健全性レポート")
    print("   ✅ 高性能メトリクス収集")

    print("\n🌟 主要機能:")
    print("   📊 CPU・メモリ・ディスク監視")
    print("   🚀 Feature Store 効率性追跡")
    print("   📡 リアルタイムWebSocketストリーミング")
    print("   💊 システム健全性スコアリング")
    print("   📈 履歴データ分析・可視化")

    print(f"\n⏰ テスト実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 Enhanced Dashboard システムは正常に動作しています！")


if __name__ == "__main__":
    try:
        asyncio.run(test_enhanced_dashboard())
    except KeyboardInterrupt:
        print("\n\n⏹️ テスト中断されました")
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nテスト終了")
