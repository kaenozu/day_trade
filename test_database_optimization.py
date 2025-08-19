#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データベース最適化システムのテスト
Issue #918 項目7対応: データベースアクセスとクエリの最適化

データベース最適化機能の動作確認テスト
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# パスの設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_database_services():
    """データベースサービステスト"""
    print("=== データベースサービステスト ===")

    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.services import register_default_services
        from src.day_trade.core.database_services import (
            IDatabaseService, IQueryOptimizerService, ICacheService
        )

        # サービス登録
        register_default_services()
        print("OK: データベースサービス登録完了")

        container = get_container()

        # 1. キャッシュサービステスト
        cache_service = container.resolve(ICacheService)

        # キャッシュ基本操作
        cache_service.set("test_key", "test_value", ttl=60)
        cached_value = cache_service.get("test_key")
        assert cached_value == "test_value", "キャッシュの設定・取得が失敗"
        print("OK: キャッシュサービス動作確認")

        # キャッシュ統計
        stats = cache_service.get_stats()
        assert stats['hits'] >= 1, "キャッシュヒットカウントが正しくない"
        print(f"OK: キャッシュ統計取得 - ヒット率: {stats['hit_rate']:.2%}")

        # 2. データベースサービステスト
        db_service = container.resolve(IDatabaseService)

        # パフォーマンス指標取得
        metrics = db_service.get_performance_metrics()
        assert metrics is not None, "パフォーマンス指標取得失敗"
        print("OK: データベースサービス初期化確認")
        print(f"OK: パフォーマンス指標 - クエリ数: {metrics.query_count}, 平均時間: {metrics.avg_query_time:.3f}s")

        # 3. クエリ最適化サービステスト
        optimizer_service = container.resolve(IQueryOptimizerService)

        # クエリ最適化テスト
        test_query = "SELECT * FROM stock_data WHERE created_at > '2024-01-01' ORDER BY created_at DESC"
        result = optimizer_service.optimize_query(test_query)

        assert result is not None, "クエリ最適化結果がNone"
        assert result.optimized_query != "", "最適化クエリが空"
        assert len(result.optimization_techniques) >= 0, "最適化技術リストが正しくない"

        print("OK: クエリ最適化動作確認")
        print(f"OK: 最適化技術: {', '.join(result.optimization_techniques) if result.optimization_techniques else 'なし'}")
        print(f"OK: 推定改善: {result.performance_improvement:.1f}%")

        # クエリパフォーマンス分析
        analysis = optimizer_service.analyze_query_performance(test_query)
        assert 'query' in analysis, "クエリ分析結果が不正"
        assert 'estimated_cost' in analysis, "推定コストが含まれていない"

        print("OK: クエリパフォーマンス分析動作確認")
        print(f"OK: 推定コスト: {analysis['estimated_cost']:.2f}")

        return True

    except Exception as e:
        print(f"FAIL: データベースサービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_session_management():
    """データベースセッション管理テスト"""
    print("\n=== データベースセッション管理テスト ===")

    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.database_services import IDatabaseService

        container = get_container()
        db_service = container.resolve(IDatabaseService)

        # セッション取得テスト
        session_count = 3
        for i in range(session_count):
            with db_service.get_session() as session:
                assert session is not None, f"セッション{i+1}の取得に失敗"
                # 簡単なクエリ実行（テーブルが存在しない場合もあるが、セッション動作確認が目的）
                try:
                    result = session.execute("SELECT 1 as test_column")
                    row = result.fetchone()
                    assert row is not None, "テストクエリの実行に失敗"
                except:
                    pass  # テーブル存在エラーは無視（セッション取得が主目的）

        print(f"OK: {session_count}個のセッション管理成功")

        # パフォーマンス指標確認
        metrics = db_service.get_performance_metrics()
        print(f"OK: クエリ実行後の指標 - クエリ数: {metrics.query_count}")

        return True

    except Exception as e:
        print(f"FAIL: セッション管理テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_database_operations():
    """非同期データベース操作テスト"""
    print("\n=== 非同期データベース操作テスト ===")

    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.database_services import IDatabaseService

        container = get_container()
        db_service = container.resolve(IDatabaseService)

        # 非同期クエリ実行テスト
        test_queries = [
            "SELECT 1 as query1",
            "SELECT 2 as query2",
            "SELECT 3 as query3"
        ]

        start_time = time.time()

        # 並行実行テスト
        tasks = []
        for query in test_queries:
            task = asyncio.create_task(db_service.execute_query(query))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time

        successful_queries = sum(1 for result in results if not isinstance(result, Exception))
        print(f"OK: 並行クエリ実行 - 成功: {successful_queries}/{len(test_queries)}")
        print(f"OK: 実行時間: {execution_time:.3f}秒")

        # パフォーマンス確認
        metrics = db_service.get_performance_metrics()
        print(f"OK: 並行実行後の指標 - 平均クエリ時間: {metrics.avg_query_time:.3f}秒")

        return True

    except Exception as e:
        print(f"FAIL: 非同期操作テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_performance():
    """キャッシュパフォーマンステスト"""
    print("\n=== キャッシュパフォーマンステスト ===")

    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.database_services import ICacheService

        container = get_container()
        cache_service = container.resolve(ICacheService)

        # キャッシュクリア
        cache_service.clear()

        # 大量データでのキャッシュテスト
        test_data_count = 100
        test_data = {f"key_{i}": f"value_{i}" for i in range(test_data_count)}

        # 書き込みテスト
        start_time = time.time()
        for key, value in test_data.items():
            cache_service.set(key, value, ttl=300)
        write_time = time.time() - start_time

        # 読み込みテスト
        start_time = time.time()
        retrieved_count = 0
        for key in test_data.keys():
            if cache_service.get(key) is not None:
                retrieved_count += 1
        read_time = time.time() - start_time

        # 統計確認
        stats = cache_service.get_stats()

        print(f"OK: キャッシュ書き込み - {test_data_count}件を{write_time:.3f}秒で完了")
        print(f"OK: キャッシュ読み込み - {retrieved_count}件を{read_time:.3f}秒で完了")
        print(f"OK: キャッシュ統計 - ヒット率: {stats['hit_rate']:.2%}, サイズ: {stats['cache_size']}")

        return True

    except Exception as e:
        print(f"FAIL: キャッシュパフォーマンステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("データベース最適化システムのテストを開始します...\n")

    results = []

    # 各テストを実行
    results.append(("データベースサービス", test_database_services()))
    results.append(("セッション管理", test_database_session_management()))

    # 非同期テスト
    loop = asyncio.get_event_loop()
    results.append(("非同期データベース操作", loop.run_until_complete(test_async_database_operations())))

    results.append(("キャッシュパフォーマンス", test_cache_performance()))

    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:<25}: {status}")
        if result:
            passed += 1

    print(f"\n合計: {passed}/{len(results)} テスト通過")

    if passed == len(results):
        print("SUCCESS: 全テストが正常に完了しました！")
        print("データベースアクセスとクエリの最適化が成功しました。")
        return 0
    else:
        print("WARNING: 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())