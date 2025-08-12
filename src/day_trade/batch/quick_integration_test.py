#!/usr/bin/env python3
"""
バッチ処理システム統合クイックテスト
Issue #376で実装したシステムの基本動作確認
"""

import os

# テスト対象システム
import sys
import tempfile
import time
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_request_consolidator import RequestPriority, create_consolidator
from database_bulk_optimizer import OptimizationStrategy, create_bulk_optimizer
from parallel_batch_engine import ProcessingMode, TaskPriority, create_batch_engine


def test_api_request_consolidator() -> Dict[str, Any]:
    """APIリクエスト統合システムクイックテスト"""
    print("APIリクエスト統合システムテスト開始...")

    consolidator = create_consolidator(batch_size=20, max_workers=2, enable_caching=True)

    consolidator.start()

    try:
        symbols = ["7203", "8306", "9984"]
        responses_received = []

        def response_callback(response):
            responses_received.append(response)

        # リクエスト投入
        request_id = consolidator.submit_request(
            endpoint="test_endpoint",
            symbols=symbols,
            priority=RequestPriority.HIGH,
            callback=response_callback,
        )

        # 完了待機
        time.sleep(2.0)

        # 統計取得
        stats = consolidator.get_stats()

        result = {
            "success": len(responses_received) > 0,
            "request_id": request_id,
            "responses_received": len(responses_received),
            "total_requests": stats.total_requests,
            "success_rate": stats.success_rate,
        }

        print(f"  ✓ リクエスト投入成功: {request_id}")
        print(f"  ✓ レスポンス受信: {len(responses_received)}")
        print(f"  ✓ 統計: requests={stats.total_requests}, success_rate={stats.success_rate:.1%}")

        return result

    finally:
        consolidator.stop()


def test_database_bulk_optimizer() -> Dict[str, Any]:
    """データベースバルク最適化クイックテスト"""
    print("データベースバルク最適化テスト開始...")

    # テスト用一時DBファイル
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = tmp_file.name

    try:
        optimizer = create_bulk_optimizer(
            db_path=test_db_path,
            batch_size=100,
            max_workers=2,
            optimization_strategy=OptimizationStrategy.SPEED_OPTIMIZED,
        )

        # テストテーブル作成
        with optimizer.connection_pool.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_stocks (
                    symbol TEXT,
                    price REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol)
                )
            """
            )
            conn.commit()

        # テストデータ
        test_data = [
            {"symbol": "7203", "price": 2500.0, "volume": 1000000},
            {"symbol": "8306", "price": 850.0, "volume": 2000000},
            {"symbol": "9984", "price": 8500.0, "volume": 500000},
        ]

        # バルク挿入テスト
        insert_result = optimizer.bulk_insert(
            table_name="test_stocks", data=test_data, conflict_resolution="REPLACE"
        )

        # バルクUPSERTテスト
        update_data = test_data.copy()
        for record in update_data:
            record["price"] *= 1.1

        upsert_result = optimizer.bulk_upsert(
            table_name="test_stocks", data=update_data, key_columns=["symbol"]
        )

        optimizer.close()

        result = {
            "success": insert_result.success_rate > 0.8 and upsert_result.success_rate > 0.8,
            "insert_processed": insert_result.processed_records,
            "insert_success_rate": insert_result.success_rate,
            "upsert_processed": upsert_result.processed_records,
            "upsert_success_rate": upsert_result.success_rate,
            "insert_throughput": insert_result.throughput_rps,
            "upsert_throughput": upsert_result.throughput_rps,
        }

        print(
            f"  ✓ バルク挿入: {insert_result.processed_records} records, {insert_result.success_rate:.1%} success"
        )
        print(
            f"  ✓ バルクUPSERT: {upsert_result.processed_records} records, {upsert_result.success_rate:.1%} success"
        )
        print(
            f"  ✓ スループット: insert={insert_result.throughput_rps:.1f} RPS, upsert={upsert_result.throughput_rps:.1f} RPS"
        )

        return result

    finally:
        try:
            os.unlink(test_db_path)
        except:
            pass


def test_parallel_batch_engine() -> Dict[str, Any]:
    """並列バッチ処理エンジンクイックテスト"""
    print("並列バッチ処理エンジンテスト開始...")

    def test_task(n: int, delay: float = 0.01) -> int:
        time.sleep(delay)
        return sum(i for i in range(n))

    def failing_task(should_fail: bool = True) -> str:
        if should_fail:
            raise ValueError("Test failure")
        return "Success"

    engine = create_batch_engine(
        workers=2,
        processing_mode=ProcessingMode.THREAD_BASED,
        enable_adaptive_scaling=False,
    )

    engine.start()

    try:
        task_ids = []

        # 成功タスク投入
        for i in range(5):
            task_id = engine.submit_task(
                test_task,
                50 + i * 10,
                delay=0.01,
                priority=TaskPriority.NORMAL,
                task_id=f"success_task_{i}",
            )
            task_ids.append(task_id)

        # 失敗タスク投入
        for i in range(2):
            task_id = engine.submit_task(
                failing_task,
                should_fail=(i == 0),  # 最初のは失敗
                retry_count=1,
                task_id=f"fail_task_{i}",
            )
            task_ids.append(task_id)

        # 完了待機
        time.sleep(2.0)

        # 結果取得
        results = engine.get_results_batch(task_ids, timeout=5.0)

        # 統計取得
        stats = engine.get_stats()

        successful_tasks = [r for r in results.values() if r.success]
        failed_tasks = [r for r in results.values() if not r.success]

        result = {
            "success": len(successful_tasks) >= 5,  # 最低5つの成功タスクが必要
            "total_tasks": len(results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": stats.success_rate,
            "throughput_tps": stats.throughput_tps,
            "avg_execution_time": stats.average_execution_time,
        }

        print(f"  ✓ タスク完了: {len(results)}/7 tasks")
        print(f"  ✓ 成功: {len(successful_tasks)}, 失敗: {len(failed_tasks)}")
        print(
            f"  ✓ エンジン統計: success_rate={stats.success_rate:.1%}, throughput={stats.throughput_tps:.2f} TPS"
        )

        return result

    finally:
        engine.stop(timeout=5.0)


def run_quick_integration_test() -> Dict[str, Any]:
    """クイック統合テスト実行"""
    print("=== バッチ処理システム統合クイックテスト ===\n")

    results = {}
    all_success = True

    # 各システムテスト実行
    tests = [
        ("api_request_consolidator", test_api_request_consolidator),
        ("database_bulk_optimizer", test_database_bulk_optimizer),
        ("parallel_batch_engine", test_parallel_batch_engine),
    ]

    for test_name, test_func in tests:
        try:
            test_result = test_func()
            results[test_name] = test_result

            if test_result["success"]:
                print(f"✓ {test_name} テスト成功\n")
            else:
                print(f"✗ {test_name} テスト失敗\n")
                all_success = False

        except Exception as e:
            print(f"✗ {test_name} テスト例外: {e}\n")
            results[test_name] = {"success": False, "error": str(e)}
            all_success = False

    # 総合結果
    total_tests = len(tests)
    successful_tests = len([r for r in results.values() if r.get("success", False)])

    print("=== 統合テスト結果 ===")
    print(f"総テスト数: {total_tests}")
    print(f"成功: {successful_tests}")
    print(f"失敗: {total_tests - successful_tests}")
    print(f"成功率: {successful_tests/total_tests:.1%}")
    print(f"全体成功: {'✓' if all_success else '✗'}")

    summary = {
        "overall_success": all_success,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / total_tests,
        "test_results": results,
        "timestamp": time.time(),
    }

    return summary


if __name__ == "__main__":
    summary = run_quick_integration_test()

    if summary["overall_success"]:
        print("\n🎉 すべてのバッチ処理システムが正常に動作しています！")
        exit(0)
    else:
        print("\n⚠️  一部のシステムに問題があります。")
        exit(1)
