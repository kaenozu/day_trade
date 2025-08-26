#!/usr/bin/env python3
"""
Dask Processor Utils
Issue #384: 並列処理のさらなる強化 - Utility Functions

Dask処理関連のユーティリティ関数とファクトリ関数を提供する
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict

from .core_processor import DaskDataProcessor
from .stock_analyzer import DaskStockAnalyzer
from .batch_processor import DaskBatchProcessor


def create_dask_data_processor(
    enable_distributed: bool = True, n_workers: int = None, **kwargs
) -> DaskDataProcessor:
    """
    DaskDataProcessorファクトリ関数

    Args:
        enable_distributed: 分散処理有効化
        n_workers: ワーカー数
        **kwargs: その他の設定

    Returns:
        DaskDataProcessorインスタンス
    """
    return DaskDataProcessor(
        enable_distributed=enable_distributed, n_workers=n_workers, **kwargs
    )


def create_complete_dask_system(
    enable_distributed: bool = True, n_workers: int = None, **kwargs
) -> Dict[str, Any]:
    """
    完全なDaskシステム構成を作成

    Args:
        enable_distributed: 分散処理有効化
        n_workers: ワーカー数
        **kwargs: その他の設定

    Returns:
        システムコンポーネント辞書
    """
    # コアプロセッサー作成
    processor = create_dask_data_processor(
        enable_distributed=enable_distributed, n_workers=n_workers, **kwargs
    )

    # アナライザーとバッチプロセッサー作成
    analyzer = DaskStockAnalyzer(processor)
    batch_processor = DaskBatchProcessor(processor)

    return {
        "processor": processor,
        "analyzer": analyzer,
        "batch_processor": batch_processor,
        "system_info": {
            "distributed_enabled": enable_distributed,
            "n_workers": n_workers,
            "creation_timestamp": datetime.now().isoformat(),
        },
    }


async def demonstrate_dask_system():
    """
    Daskシステムのデモンストレーション実行

    完全なDaskシステムの機能をテストし、結果を表示する
    """
    print("=== Issue #384 Dask分散処理テスト ===")

    processor = None
    try:
        # プロセッサー初期化
        processor = create_dask_data_processor(
            enable_distributed=True, n_workers=4, memory_limit="1GB"
        )

        # テストデータ
        test_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # 1. 並列データ処理テスト
        print("\n1. 並列データ処理テスト")
        result_data = await processor.process_multiple_symbols_parallel(
            test_symbols, start_date, end_date, include_technical=True
        )
        print(f"処理結果: {len(result_data)}レコード")

        # 2. 相関分析テスト
        print("\n2. 分散相関分析テスト")
        analyzer = DaskStockAnalyzer(processor)
        portfolio_analysis = (
            await analyzer.analyze_portfolio_performance_distributed(
                test_symbols[:3], benchmark_symbol=test_symbols[-1]
            )
        )
        print(
            f"ポートフォリオ分析結果: {portfolio_analysis.get('portfolio_summary', {})}"
        )

        # 3. バッチパイプラインテスト
        print("\n3. バッチパイプラインテスト")
        batch_processor = DaskBatchProcessor(processor)
        pipeline_result = await batch_processor.process_market_data_pipeline(
            test_symbols,
            ["fetch_data", "technical_analysis", "data_cleaning"],
            start_date,
            end_date,
        )
        print(
            f"パイプライン結果: {pipeline_result.get('records_processed', 0)}レコード処理"
        )

        # 統計情報
        print("\n4. パフォーマンス統計")
        stats = processor.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # ヘルスチェック
        print("\n5. ヘルスステータス")
        health = processor.get_health_status()
        print(f"  状態: {health['status']}")
        print(f"  分散処理: {health['distributed_enabled']}")
        print(f"  Dask利用可能: {health['dask_available']}")

    except Exception as e:
        print(f"テスト実行エラー: {e}")

    finally:
        if processor:
            processor.cleanup()


def validate_dask_environment() -> Dict[str, Any]:
    """
    Dask環境の検証

    Returns:
        環境検証結果
    """
    validation_result = {
        "dask_available": False,
        "dependencies_satisfied": False,
        "recommended_settings": {},
        "issues": [],
    }

    try:
        # Dask可用性チェック
        import dask
        import dask.dataframe as dd
        from dask.distributed import Client, LocalCluster

        validation_result["dask_available"] = True
        validation_result["dask_version"] = dask.__version__

        # 依存関係チェック
        try:
            import pandas as pd
            import numpy as np

            validation_result["dependencies_satisfied"] = True
            validation_result["pandas_version"] = pd.__version__
            validation_result["numpy_version"] = np.__version__

        except ImportError as e:
            validation_result["issues"].append(f"依存関係不足: {e}")

        # 推奨設定
        import os

        n_cores = os.cpu_count() or 4
        validation_result["recommended_settings"] = {
            "n_workers": min(8, n_cores),
            "threads_per_worker": 2,
            "memory_limit": "2GB",
        }

    except ImportError:
        validation_result["issues"].append("Daskが利用できません")

    return validation_result


def get_system_performance_metrics() -> Dict[str, Any]:
    """
    システムパフォーマンス指標を取得

    Returns:
        パフォーマンス指標
    """
    import psutil
    import os

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_count": os.cpu_count(),
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_usage_percent": psutil.virtual_memory().percent,
    }

    try:
        # ディスク使用量
        disk_usage = psutil.disk_usage("/")
        metrics["disk_total_gb"] = disk_usage.total / (1024**3)
        metrics["disk_free_gb"] = disk_usage.free / (1024**3)
        metrics["disk_usage_percent"] = (
            (disk_usage.total - disk_usage.free) / disk_usage.total * 100
        )
    except Exception:
        metrics["disk_info"] = "利用不可"

    return metrics


async def run_performance_benchmark(
    symbol_count: int = 10, 
    analysis_period_days: int = 30
) -> Dict[str, Any]:
    """
    パフォーマンスベンチマークの実行

    Args:
        symbol_count: 処理する銘柄数
        analysis_period_days: 分析期間

    Returns:
        ベンチマーク結果
    """
    import time

    benchmark_results = {
        "start_time": datetime.now().isoformat(),
        "symbol_count": symbol_count,
        "analysis_period_days": analysis_period_days,
        "results": {},
    }

    processor = None
    try:
        # システム作成
        system = create_complete_dask_system(enable_distributed=True)
        processor = system["processor"]
        analyzer = system["analyzer"]
        batch_processor = system["batch_processor"]

        # テスト銘柄（実際の処理では適切な銘柄を使用）
        test_symbols = [f"TEST{i:03d}" for i in range(symbol_count)]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_period_days)

        # 1. 基本データ処理ベンチマーク
        start_time = time.time()
        try:
            data_result = await processor.process_multiple_symbols_parallel(
                test_symbols[:5], start_date, end_date
            )  # 実際のAPIリクエストを避けるため制限
            data_processing_time = time.time() - start_time
            benchmark_results["results"]["data_processing"] = {
                "time_seconds": data_processing_time,
                "records_processed": len(data_result),
                "symbols_processed": min(5, symbol_count),
            }
        except Exception as e:
            benchmark_results["results"]["data_processing"] = {"error": str(e)}

        # 2. バッチ処理ベンチマーク
        start_time = time.time()
        try:
            pipeline_result = await batch_processor.process_market_data_pipeline(
                test_symbols[:3],
                ["fetch_data", "data_cleaning"],
                start_date,
                end_date,
            )
            batch_processing_time = time.time() - start_time
            benchmark_results["results"]["batch_processing"] = {
                "time_seconds": batch_processing_time,
                "records_processed": pipeline_result.get("records_processed", 0),
                "processing_successful": pipeline_result.get("processing_successful", False),
            }
        except Exception as e:
            benchmark_results["results"]["batch_processing"] = {"error": str(e)}

        # 3. システム統計
        benchmark_results["results"]["system_stats"] = processor.get_stats()
        benchmark_results["results"]["health_status"] = processor.get_health_status()

    except Exception as e:
        benchmark_results["error"] = str(e)

    finally:
        if processor:
            processor.cleanup()

        benchmark_results["end_time"] = datetime.now().isoformat()
        total_time = (
            datetime.fromisoformat(benchmark_results["end_time"])
            - datetime.fromisoformat(benchmark_results["start_time"])
        ).total_seconds()
        benchmark_results["total_benchmark_time_seconds"] = total_time

    return benchmark_results


if __name__ == "__main__":
    # メイン実行部
    asyncio.run(demonstrate_dask_system())
    print("\n=== Dask分散処理テスト完了 ===")