#!/usr/bin/env python3
"""
バッチ処理エンジンユーティリティ
Issue #376: バッチ処理の強化

便利関数とヘルパーメソッド
"""

from typing import List

from .core_types import JobResult
from .engine import BatchProcessingEngine


async def execute_stock_batch_pipeline(
    symbols: List[str],
    include_historical: bool = False,
    store_data: bool = True,
    engine: BatchProcessingEngine = None,
) -> JobResult:
    """株価データバッチパイプライン実行"""

    if not engine:
        engine = BatchProcessingEngine()

    try:
        # ジョブ作成
        job = engine.create_stock_data_pipeline_job(
            symbols=symbols,
            include_historical=include_historical,
            store_data=store_data,
        )

        # 実行
        result = await engine.execute_job(job)
        return result

    finally:
        if not engine:  # 一時的に作成したエンジンの場合
            await engine.cleanup()