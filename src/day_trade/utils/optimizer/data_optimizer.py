#!/usr/bin/env python3
"""
データ取得最適化

複数銘柄のデータを効率的に取得するための並列処理とバッチ処理を提供します。
"""

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple


class DataFetchOptimizer:
    """データ取得の最適化クラス"""

    def __init__(self, max_workers: int = 4, chunk_size: int = 50):
        self.max_workers = max_workers
        self.chunk_size = chunk_size

    async def fetch_multiple_async(
        self, symbols: List[str], fetch_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """複数銘柄の非同期並列取得"""

        async def fetch_symbol(symbol: str) -> Tuple[str, Any]:
            try:
                # 同期関数を非同期で実行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: fetch_func(symbol, **kwargs)
                )
                return symbol, result
            except Exception as e:
                return symbol, e

        # 並列実行
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果整理
        success_results = {}
        errors = {}

        for symbol, result in results:
            if isinstance(result, Exception):
                errors[symbol] = result
            else:
                success_results[symbol] = result

        return {
            "success": success_results,
            "errors": errors,
            "success_count": len(success_results),
            "error_count": len(errors),
        }

    def fetch_multiple_threaded(
        self, symbols: List[str], fetch_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """複数銘柄のスレッド並列取得"""
        success_results = {}
        errors = {}

        # チャンク分割
        chunks = [
            symbols[i: i + self.chunk_size]
            for i in range(0, len(symbols), self.chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # チャンク単位で並列実行
            for chunk in chunks:
                future_to_symbol = {
                    executor.submit(fetch_func, symbol, **kwargs): symbol
                    for symbol in chunk
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        success_results[symbol] = result
                    except Exception as e:
                        errors[symbol] = e

        return {
            "success": success_results,
            "errors": errors,
            "success_count": len(success_results),
            "error_count": len(errors),
        }

    def optimize_bulk_request(
        self,
        symbols: List[str],
        bulk_fetch_func: Callable,
        single_fetch_func: Callable,
        bulk_threshold: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """一括取得と個別取得の最適な組み合わせ"""
        if len(symbols) >= bulk_threshold:
            try:
                # 一括取得を試行
                return bulk_fetch_func(symbols, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"一括取得に失敗、個別取得にフォールバック: {e}",
                    stacklevel=2
                )
                # 個別取得にフォールバック
                return self.fetch_multiple_threaded(
                    symbols, single_fetch_func, **kwargs
                )
        else:
            # 少数の場合は個別取得
            return self.fetch_multiple_threaded(
                symbols, single_fetch_func, **kwargs
            )
