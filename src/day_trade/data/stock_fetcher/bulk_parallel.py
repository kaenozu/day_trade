"""
株価データ並列・一括取得拡張機能
大量銘柄の並列処理とヒストリカルデータ一括取得
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from ...utils.logging_config import log_performance_metric
from ...utils.yfinance_import import get_yfinance

from .parallel_core import ParallelProcessorCore

# yfinance統一インポート - Issue #614対応
yf, YFINANCE_AVAILABLE = get_yfinance()


class BulkParallelFetcher:
    """
    一括・並列株価データ取得拡張機能
    BulkStockFetcherの補完クラス
    """

    def __init__(self, stock_fetcher):
        """
        Args:
            stock_fetcher: メインのStockFetcherインスタンス
        """
        self.stock_fetcher = stock_fetcher
        self.logger = stock_fetcher.logger
        self.parallel_core = ParallelProcessorCore(stock_fetcher)

    def bulk_get_historical_data(
        self,
        codes: List[str],
        period: str = "1y",
        interval: str = "1d",
        batch_size: int = 50,
        delay: float = 0.1,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        複数銘柄のヒストリカルデータを一括取得

        Args:
            codes: 銘柄コードのリスト
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）
            batch_size: 一回の処理で取得する銘柄数
            delay: バッチ間の遅延（秒）

        Returns:
            {銘柄コード: 価格データのDataFrame} の辞書
        """
        if not codes:
            return {}

        results = {}
        start_time = time.time()

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_start = time.time()
            formatted_symbols = [
                self.stock_fetcher._format_symbol(code) for code in batch_codes
            ]

            try:
                # yf.downloadを使用した真の一括取得 - Issue #614対応
                if not YFINANCE_AVAILABLE:
                    logging.error("yfinanceが利用できません")
                    continue

                data = yf.download(
                    formatted_symbols,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    prepost=True,
                    threads=True,
                    progress=False,
                )

                # データ処理を別メソッドに分離
                batch_results = self._process_bulk_historical_data(
                    data, batch_codes, formatted_symbols
                )
                results.update(batch_results)

            except Exception as e:
                self.logger.error(
                    f"ヒストリカルデータバッチ処理エラー (codes {i}-{i + batch_size - 1}): {e}"
                )
                # フォールバック処理
                for code in batch_codes:
                    try:
                        results[code] = self.stock_fetcher.get_historical_data(
                            code, period, interval
                        )
                    except Exception:
                        results[code] = None

            # パフォーマンス記録
            batch_elapsed = time.time() - batch_start
            log_performance_metric(
                "bulk_historical_data_batch",
                {
                    "batch_size": len(batch_codes),
                    "batch_index": i // batch_size,
                    "elapsed_ms": batch_elapsed * 1000,
                    "codes_processed": len(batch_codes),
                },
            )

            if delay > 0 and i + batch_size < len(codes):
                time.sleep(delay)

        # 最終結果記録
        total_elapsed = time.time() - start_time
        successful_count = sum(1 for result in results.values() if result is not None)

        log_performance_metric(
            "bulk_historical_data_complete",
            {
                "total_codes": len(codes),
                "successful_count": successful_count,
                "failure_count": len(codes) - successful_count,
                "success_rate": successful_count / len(codes) if codes else 0,
                "total_elapsed_ms": total_elapsed * 1000,
                "avg_time_per_code": (
                    (total_elapsed / len(codes)) * 1000 if codes else 0
                ),
            },
        )

        self.logger.info(
            f"一括ヒストリカルデータ取得完了: {successful_count}/{len(codes)}件成功 ({total_elapsed:.2f}秒)"
        )
        return results

    def _process_bulk_historical_data(
        self, data, batch_codes: List[str], formatted_symbols: List[str]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """バルクヒストリカルデータの処理"""
        results = {}
        
        for code in batch_codes:
            symbol = self.stock_fetcher._format_symbol(code)
            ticker_data = None

            if len(formatted_symbols) > 1 and hasattr(data.columns, "levels"):
                # 複数銘柄の場合、MultiIndexからデータを抽出
                if symbol in data.columns.levels[0]:
                    ticker_data = data[symbol]
            else:
                # 単一銘柄の場合、直接データを使用
                ticker_data = data

            if ticker_data is not None and not ticker_data.empty:
                if ticker_data.index.tz is not None:
                    ticker_data.index = ticker_data.index.tz_localize(None)
                results[code] = ticker_data
            else:
                self.logger.warning(f"ヒストリカルデータが取得できませんでした: {code}")
                results[code] = None
        
        return results

    # 並列処理メソッドは ParallelProcessorCore に委譲
    def parallel_get_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """並列ヒストリカルデータ取得（並列処理コアに委譲）"""
        return self.parallel_core.parallel_get_historical_data(
            symbols, period, interval, max_concurrent
        )

    def parallel_get_current_prices(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """並列現在価格取得（並列処理コアに委譲）"""
        return self.parallel_core.parallel_get_current_prices(symbols, max_concurrent)

    def parallel_get_company_info(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """並列企業情報取得（並列処理コアに委譲）"""
        return self.parallel_core.parallel_get_company_info(symbols, max_concurrent)