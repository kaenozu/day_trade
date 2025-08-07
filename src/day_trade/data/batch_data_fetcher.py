#!/usr/bin/env python3
"""
バッチ株価データ取得システム

複数銘柄の並列取得とキャッシュ最適化
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from ..utils.logging_config import get_context_logger
from .real_market_data import RealMarketDataManager

logger = get_context_logger(__name__)


class BatchDataFetcher:
    """
    バッチ株価データ取得クラス

    複数銘柄の効率的な並列取得とキャッシュ最適化を提供
    """

    def __init__(self, max_workers: int = 3):
        self.data_manager = RealMarketDataManager()
        self.max_workers = max_workers

    def fetch_multiple_symbols(
        self, symbols: List[str], period: str = "60d", use_parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        複数銘柄データの並列取得

        Args:
            symbols: 銘柄コードリスト
            period: 取得期間
            use_parallel: 並列処理の有効/無効

        Returns:
            Dict[symbol, DataFrame]: 銘柄ごとのデータ
        """
        start_time = time.time()
        logger.info(f"バッチ取得開始: {len(symbols)}銘柄 (並列={use_parallel})")

        results = {}

        if use_parallel and len(symbols) > 1:
            results = self._fetch_parallel(symbols, period)
        else:
            results = self._fetch_sequential(symbols, period)

        elapsed = time.time() - start_time
        success_count = len([v for v in results.values() if v is not None])

        logger.info(
            f"バッチ取得完了: {success_count}/{len(symbols)}銘柄成功 ({elapsed:.2f}秒)"
        )

        return results

    def _fetch_parallel(
        self, symbols: List[str], period: str
    ) -> Dict[str, pd.DataFrame]:
        """並列取得"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 非同期タスク投入
            future_to_symbol = {
                executor.submit(self._safe_fetch_single, symbol, period): symbol
                for symbol in symbols
            }

            # 結果回収
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=30)  # 30秒タイムアウト
                    results[symbol] = data
                    if data is not None:
                        logger.info(f"並列取得完了: {symbol}")
                    else:
                        logger.warning(f"並列取得失敗: {symbol}")
                except Exception as e:
                    logger.error(f"並列取得エラー {symbol}: {e}")
                    results[symbol] = None

        return results

    def _fetch_sequential(
        self, symbols: List[str], period: str
    ) -> Dict[str, pd.DataFrame]:
        """逐次取得"""
        results = {}

        for symbol in symbols:
            try:
                data = self._safe_fetch_single(symbol, period)
                results[symbol] = data
                if data is not None:
                    logger.info(f"逐次取得完了: {symbol}")
                else:
                    logger.warning(f"逐次取得失敗: {symbol}")
            except Exception as e:
                logger.error(f"逐次取得エラー {symbol}: {e}")
                results[symbol] = None

        return results

    def _safe_fetch_single(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """安全な単一銘柄取得（エラーハンドリング付き）"""
        try:
            return self.data_manager.get_stock_data(symbol, period)
        except Exception as e:
            logger.error(f"単一取得エラー {symbol}: {e}")
            return None

    def preload_cache(self, symbols: List[str], periods: List[str] = None) -> int:
        """
        キャッシュの事前読み込み

        Args:
            symbols: 銘柄コードリスト
            periods: 期間リスト（None時は["5d", "60d"]）

        Returns:
            int: 成功した取得数
        """
        if periods is None:
            periods = ["5d", "60d"]

        logger.info(
            f"キャッシュ事前読み込み開始: {len(symbols)}銘柄 x {len(periods)}期間"
        )

        success_count = 0

        for period in periods:
            results = self.fetch_multiple_symbols(symbols, period, use_parallel=True)
            period_success = len([v for v in results.values() if v is not None])
            success_count += period_success

            logger.info(
                f"期間{period}読み込み完了: {period_success}/{len(symbols)}銘柄"
            )

        logger.info(f"キャッシュ事前読み込み完了: 総計{success_count}件")
        return success_count

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計情報取得"""
        memory_cache_size = len(self.data_manager.memory_cache)
        cache_expiry_size = len(self.data_manager.cache_expiry)

        # SQLiteキャッシュ件数
        sqlite_count = 0
        try:
            import sqlite3

            with sqlite3.connect(self.data_manager.cache_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_cache")
                sqlite_count = cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"SQLiteキャッシュ統計エラー: {e}")

        return {
            "memory_cache_entries": memory_cache_size,
            "memory_expiry_entries": cache_expiry_size,
            "sqlite_cached_symbols": sqlite_count,
        }

    def clear_expired_cache(self) -> int:
        """期限切れキャッシュのクリアリング"""
        cleared_count = 0
        current_time = pd.Timestamp.now()

        # メモリキャッシュクリア
        expired_keys = [
            key
            for key, expiry_time in self.data_manager.cache_expiry.items()
            if current_time > expiry_time
        ]

        for key in expired_keys:
            if key in self.data_manager.memory_cache:
                del self.data_manager.memory_cache[key]
                cleared_count += 1
            if key in self.data_manager.cache_expiry:
                del self.data_manager.cache_expiry[key]

        logger.info(f"期限切れキャッシュクリア完了: {cleared_count}件")
        return cleared_count


# 便利な関数（モジュールレベル）
def fetch_symbols_batch(
    symbols: List[str], period: str = "60d", parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    複数銘柄バッチ取得（簡易インターフェース）

    Args:
        symbols: 銘柄コードリスト
        period: 取得期間
        parallel: 並列処理使用

    Returns:
        Dict[symbol, DataFrame]: 取得結果
    """
    fetcher = BatchDataFetcher()
    return fetcher.fetch_multiple_symbols(symbols, period, parallel)


def preload_symbols_cache(symbols: List[str]) -> int:
    """
    銘柄キャッシュ事前読み込み（簡易インターフェース）

    Args:
        symbols: 銘柄コードリスト

    Returns:
        int: 成功取得数
    """
    fetcher = BatchDataFetcher()
    return fetcher.preload_cache(symbols)


if __name__ == "__main__":
    # テスト実行
    test_symbols = ["7203", "8306", "9984", "6758", "4689"]

    print("=== バッチデータ取得テスト ===")

    # バッチ取得テスト
    fetcher = BatchDataFetcher(max_workers=3)

    # キャッシュ統計（取得前）
    stats_before = fetcher.get_cache_stats()
    print(f"取得前キャッシュ統計: {stats_before}")

    # バッチ取得実行
    results = fetcher.fetch_multiple_symbols(test_symbols, "60d", use_parallel=True)

    # 結果確認
    for symbol, data in results.items():
        if data is not None:
            print(f"{symbol}: {len(data)}日分取得")
        else:
            print(f"{symbol}: 取得失敗")

    # キャッシュ統計（取得後）
    stats_after = fetcher.get_cache_stats()
    print(f"取得後キャッシュ統計: {stats_after}")

    # キャッシュ事前読み込みテスト
    print("\n=== キャッシュ事前読み込みテスト ===")
    preload_count = fetcher.preload_cache(["2914", "6861"], ["5d", "20d"])
    print(f"事前読み込み成功数: {preload_count}")
