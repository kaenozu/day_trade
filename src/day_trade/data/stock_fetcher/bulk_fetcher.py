"""
株価データ一括取得機能
大量銘柄の効率的なデータ取得を提供
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from ...utils.logging_config import log_performance_metric
from ...utils.yfinance_import import get_yfinance

from .exceptions import DataNotFoundError, InvalidSymbolError

# yfinance統一インポート - Issue #614対応
yf, YFINANCE_AVAILABLE = get_yfinance()


class BulkStockFetcher:
    """
    一括株価データ取得機能を提供するクラス
    StockFetcherクラスの補助クラスとして機能
    """

    def __init__(self, stock_fetcher):
        """
        Args:
            stock_fetcher: メインのStockFetcherインスタンス
        """
        self.stock_fetcher = stock_fetcher
        self.logger = stock_fetcher.logger

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得

        Args:
            codes: 証券コードのリスト

        Returns:
            銘柄コードをキーとした価格情報の辞書
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not codes or not isinstance(codes, list):
            raise InvalidSymbolError(f"無効な銘柄コードリスト: {codes}")

        results = {}
        failed_codes = []

        # Max workers to avoid overwhelming the API or local resources
        max_workers = min(len(codes), 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(self.stock_fetcher.get_current_price, code): code
                for code in codes
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    data = future.result()
                    if data:
                        results[code] = data
                    else:
                        failed_codes.append(code)
                except Exception as e:
                    self.logger.warning(f"銘柄 {code} の取得に失敗: {e}")
                    failed_codes.append(code)

        if failed_codes:
            self.logger.info(f"取得に失敗した銘柄: {failed_codes}")

        return results

    def bulk_get_company_info(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の企業情報を一括取得（yfinance.Tickers使用）

        Args:
            codes: 銘柄コードのリスト
            batch_size: 一回の処理で取得する銘柄数
            delay: バッチ間の遅延（秒）

        Returns:
            {銘柄コード: 企業情報辞書} の辞書
        """
        if not codes:
            return {}

        results = {}
        start_time = time.time()

        # バッチごとに処理
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_start = time.time()

            try:
                # yfinance.Tickersを使用して一括取得 - Issue #614対応
                if not YFINANCE_AVAILABLE:
                    logging.error("yfinanceが利用できません")
                    continue

                tickers = yf.Tickers(" ".join(f"{code}.T" for code in batch_codes))

                for code in batch_codes:
                    try:
                        ticker_symbol = f"{code}.T"
                        ticker = tickers.tickers.get(ticker_symbol, None)

                        if ticker:
                            info = ticker.info
                            if info and info.get("symbol"):
                                # 企業情報を整理
                                company_info = {
                                    "name": info.get(
                                        "longName", info.get("shortName", "")
                                    ),
                                    "sector": info.get("sector", ""),
                                    "industry": info.get("industry", ""),
                                    "country": info.get("country", ""),
                                    "website": info.get("website", ""),
                                    "business_summary": info.get("businessSummary", ""),
                                    "market_cap": info.get("marketCap"),
                                    "employees": info.get("fullTimeEmployees"),
                                    "symbol": code,
                                }
                                results[code] = company_info

                                # キャッシュに保存
                                cache_key = f"company_info_{code}"
                                if hasattr(
                                    self.stock_fetcher, "_data_cache"
                                ) and self.stock_fetcher._data_cache:
                                    self.stock_fetcher._data_cache.set(
                                        cache_key, company_info, ttl=3600
                                    )
                            else:
                                print(f"警告: 企業情報が取得できません: {code}")
                                results[code] = None
                        else:
                            print(f"警告: Tickerオブジェクトが取得できません: {code}")
                            results[code] = None

                    except Exception as e:
                        print(f"エラー: 個別銘柄処理エラー {code}: {e}")
                        results[code] = None

                batch_elapsed = time.time() - batch_start
                log_performance_metric(
                    "bulk_company_info_batch",
                    {
                        "batch_size": len(batch_codes),
                        "batch_index": i // batch_size,
                        "elapsed_ms": batch_elapsed * 1000,
                        "codes_processed": len(batch_codes),
                    },
                )

                # レート制限対応の遅延
                if delay > 0 and i + batch_size < len(codes):
                    time.sleep(delay)

            except Exception as e:
                print(f"エラー: バッチ処理エラー (codes {i}-{i + batch_size - 1}): {e}")
                # バッチ失敗時は個別に処理
                for code in batch_codes:
                    try:
                        results[code] = self.stock_fetcher.get_company_info(code)
                    except Exception:
                        results[code] = None

        total_elapsed = time.time() - start_time
        successful_count = sum(1 for result in results.values() if result is not None)

        log_performance_metric(
            "bulk_company_info_complete",
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

        print(
            f"一括企業情報取得完了: {successful_count}/{len(codes)}件成功 "
            f"({total_elapsed:.2f}秒)"
        )

        return results