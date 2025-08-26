"""
レガシーStockFetcherクラス
元の大きなファイルからの完全な機能を統合したクラス
"""

import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...utils.yfinance_import import get_yfinance
from ...utils.logging_config import (
    get_context_logger,
    get_performance_logger,
    log_api_call,
    log_error_with_context,
    log_performance_metric,
)
from .fetcher_base import StockFetcherBase
from .cache import cache_with_ttl
from .exceptions import DataNotFoundError, InvalidSymbolError, StockFetcherError
from .advanced_features import AdvancedStockFetcherMixin
from .bulk_operations import BulkOperationsMixin

# yfinance統一インポート
yf, YFINANCE_AVAILABLE = get_yfinance()

# 並列処理サポート
try:
    from ...utils.parallel_executor_manager import (
        TaskType,
        execute_parallel,
        get_global_executor_manager,
    )
    PARALLEL_SUPPORT = True
except ImportError:
    PARALLEL_SUPPORT = False


class LegacyStockFetcher(StockFetcherBase, AdvancedStockFetcherMixin, BulkOperationsMixin):
    """元のStockFetcherの完全な機能を持つレガシークラス"""

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        # 基底クラスを初期化
        super().__init__(cache_size, retry_count, retry_delay)
        
        # 高度な機能を初期化
        self.__init_advanced_features__()

        # キャッシュTTLを設定
        self.price_cache_ttl = price_cache_ttl
        self.historical_cache_ttl = historical_cache_ttl

        # 条件付きデバッグロギング（本番環境では無効化）
        self.enable_debug_logging = (
            os.getenv("STOCK_FETCHER_DEBUG", "false").lower() == "true"
        )

        self.logger.info(
            "LegacyStockFetcher初期化完了",
            extra={
                "cache_size": cache_size,
                "price_cache_ttl": price_cache_ttl,
                "historical_cache_ttl": historical_cache_ttl,
                "retry_count": retry_count,
                "auto_cache_tuning": self.auto_cache_tuning_enabled,
            },
        )

    @cache_with_ttl(30)  # 30秒キャッシュ
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在の株価情報を取得

        Args:
            code: 証券コード

        Returns:
            価格情報の辞書（現在値、前日終値、変化額、変化率など）
        """
        def _get_price():
            price_logger = self.logger
            price_logger.info("現在価格取得開始")

            start_time = time.time()

            try:
                self._validate_symbol(code)
                symbol = self._format_symbol(code)
                ticker = self._get_ticker(symbol)

                # 条件付きAPI呼び出しログ（デバッグ時のみ）
                if self.enable_debug_logging:
                    log_api_call("yfinance", "GET", f"ticker.info for {symbol}")
                info = ticker.info

                # infoが空または無効な場合
                if not info or len(info) < 5:
                    price_logger.error(
                        "企業情報取得失敗", info_size=len(info) if info else 0
                    )
                    raise DataNotFoundError(f"企業情報を取得できません: {symbol}")

                # 基本情報を取得
                current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )
                previous_close = info.get("previousClose")

                if current_price is None:
                    price_logger.error(
                        "現在価格データなし", available_keys=list(info.keys())[:10]
                    )
                    raise DataNotFoundError(f"現在価格を取得できません: {symbol}")

                # 変化額・変化率を計算
                change = current_price - previous_close if previous_close else 0
                change_percent = (
                    (change / previous_close * 100) if previous_close else 0
                )

                # パフォーマンスメトリクス
                elapsed_time = (time.time() - start_time) * 1000
                log_performance_metric(
                    "price_fetch_time", elapsed_time, "ms", stock_code=code
                )

                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": info.get("volume", 0),
                    "market_cap": info.get("marketCap"),
                    "timestamp": datetime.now(),
                }

                # パフォーマンス最適化: サンプリングログ（簡易版）
                if (
                    hasattr(self.performance_logger, "logger")
                    and hasattr(self.performance_logger.logger, "isEnabledFor")
                    and self.performance_logger.logger.isEnabledFor(logging.INFO)
                ):
                    # 10%の確率でログ出力（サンプリング）
                    import random

                    if random.random() < 0.1:
                        self.performance_logger.info(
                            "現在価格取得完了",
                            extra={
                                "current_price": current_price,
                                "change_percent": change_percent,
                                "elapsed_ms": elapsed_time,
                            },
                        )

                return result

            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                log_error_with_context(
                    e,
                    {
                        "operation": "get_current_price",
                        "stock_code": code,
                        "elapsed_ms": elapsed_time,
                    },
                )
                price_logger.error(
                    "現在価格取得失敗", error=str(e), elapsed_ms=elapsed_time
                )
                raise

        return self._retry_on_error(_get_price)

    @cache_with_ttl(300)  # 5分キャッシュ
    def get_historical_data(
        self, code: str, period: str = "1mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        ヒストリカルデータを取得
        """
        def _get_historical():
            self._validate_symbol(code)
            self._validate_period_interval(period, interval)

            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)

            # データ取得
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise DataNotFoundError(f"ヒストリカルデータが空です: {symbol}")

            # インデックスをタイムゾーン除去
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # データの妥当性チェック
            if len(df) == 0:
                raise DataNotFoundError(f"有効なデータがありません: {symbol}")

            return df

        return self._retry_on_error(_get_historical)

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """期間と間隔の妥当性をチェック"""
        valid_periods = [
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        ]
        valid_intervals = [
            "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
            "1d", "5d", "1wk", "1mo", "3mo"
        ]

        if period not in valid_periods:
            raise InvalidSymbolError(f"無効な期間: {period}. 有効な値: {valid_periods}")

        if interval not in valid_intervals:
            raise InvalidSymbolError(
                f"無効な間隔: {interval}. 有効な値: {valid_intervals}"
            )

        # 分足データは短期間のみ対応
        if interval.endswith("m") and period not in ["1d", "5d"]:
            raise InvalidSymbolError(
                "分足データは1d または 5d 期間のみサポートされています"
            )

    @cache_with_ttl(300)  # 5分キャッシュ
    def get_historical_data_range(
        self,
        code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        指定期間のヒストリカルデータを取得
        """
        def _get_historical_range():
            self._validate_symbol(code)
            self._validate_date_range(start_date, end_date)

            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)

            # 文字列の場合はdatetimeに変換
            start_dt = start_date
            end_dt = end_date
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # データ取得
            df = ticker.history(start=start_dt, end=end_dt, interval=interval)

            if df.empty:
                raise DataNotFoundError(
                    f"指定期間のヒストリカルデータが空です: {symbol}"
                )

            # インデックスをタイムゾーン除去
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df

        return self._retry_on_error(_get_historical_range)

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得
        """
        if not codes or not isinstance(codes, list):
            raise InvalidSymbolError(f"無効な銘柄コードリスト: {codes}")

        results = {}
        failed_codes = []

        # Max workers to avoid overwhelming the API or local resources
        max_workers = min(len(codes), 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(self.get_current_price, code): code for code in codes
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

    @cache_with_ttl(3600)  # 1時間キャッシュ
    def get_company_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        企業情報を取得
        """
        def _get_company_info():
            self._validate_symbol(code)
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or len(info) < 5:
                raise DataNotFoundError(f"企業情報を取得できません: {symbol}")

            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "employees": info.get("fullTimeEmployees"),
                "description": info.get("longBusinessSummary"),
                "website": info.get("website"),
                "headquarters": info.get("city"),
                "country": info.get("country"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
            }

        return self._retry_on_error(_get_company_info)

    def bulk_get_current_prices(
        self, codes: List[str], batch_size: int = 100, delay: float = 0.05
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の現在価格を一括取得（シンプル版）
        """
        if not codes:
            return {}

        results = {}
        for code in codes:
            try:
                results[code] = self.get_current_price(code)
            except Exception:
                results[code] = None
        return results

    # 並列処理機能 (Issue #383)
    def parallel_get_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        複数銘柄のヒストリカルデータを並列取得
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return self._sequential_get_historical_data(symbols, period, interval)

        self.logger.info(f"並列ヒストリカルデータ取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (self.get_historical_data, (symbol, period, interval), {})
            tasks.append(task)

        # 並列実行マネージャーで実行
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks, max_concurrent=max_concurrent or min(len(symbols), 10)
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"ヒストリカルデータ取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        return symbol_results

    def parallel_get_current_prices(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の現在価格を並列取得
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return {symbol: self.get_current_price(symbol) for symbol in symbols}

        self.logger.info(f"並列現在価格取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (self.get_current_price, (symbol,), {})
            tasks.append(task)

        # 並列実行マネージャーで実行
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks,
            max_concurrent=max_concurrent or min(len(symbols), 15),
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"現在価格取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        return symbol_results

    def parallel_get_company_info(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の企業情報を並列取得
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return {symbol: self.get_company_info(symbol) for symbol in symbols}

        self.logger.info(f"並列企業情報取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (self.get_company_info, (symbol,), {})
            tasks.append(task)

        # 並列実行マネージャーで実行
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks,
            max_concurrent=max_concurrent or min(len(symbols), 8),
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"企業情報取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        return symbol_results

    def _sequential_get_historical_data(
        self, symbols: List[str], period: str, interval: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """シーケンシャル実行フォールバック（並列処理が利用できない場合）"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_historical_data(symbol, period, interval)
            except Exception as e:
                self.logger.error(f"シーケンシャル取得失敗: {symbol}, エラー: {e}")
                results[symbol] = None
        return results

    def clear_all_caches(self) -> None:
        """すべてのキャッシュをクリア"""
        self.clear_all_performance_caches()


# 後方互換性のためのエイリアス
StockFetcher = LegacyStockFetcher