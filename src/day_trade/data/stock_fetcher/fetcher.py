"""
株価データ取得メインクラス
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from ...utils.logging_config import (
    log_api_call,
    log_error_with_context,
    log_performance_metric,
)

from .cache import cache_with_ttl
from .exceptions import DataNotFoundError, InvalidSymbolError
from .fetcher_base import StockFetcherBase


class StockFetcher(StockFetcherBase):
    """株価データ取得クラス（完全版）"""

    @cache_with_ttl(30)  # 30秒キャッシュ
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在の株価情報を取得

        Args:
            code: 証券コード

        Returns:
            価格情報の辞書（現在値、前日終値、変化額、変化率など）

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
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

        Args:
            code: 証券コード
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）

        Returns:
            価格データのDataFrame（Open, High, Low, Close, Volume）

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
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
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ]
        valid_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
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

        Args:
            code: 証券コード
            start_date: 開始日
            end_date: 終了日
            interval: 間隔

        Returns:
            価格データのDataFrame

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
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

    @cache_with_ttl(3600)  # 1時間キャッシュ
    def get_company_info(self, code: str) -> Optional[Dict[str, any]]:
        """
        企業情報を取得

        Args:
            code: 証券コード

        Returns:
            企業情報の辞書

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
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