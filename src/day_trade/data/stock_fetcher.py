"""
株価データ取得モジュール
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得
"""

import time
from datetime import datetime
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Union

import pandas as pd
import requests.exceptions as req_exc
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from ..utils.cache_utils import (
    CacheStats,
    generate_safe_cache_key,
    sanitize_cache_value,
    validate_cache_key,
)
from ..utils.exceptions import (
    APIError,
    DataError,
    NetworkError,
    ValidationError,
    handle_network_exception,
)
from ..utils.logging_config import get_context_logger, log_api_call, log_error_with_context, log_performance_metric


def _is_retryable_error(error: Exception) -> bool:
    """リトライ可能なエラーかどうかを判定"""
    if isinstance(error, (req_exc.ConnectionError, req_exc.Timeout)):
        return True

    if isinstance(error, req_exc.HTTPError):
        if hasattr(error, "response") and error.response is not None:
            status_code = error.response.status_code
            retryable_codes = [500, 502, 503, 504, 408, 429]
            return status_code in retryable_codes

    # yfinanceがrequests以外のエラーを出す可能性も考慮
    error_message = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "network",
        "temporary",
        "read timeout",
        "connection timeout",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(pattern in error_message for pattern in retryable_patterns)


# tenacityリトライデコレータの共通設定
retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_error),
    before_sleep=lambda retry_state: get_context_logger(__name__).warning(
        f"Retrying API call due to {retry_state.outcome.exception()}",
        attempt=retry_state.attempt_number,
        wait_time=retry_state.next_action.sleep,
    ),
)


class DataCache:
    """データキャッシュクラス"""

    def __init__(self, ttl_seconds: int = 60):
        """
        Args:
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.ttl_seconds = ttl_seconds
        self._cache = {}

    def get(self, key: str) -> Optional[any]:
        """キャッシュから値を取得"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                # 期限切れのキャッシュを削除
                del self._cache[key]
        return None

    def set(self, key: str, value: any) -> None:
        """キャッシュに値を設定"""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self._cache)


def cache_with_ttl(ttl_seconds: int):
    """TTL付きキャッシュデコレータ（改善版）"""
    cache = DataCache(ttl_seconds)
    stats = CacheStats()
    # キャッシュデコレータ用のロガーを取得
    cache_logger = get_context_logger(__name__)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 安全なキャッシュキーを生成
                cache_key = generate_safe_cache_key(func.__name__, *args, **kwargs)

                # キーの妥当性を検証
                if not validate_cache_key(cache_key):
                    cache_logger.warning(
                        f"Invalid cache key generated for {func.__name__}, skipping cache"
                    )
                    stats.record_error()
                    return func(*args, **kwargs)

                # キャッシュから取得を試行
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    cache_logger.debug(f"キャッシュヒット: {func.__name__}")
                    stats.record_hit()
                    return cached_result

                # キャッシュにない場合は実行
                stats.record_miss()
                result = func(*args, **kwargs)

                # 結果をサニタイズしてキャッシュに保存
                if result is not None:
                    sanitized_result = sanitize_cache_value(result)
                    cache.set(cache_key, sanitized_result)
                    stats.record_set()
                    cache_logger.debug(f"キャッシュに保存: {func.__name__}")

                return result

            except Exception as e:
                cache_logger.error(f"キャッシュ処理でエラーが発生: {e}")
                stats.record_error()
                # キャッシュエラーでも関数実行は継続
                return func(*args, **kwargs)

        # キャッシュ管理メソッドを追加
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.reset_stats = stats.reset

        return wrapper

    return decorator


# 従来の例外クラス（下位互換性のため残す）
class StockFetcherError(APIError):
    """株価データ取得エラー（下位互換性のため）"""

    pass


class InvalidSymbolError(ValidationError):
    """無効なシンボルエラー（下位互換性のため）"""

    pass


class DataNotFoundError(DataError):
    """データが見つからないエラー（下位互換性のため）"""

    pass


class StockFetcher:
    """株価データ取得クラス"""

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
    ):
        """
        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
        """
        self.cache_size = cache_size

        # ロガーを初期化
        self.logger = get_context_logger(__name__)

        # LRUキャッシュを動的に設定
        self._get_ticker = lru_cache(maxsize=cache_size)(self._create_ticker)

        # キャッシュTTLを設定
        self.price_cache_ttl = price_cache_ttl
        self.historical_cache_ttl = historical_cache_ttl

        self.logger.info(
            "StockFetcher初期化完了",
            cache_size=cache_size,
            price_cache_ttl=price_cache_ttl,
            historical_cache_ttl=historical_cache_ttl,
        )

    def _create_ticker(self, symbol: str) -> yf.Ticker:
        """Tickerオブジェクトを作成（内部メソッド）"""
        return yf.Ticker(symbol)

    def _handle_error(self, error: Exception) -> None:
        """エラーを適切な例外クラスに変換して再発生（改善版）"""
        # 既存のカスタム例外はそのまま再発生
        if isinstance(error, (StockFetcherError, InvalidSymbolError, DataNotFoundError)):
            raise error

        # ネットワーク関連の例外を統一的に処理
        if isinstance(error, (req_exc.ConnectionError, req_exc.Timeout, req_exc.HTTPError)):
            try:
                converted_error = handle_network_exception(error)
                raise converted_error
            except NetworkError:
                # handle_network_exception で処理できた場合
                raise
            except Exception:
                # handle_network_exception で処理できなかった場合は従来の処理
                pass

        # 従来の文字列ベースの判定（後方互換性のため）
        error_message = str(error).lower()

        if "connection" in error_message or "network" in error_message or "timeout" in error_message:
            raise NetworkError(
                message=f"ネットワークエラー: {error}",
                error_code="NETWORK_ERROR",
                details={"original_error": str(error)},
            ) from error
        elif "not found" in error_message or "invalid" in error_message:
            raise InvalidSymbolError(
                message=f"無効なシンボル: {error}",
                error_code="INVALID_SYMBOL",
                details={"original_error": str(error)},
            ) from error
        elif "no data" in error_message or "empty" in error_message:
            raise DataNotFoundError(
                message=f"データが見つかりません: {error}",
                error_code="DATA_NOT_FOUND",
                details={"original_error": str(error)},
            ) from error
        else:
            raise StockFetcherError(
                message=f"株価データ取得エラー: {error}",
                error_code="STOCK_FETCH_ERROR",
                details={"original_error": str(error)},
            ) from error

    def _validate_symbol(self, symbol: str) -> None:
        """シンボルの妥当性をチェック"""
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"無効なシンボル: {symbol}")

        if len(symbol) < 2:
            raise InvalidSymbolError(f"シンボルが短すぎます: {symbol}")

    def _validate_date_range(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> None:
        """日付範囲の妥当性をチェック"""
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な開始日形式: {start_date}") from e

        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な終了日形式: {end_date}") from e

        if start_date >= end_date:
            raise InvalidSymbolError(f"開始日が終了日以降です: {start_date} >= {end_date}")

        if end_date > datetime.now():
            self.logger.warning(f"終了日が未来の日付です: {end_date}")

    def clear_all_caches(self) -> None:
        """すべてのキャッシュをクリア"""
        self._get_ticker.cache_clear()

        # メソッドレベルのキャッシュもクリア
        if hasattr(self.get_current_price, "clear_cache"):
            self.get_current_price.clear_cache()
        if hasattr(self.get_historical_data, "clear_cache"):
            self.get_historical_data.clear_cache()
        if hasattr(self.get_historical_data_range, "clear_cache"):
            self.get_historical_data_range.clear_cache()

        self.logger.info("すべてのキャッシュをクリアしました")

    def _format_symbol(self, code: str, market: str = "T") -> str:
        """
        証券コードをyfinance形式にフォーマット

        Args:
            code: 証券コード（例：7203）
            market: 市場コード（T:東証、デフォルト）

        Returns:
            フォーマット済みシンボル（例：7203.T）
        """
        # すでに市場コードが付いている場合はそのまま返す
        if "." in code:
            return code
        return f"{code}.{market}"

    @cache_with_ttl(30)  # 30秒キャッシュ
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在の株価情報を取得
        """

        @retry_decorator
        def _get_price():
            price_logger = self.logger.bind(operation="get_current_price", stock_code=code)
            price_logger.info("現在価格取得開始")

            start_time = time.time()

            try:
                self._validate_symbol(code)
                symbol = self._format_symbol(code)
                ticker = self._get_ticker(symbol)

                log_api_call("yfinance", "GET", f"ticker.info for {symbol}")
                info = ticker.info

                if not info or len(info) < 5:
                    price_logger.error("企業情報取得失敗", info_size=len(info) if info else 0)
                    raise DataNotFoundError(f"企業情報を取得できません: {symbol}")

                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                previous_close = info.get("previousClose")

                if current_price is None:
                    price_logger.error("現在価格データなし", available_keys=list(info.keys())[:10])
                    raise DataNotFoundError(f"現在価格を取得できません: {symbol}")

                change = current_price - previous_close if previous_close else 0
                change_percent = (change / previous_close * 100) if previous_close else 0

                elapsed_time = (time.time() - start_time) * 1000
                log_performance_metric("price_fetch_time", elapsed_time, "ms", stock_code=code)

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

                price_logger.info(
                    "現在価格取得完了",
                    current_price=current_price,
                    change_percent=change_percent,
                    elapsed_ms=elapsed_time,
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
                price_logger.error("現在価格取得失敗", error=str(e), elapsed_ms=elapsed_time)
                self._handle_error(e)

        return _get_price()

    @cache_with_ttl(300)  # 5分キャッシュ
    def get_historical_data(
        self,
        code: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        ヒストリカルデータを取得
        """

        @retry_decorator
        def _get_historical():
            try:
                self._validate_symbol(code)
                self._validate_period_interval(period, interval)

                symbol = self._format_symbol(code)
                ticker = self._get_ticker(symbol)

                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    raise DataNotFoundError(f"ヒストリカルデータが空です: {symbol}")

                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                if len(df) == 0:
                    raise DataNotFoundError(f"有効なデータがありません: {symbol}")

                return df
            except Exception as e:
                self._handle_error(e)

        return _get_historical()

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """期間と間隔の妥当性をチェック"""
        valid_periods = [
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
        ]
        valid_intervals = [
            "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
        ]

        if period not in valid_periods:
            raise InvalidSymbolError(f"無効な期間: {period}. 有効な値: {valid_periods}")

        if interval not in valid_intervals:
            raise InvalidSymbolError(f"無効な間隔: {interval}. 有効な値: {valid_intervals}")

        if interval.endswith("m") and period not in ["1d", "5d"]:
            raise InvalidSymbolError("分足データは1d または 5d 期間のみサポートされています")

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

        @retry_decorator
        def _get_historical_range():
            try:
                self._validate_symbol(code)
                self._validate_date_range(start_date, end_date)

                symbol = self._format_symbol(code)
                ticker = self._get_ticker(symbol)

                start_dt = start_date
                end_dt = end_date
                if isinstance(start_date, str):
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if isinstance(end_date, str):
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

                df = ticker.history(start=start_dt, end=end_dt, interval=interval)

                if df.empty:
                    raise DataNotFoundError(f"指定期間のヒストリカルデータが空です: {symbol}")

                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                return df
            except Exception as e:
                self._handle_error(e)

        return _get_historical_range()

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not codes or not isinstance(codes, list):
            raise InvalidSymbolError(f"無効な銘柄コードリスト: {codes}")

        results = {}
        failed_codes = []

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
    def get_company_info(self, code: str) -> Optional[Dict[str, any]]:
        """
        企業情報を取得
        """

        @retry_decorator
        def _get_company_info():
            try:
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
            except Exception as e:
                self._handle_error(e)

        return _get_company_info()


# 使用例
if __name__ == "__main__":
    # ロギング設定
    from ..utils.logging_config import setup_logging

    setup_logging()

    # インスタンス作成
    fetcher = StockFetcher()

    # トヨタ自動車の現在価格を取得
    print("=== 現在価格 ===")
    current = fetcher.get_current_price("7203")
    if current:
        print(f"銘柄: {current['symbol']}")
        print(f"現在値: {current['current_price']:,.0f}円")
        print(f"前日比: {current['change']:+,.0f}円 ({current['change_percent']:+.2f}%)")

    # ヒストリカルデータを取得
    print("\n=== ヒストリカルデータ（過去5日） ===")
    hist = fetcher.get_historical_data("7203", period="5d", interval="1d")
    if hist is not None:
        print(hist[["Open", "High", "Low", "Close", "Volume"]])

    # 企業情報を取得
    print("\n=== 企業情報 ===")
    info = fetcher.get_company_info("7203")
    if info:
        print(f"企業名: {info['name']}")
        print(f"セクター: {info['sector']}")
        print(f"業種: {info['industry']}")
