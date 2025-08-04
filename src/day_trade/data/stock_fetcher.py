"""
株価データ取得モジュール
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得
"""

import logging
import time
from datetime import datetime
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

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
from ..utils.logging_config import (
    get_context_logger,
    log_api_call,
    log_error_with_context,
    log_performance_metric,
)


class DataCache:
    """高度なデータキャッシュクラス（フォールバック機能付き）"""

    def __init__(
        self,
        ttl_seconds: int = 60,
        max_size: int = 1000,
        stale_while_revalidate: int = 300,
    ):
        """
        Args:
            ttl_seconds: キャッシュの有効期限（秒）
            max_size: 最大キャッシュサイズ（LRU eviction）
            stale_while_revalidate: 期限切れ後もフォールバックとして利用可能な期間（秒）
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.stale_while_revalidate = stale_while_revalidate
        self._cache = {}
        self._access_order = []  # LRU tracking

        # パフォーマンス統計
        self._hit_count = 0
        self._miss_count = 0
        self._stale_hit_count = 0
        self._eviction_count = 0

    def get(self, key: str, allow_stale: bool = False) -> Optional[any]:
        """
        キャッシュから値を取得

        Args:
            key: キャッシュキー
            allow_stale: 期限切れキャッシュも許可するか
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            current_time = time.time()
            age = current_time - timestamp

            # フレッシュなキャッシュ
            if age < self.ttl_seconds:
                self._update_access_order(key)
                self._hit_count += 1
                return value

            # stale-while-revalidate期間内のキャッシュ
            elif allow_stale and age < self.ttl_seconds + self.stale_while_revalidate:
                self._update_access_order(key)
                self._stale_hit_count += 1
                return value

            # 完全に期限切れ
            else:
                self._remove_key(key)

        self._miss_count += 1
        return None

    def set(self, key: str, value: any) -> None:
        """キャッシュに値を設定（LRU eviction付き）"""
        current_time = time.time()

        # キャッシュサイズ制限の確認
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (value, current_time)
        self._update_access_order(key)

    def _update_access_order(self, key: str) -> None:
        """アクセス順序を更新（LRU tracking）"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self) -> None:
        """最も古いキャッシュエントリを削除"""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_key(lru_key)
            self._eviction_count += 1

    def _remove_key(self, key: str) -> None:
        """キーをキャッシュから完全に削除"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        self._access_order.clear()
        # 統計もリセット
        self._hit_count = 0
        self._miss_count = 0
        self._stale_hit_count = 0
        self._eviction_count = 0

    def get_cache_stats(self) -> dict:
        """キャッシュ統計を取得"""
        total_requests = self._hit_count + self._miss_count + self._stale_hit_count

        if total_requests == 0:
            hit_rate = 0.0
            stale_hit_rate = 0.0
        else:
            hit_rate = self._hit_count / total_requests
            stale_hit_rate = self._stale_hit_count / total_requests

        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "stale_hit_count": self._stale_hit_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "stale_hit_rate": stale_hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "eviction_count": self._eviction_count,
            "cache_utilization": len(self._cache) / self.max_size
            if self.max_size > 0
            else 0.0,
        }

    def optimize_cache_settings(self) -> dict:
        """キャッシュ統計に基づいた設定最適化の提案"""
        stats = self.get_cache_stats()
        recommendations = {}

        # ヒット率が低い場合の提案
        if stats["hit_rate"] < 0.5:
            recommendations["ttl_increase"] = (
                "TTLを延長してキャッシュヒット率を向上させることを検討"
            )

        # 退避回数が多い場合の提案
        if stats["eviction_count"] > stats["hit_count"] * 0.1:
            recommendations["size_increase"] = "キャッシュサイズを増加させることを検討"

        # キャッシュ使用率が低い場合の提案
        if stats["cache_utilization"] < 0.3:
            recommendations["size_decrease"] = (
                "キャッシュサイズを減少させてメモリ効率を向上"
            )

        # stale hitが多い場合の提案
        if stats["stale_hit_rate"] > 0.2:
            recommendations["stale_period_adjust"] = (
                "stale-while-revalidate期間の調整を検討"
            )

        return {"current_stats": stats, "recommendations": recommendations}

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self._cache)

    def get_cache_info(self) -> Dict[str, any]:
        """キャッシュ統計情報を取得"""
        current_time = time.time()
        fresh_count = 0
        stale_count = 0

        for _, (_, timestamp) in self._cache.items():
            age = current_time - timestamp
            if age < self.ttl_seconds:
                fresh_count += 1
            elif age < self.ttl_seconds + self.stale_while_revalidate:
                stale_count += 1

        return {
            "total_entries": len(self._cache),
            "fresh_entries": fresh_count,
            "stale_entries": stale_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "stale_while_revalidate": self.stale_while_revalidate,
        }


def cache_with_ttl(
    ttl_seconds: int, max_size: int = 1000, stale_while_revalidate: int = None
):
    """TTL付きキャッシュデコレータ（フォールバック機能付き改善版）"""
    if stale_while_revalidate is None:
        stale_while_revalidate = ttl_seconds * 5  # デフォルトでTTLの5倍

    cache = DataCache(ttl_seconds, max_size, stale_while_revalidate)
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

                # フレッシュキャッシュを試行
                cached_result = cache.get(cache_key, allow_stale=False)
                if cached_result is not None:
                    cache_logger.debug(f"フレッシュキャッシュヒット: {func.__name__}")
                    stats.record_hit()
                    return cached_result

                # APIリクエスト実行を試行
                try:
                    stats.record_miss()
                    result = func(*args, **kwargs)

                    # 結果をサニタイズしてキャッシュに保存
                    if result is not None:
                        sanitized_result = sanitize_cache_value(result)
                        cache.set(cache_key, sanitized_result)
                        stats.record_set()
                        cache_logger.debug(
                            f"新しいデータをキャッシュに保存: {func.__name__}"
                        )

                    return result

                except (APIError, NetworkError, DataError) as api_error:
                    # APIエラー時はstaleキャッシュをフォールバックとして使用
                    stale_result = cache.get(cache_key, allow_stale=True)
                    if stale_result is not None:
                        cache_logger.warning(
                            f"API失敗、staleキャッシュを使用: {func.__name__} - {api_error}"
                        )
                        stats.record_fallback()
                        return stale_result
                    else:
                        # staleキャッシュもない場合は例外を再発生
                        cache_logger.error(
                            f"API失敗かつキャッシュなし: {func.__name__} - {api_error}"
                        )
                        raise api_error

            except Exception as e:
                cache_logger.error(f"キャッシュ処理でエラーが発生: {e}")
                stats.record_error()
                # 重要でないエラーの場合は継続を試行
                if not isinstance(e, (APIError, NetworkError, DataError)):
                    return func(*args, **kwargs)
                raise e

        # キャッシュ管理メソッドを追加
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.reset_stats = stats.reset
        wrapper.get_cache_info = cache.get_cache_info

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
        self.cache_size = cache_size
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # ロガーを初期化
        self.logger = get_context_logger(__name__)

        # リトライ統計を初期化
        self.retry_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "retry_success": 0,
            "errors_by_type": {},
        }

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
            retry_count=retry_count,
        )

    def _create_ticker(self, symbol: str) -> yf.Ticker:
        """Tickerオブジェクトを作成（内部メソッド）"""
        return yf.Ticker(symbol)

    def _record_request_start(self):
        """リクエスト開始を記録"""
        self.retry_stats["total_requests"] += 1

    def _record_request_success(self):
        """リクエスト成功を記録"""
        self.retry_stats["successful_requests"] += 1

    def _record_request_failure(self, error: Exception):
        """リクエスト失敗を記録"""
        self.retry_stats["failed_requests"] += 1
        error_type = type(error).__name__
        self.retry_stats["errors_by_type"][error_type] = (
            self.retry_stats["errors_by_type"].get(error_type, 0) + 1
        )

    def _record_retry_attempt(self):
        """リトライ試行を記録"""
        self.retry_stats["total_retries"] += 1

    def _record_retry_success(self):
        """リトライ成功を記録"""
        self.retry_stats["retry_success"] += 1

    def get_retry_stats(self) -> Dict:
        """リトライ統計を取得"""
        stats = self.retry_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )
            stats["failure_rate"] = stats["failed_requests"] / stats["total_requests"]
        if stats["total_retries"] > 0:
            stats["retry_success_rate"] = (
                stats["retry_success"] / stats["total_retries"]
            )
        return stats

    def _create_retry_decorator(self):
        """tenacityを使用したリトライデコレータを作成"""
        return retry(
            stop=stop_after_attempt(self.retry_count),
            wait=wait_exponential(
                multiplier=self.retry_delay,
                min=self.retry_delay,
                max=60,  # 最大60秒待機
            ),
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            after=after_log(self.logger, logging.INFO),
            reraise=True,
        )

    def _retry_on_error(self, func, *args, **kwargs):
        """tenacityベースのリトライ機能（統計記録付き）"""
        self._record_request_start()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _execute_with_retry():
            # _record_request_success() は最終的な成功時のみ呼ばれるように
            # ここでは呼び出さない
            return func(*args, **kwargs)

        try:
            return _execute_with_retry()
        except Exception as e:
            self._record_request_failure(e)
            # _handle_errorで適切な例外に変換して再発生
            self._handle_error(e)

    def _is_retryable_error(self, error: Exception) -> bool:
        """リトライ可能なエラーかどうかを判定（改善版）"""
        import requests.exceptions as req_exc

        # 例外の型による判定を優先（より信頼性が高い）
        if isinstance(error, (req_exc.ConnectionError, req_exc.Timeout)):
            return True

        if (
            isinstance(error, req_exc.HTTPError)
            and hasattr(error, "response")
            and error.response is not None
        ):
            status_code = error.response.status_code
            # リトライ可能なHTTPステータスコード
            retryable_codes = [500, 502, 503, 504, 408, 429]
            return status_code in retryable_codes

        return False

    def _handle_error(self, error: Exception) -> None:
        """エラーを適切な例外クラスに変換して再発生（改善版）"""
        import requests.exceptions as req_exc

        # 既存のカスタム例外はそのまま再発生
        if isinstance(
            error, (StockFetcherError, InvalidSymbolError, DataNotFoundError)
        ):
            raise error

        # ネットワーク関連の例外を統一的に処理
        if isinstance(
            error, (req_exc.ConnectionError, req_exc.Timeout, req_exc.HTTPError)
        ):
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

        if (
            "connection" in error_message
            or "network" in error_message
            or "timeout" in error_message
        ):
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
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
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
            raise InvalidSymbolError(
                f"開始日が終了日以降です: {start_date} >= {end_date}"
            )

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
            price_logger = self.logger.bind(
                operation="get_current_price", stock_code=code
            )
            price_logger.info("現在価格取得開始")

            start_time = time.time()

            try:
                self._validate_symbol(code)
                symbol = self._format_symbol(code)
                ticker = self._get_ticker(symbol)

                # API呼び出しログ
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
        # A common recommendation is (number of cores * 2) + 1, or just a small fixed number for I/O bound tasks
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


# 使用例
if __name__ == "__main__":
    # ロギング設定
    from ..utils.logging_config import get_context_logger, setup_logging

    setup_logging()

    logger = get_context_logger(__name__)

    # インスタンス作成
    fetcher = StockFetcher()

    # トヨタ自動車の現在価格を取得
    logger.info("=== 現在価格データ取得例 ===")
    current = fetcher.get_current_price("7203")
    if current:
        logger.info(
            "Current price data retrieved",
            symbol=current["symbol"],
            current_price=current["current_price"],
            change=current["change"],
            change_percent=current["change_percent"],
        )

    # ヒストリカルデータを取得
    logger.info("=== ヒストリカルデータ取得例 ===")
    hist = fetcher.get_historical_data("7203", period="5d", interval="1d")
    if hist is not None:
        logger.info(
            "Historical data retrieved",
            symbol="7203",
            period="5d",
            data_points=len(hist),
        )

    # 企業情報を取得
    logger.info("=== 企業情報取得例 ===")
    info = fetcher.get_company_info("7203")
    if info:
        logger.info(
            "Company info retrieved",
            symbol="7203",
            name=info.get("name"),
            sector=info.get("sector"),
            industry=info.get("industry"),
        )
