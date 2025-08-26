"""
株価データ取得ベースクラス
基本的な初期化、リトライ、エラー処理機能
"""

import logging
import os
import time
from functools import lru_cache
from typing import Dict

from ...utils.exceptions import (
    APIError,
    DataError,
    NetworkError,
    ValidationError,
    handle_network_exception,
)
from ...utils.logging_config import (
    get_context_logger,
    get_performance_logger,
)
from ...utils.yfinance_import import get_yfinance

from .exceptions import DataNotFoundError, InvalidSymbolError, StockFetcherError
from .base_validators import BaseValidators

# yfinance統一インポート - Issue #614対応
yf, YFINANCE_AVAILABLE = get_yfinance()

# tenacity インポート
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


class StockFetcherBase(BaseValidators):
    """株価データ取得ベースクラス"""

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
        super().__init__()
        
        self.cache_size = cache_size
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # ロガーを初期化
        self.logger = get_context_logger(__name__)
        self.performance_logger = get_performance_logger(__name__)

        # 条件付きデバッグロギング（本番環境では無効化）
        self.enable_debug_logging = (
            os.getenv("STOCK_FETCHER_DEBUG", "false").lower() == "true"
        )

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

        # 適応的キャッシュ調整用の統計とタイマー
        self.cache_adjustment_interval = int(
            os.getenv("CACHE_ADJUSTMENT_INTERVAL", "3600")
        )  # 1時間
        self.last_cache_adjustment = time.time()
        self.auto_cache_tuning_enabled = (
            os.getenv("AUTO_CACHE_TUNING", "true").lower() == "true"
        )

        self.logger.info(
            "StockFetcher初期化完了",
            extra={
                "cache_size": cache_size,
                "price_cache_ttl": price_cache_ttl,
                "historical_cache_ttl": historical_cache_ttl,
                "retry_count": retry_count,
                "auto_cache_tuning": self.auto_cache_tuning_enabled,
            },
        )

    def _create_ticker(self, symbol: str):
        """Tickerオブジェクトを作成（内部メソッド）- Issue #614対応"""
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinanceが利用できません。pip install yfinanceでインストールしてください。"
            )
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
            return func(*args, **kwargs)

        try:
            result = _execute_with_retry()
            self._record_request_success()
            return result
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

    def clear_all_caches(self) -> None:
        """すべてのキャッシュをクリア"""
        self._get_ticker.cache_clear()

        # メソッドレベルのキャッシュもクリア
        if hasattr(self, "get_current_price") and hasattr(
            self.get_current_price, "clear_cache"
        ):
            self.get_current_price.clear_cache()
        if hasattr(self, "get_historical_data") and hasattr(
            self.get_historical_data, "clear_cache"
        ):
            self.get_historical_data.clear_cache()
        if hasattr(self, "get_historical_data_range") and hasattr(
            self.get_historical_data_range, "clear_cache"
        ):
            self.get_historical_data_range.clear_cache()

        self.logger.info("すべてのキャッシュをクリアしました")