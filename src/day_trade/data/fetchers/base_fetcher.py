"""
ベースフェッチャー

データ取得の基本機能・リトライ・エラーハンドリング
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ...utils.exceptions import (
    NetworkError,
    StockFetcherError,
    handle_network_exception,
)
from ...utils.logging_config import get_context_logger, get_performance_logger

logger = get_context_logger(__name__)


class BaseFetcher(ABC):
    """データ取得の基底クラス"""

    def __init__(
        self,
        cache_size: int = 128,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初期化

        Args:
            cache_size: LRUキャッシュサイズ
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        self.cache_size = cache_size
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # ロガー初期化
        self.logger = get_context_logger(__name__)
        self.performance_logger = get_performance_logger(__name__)

        # デバッグモード設定
        self.enable_debug_logging = os.getenv("STOCK_FETCHER_DEBUG", "false").lower() == "true"

        # リトライ統計
        self.retry_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "retry_success": 0,
            "errors_by_type": {},
        }

        logger.info(
            f"BaseFetcher初期化: cache_size={cache_size}, "
            f"retry_count={retry_count}, retry_delay={retry_delay}"
        )

    @abstractmethod
    def _fetch_data(self, symbol: str, **kwargs) -> Any:
        """
        データ取得の実装（サブクラスで実装）

        Args:
            symbol: 銘柄コード
            **kwargs: 追加パラメータ

        Returns:
            取得データ
        """
        pass

    def _record_request_start(self) -> None:
        """リクエスト開始を記録"""
        self.retry_stats["total_requests"] += 1

    def _record_request_success(self) -> None:
        """リクエスト成功を記録"""
        self.retry_stats["successful_requests"] += 1

    def _record_request_failure(self, error: Exception) -> None:
        """リクエスト失敗を記録"""
        self.retry_stats["failed_requests"] += 1
        error_type = type(error).__name__
        self.retry_stats["errors_by_type"][error_type] = (
            self.retry_stats["errors_by_type"].get(error_type, 0) + 1
        )

    def _record_retry_attempt(self) -> None:
        """リトライ試行を記録"""
        self.retry_stats["total_retries"] += 1

    def _record_retry_success(self) -> None:
        """リトライ成功を記録"""
        self.retry_stats["retry_success"] += 1

    def get_retry_stats(self) -> Dict[str, Any]:
        """リトライ統計を取得"""
        stats = self.retry_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["failure_rate"] = stats["failed_requests"] / stats["total_requests"]
        if stats["total_retries"] > 0:
            stats["retry_success_rate"] = stats["retry_success"] / stats["total_retries"]
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
            before_sleep=before_sleep_log(self.logger, 30),  # WARNING level
            after=after_log(self.logger, 20),  # INFO level
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
            self._handle_error(e)

    def _is_retryable_error(self, error: Exception) -> bool:
        """リトライ可能なエラーかどうかを判定"""
        import requests.exceptions as req_exc

        # ネットワーク関連エラー
        if isinstance(error, (req_exc.ConnectionError, req_exc.Timeout)):
            return True

        # HTTPエラーの場合はステータスコードを確認
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
        """エラーを適切な例外クラスに変換して再発生"""
        import requests.exceptions as req_exc

        # 既存のカスタム例外はそのまま再発生
        if isinstance(error, (StockFetcherError, NetworkError)):
            raise error

        # ネットワーク関連の例外を統一的に処理
        if isinstance(error, (req_exc.ConnectionError, req_exc.Timeout, req_exc.HTTPError)):
            try:
                converted_error = handle_network_exception(error)
                raise converted_error
            except NetworkError:
                raise
            except Exception:
                pass

        # 文字列ベースの判定（後方互換性）
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
            )

        # その他のエラーは一般的なStockFetcherErrorとして処理
        raise StockFetcherError(
            message=f"データ取得エラー: {error}",
            error_code="FETCH_ERROR",
            details={"original_error": str(error)},
        )

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        if hasattr(self, "_get_ticker"):
            self._get_ticker.cache_clear()
        logger.info("キャッシュクリア完了")

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        if hasattr(self, "_get_ticker"):
            cache_info = self._get_ticker.cache_info()
            return {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize,
                "currsize": cache_info.currsize,
                "hit_rate": (
                    cache_info.hits / (cache_info.hits + cache_info.misses)
                    if (cache_info.hits + cache_info.misses) > 0
                    else 0.0
                ),
            }
        return {}

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        self.retry_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "retry_success": 0,
            "errors_by_type": {},
        }
        logger.info("統計情報リセット完了")

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要を取得"""
        retry_stats = self.get_retry_stats()
        cache_info = self.get_cache_info()

        return {
            "retry_stats": retry_stats,
            "cache_info": cache_info,
            "configuration": {
                "cache_size": self.cache_size,
                "retry_count": self.retry_count,
                "retry_delay": self.retry_delay,
                "debug_enabled": self.enable_debug_logging,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            stats = self.get_retry_stats()

            # 健康状態の評価
            health_score = 100
            issues = []

            # 成功率チェック
            if stats.get("success_rate", 1.0) < 0.8:
                health_score -= 20
                issues.append(f"低い成功率: {stats.get('success_rate', 0):.2%}")

            # エラー率チェック
            if stats.get("failure_rate", 0.0) > 0.1:
                health_score -= 15
                issues.append(f"高いエラー率: {stats.get('failure_rate', 0):.2%}")

            # キャッシュ効率チェック
            cache_info = self.get_cache_info()
            if cache_info.get("hit_rate", 1.0) < 0.5:
                health_score -= 10
                issues.append(f"低いキャッシュ効率: {cache_info.get('hit_rate', 0):.2%}")

            health_status = "healthy"
            if health_score < 60:
                health_status = "critical"
            elif health_score < 80:
                health_status = "warning"

            return {
                "status": health_status,
                "score": health_score,
                "issues": issues,
                "stats": stats,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            return {
                "status": "error",
                "score": 0,
                "issues": [f"ヘルスチェック実行エラー: {e}"],
                "timestamp": time.time(),
            }
