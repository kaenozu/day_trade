"""
キャッシュサーキットブレーカーモジュール

キャッシュ操作の信頼性を向上させるためのサーキットブレーカーパターンを実装します。
連続的な失敗を検知し、システムを一時的に保護する機能を提供します。
"""

import threading
import time
from typing import Any, Callable

from ..logging_config import get_logger
from .constants import CacheCircuitBreakerError

logger = get_logger(__name__)


class CacheCircuitBreaker:
    """キャッシュ操作用のサーキットブレーカー"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Args:
            failure_threshold: サーキットブレーカーが開くまでの失敗回数閾値
            recovery_timeout: OPEN状態からHALF_OPEN状態への回復時間（秒）
            half_open_max_calls: HALF_OPEN状態での最大試行回数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()

        logger.debug(
            f"CacheCircuitBreaker initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}, "
            f"half_open_max_calls={half_open_max_calls}"
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        サーキットブレーカー付きで関数を呼び出し

        Args:
            func: 実行する関数
            *args: 関数の位置引数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の実行結果

        Raises:
            CacheCircuitBreakerError: サーキットブレーカーが作動した場合
        """
        with self._lock:
            current_state = self._check_and_update_state()

            if current_state == "OPEN":
                raise CacheCircuitBreakerError(
                    "Circuit breaker is OPEN", self._state, self._failure_count
                )

            elif current_state == "HALF_OPEN" and self._half_open_calls >= self.half_open_max_calls:
                raise CacheCircuitBreakerError(
                    "Circuit breaker HALF_OPEN limit exceeded",
                    self._state,
                    self._failure_count,
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _check_and_update_state(self) -> str:
        """
        現在の状態をチェックし、必要に応じて更新

        Returns:
            更新後の状態
        """
        current_time = time.time()

        if self._state == "OPEN":
            if current_time - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
                logger.info("Circuit breaker state changed from OPEN to HALF_OPEN")

        return self._state

    def _on_success(self) -> None:
        """成功時の処理"""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._half_open_calls += 1
                logger.debug(f"HALF_OPEN success call: {self._half_open_calls}/{self.half_open_max_calls}")
                
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info("Circuit breaker state changed from HALF_OPEN to CLOSED")
            elif self._state == "CLOSED":
                # 成功時は失敗カウントを徐々に減らす
                self._failure_count = max(0, self._failure_count - 1)
                if self._failure_count == 0:
                    logger.debug("Circuit breaker failure count reset to 0")

    def _on_failure(self) -> None:
        """失敗時の処理"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker failure recorded: "
                f"count={self._failure_count}, threshold={self.failure_threshold}"
            )

            # 失敗回数が閾値を超えた場合、またはHALF_OPEN状態での失敗の場合
            if (
                self._failure_count >= self.failure_threshold
                or self._state == "HALF_OPEN"
            ):
                old_state = self._state
                self._state = "OPEN"
                if old_state != "OPEN":
                    logger.error(
                        f"Circuit breaker state changed from {old_state} to OPEN "
                        f"(failure_count={self._failure_count})"
                    )

    @property
    def state(self) -> str:
        """現在のサーキットブレーカー状態を取得"""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """現在の失敗カウントを取得"""
        with self._lock:
            return self._failure_count

    @property
    def is_available(self) -> bool:
        """サーキットブレーカーが利用可能かどうか"""
        with self._lock:
            current_state = self._check_and_update_state()
            return current_state in ("CLOSED", "HALF_OPEN")

    def reset(self) -> None:
        """サーキットブレーカーを初期状態にリセット"""
        with self._lock:
            old_state = self._state
            self._state = "CLOSED"
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0.0
            
            if old_state != "CLOSED":
                logger.info(f"Circuit breaker manually reset from {old_state} to CLOSED")

    def get_stats(self) -> dict:
        """サーキットブレーカーの統計情報を取得"""
        with self._lock:
            current_time = time.time()
            time_since_last_failure = (
                current_time - self._last_failure_time if self._last_failure_time > 0 else 0
            )
            
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "half_open_calls": self._half_open_calls,
                "half_open_max_calls": self.half_open_max_calls,
                "recovery_timeout": self.recovery_timeout,
                "time_since_last_failure": time_since_last_failure,
                "is_available": self.is_available,
                "last_failure_time": self._last_failure_time,
            }


# グローバルサーキットブレーカーインスタンス（遅延初期化）
_cache_circuit_breaker = None
_circuit_breaker_lock = threading.Lock()


def get_cache_circuit_breaker() -> CacheCircuitBreaker:
    """
    キャッシュサーキットブレーカーを取得（遅延初期化・シングルトン）

    Returns:
        CacheCircuitBreakerインスタンス
    """
    global _cache_circuit_breaker
    
    if _cache_circuit_breaker is None:
        with _circuit_breaker_lock:
            if _cache_circuit_breaker is None:  # double-checked locking
                _cache_circuit_breaker = CacheCircuitBreaker()
                logger.debug("Global CacheCircuitBreaker instance created")
    
    return _cache_circuit_breaker


def set_cache_circuit_breaker(circuit_breaker: CacheCircuitBreaker) -> None:
    """
    キャッシュサーキットブレーカーを設定（テスト用・依存性注入用）

    Args:
        circuit_breaker: 新しいCacheCircuitBreakerインスタンス
    """
    global _cache_circuit_breaker
    
    with _circuit_breaker_lock:
        _cache_circuit_breaker = circuit_breaker
        logger.debug("Global CacheCircuitBreaker instance replaced")


def reset_cache_circuit_breaker() -> None:
    """キャッシュサーキットブレーカーをリセット（テスト用）"""
    global _cache_circuit_breaker
    
    with _circuit_breaker_lock:
        if _cache_circuit_breaker is not None:
            _cache_circuit_breaker.reset()
            logger.debug("Global CacheCircuitBreaker instance reset")