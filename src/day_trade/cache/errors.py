"""
キャッシュエラーハンドリング

統合キャッシュシステムのエラークラスとサーキットブレーカー実装を提供します。
元のcache_utils.pyからエラー関連機能を分離。
"""

import threading
import time
from enum import Enum
from typing import Optional

from .config import get_cache_config
from src.day_trade.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheError(Exception):
    """キャッシュ操作の基底例外クラス"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: 追加の詳細情報
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class CacheCircuitBreakerError(CacheError):
    """サーキットブレーカーによるエラー"""

    def __init__(self, message: str = "Circuit breaker is open", **kwargs):
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", **kwargs)


class CacheTimeoutError(CacheError):
    """キャッシュ操作のタイムアウトエラー"""

    def __init__(
        self,
        message: str = "Cache operation timed out",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="CACHE_TIMEOUT", **kwargs)
        if timeout:
            self.details["timeout"] = timeout


class CircuitBreakerState(Enum):
    """サーキットブレーカーの状態"""

    CLOSED = "closed"  # 正常状態
    OPEN = "open"  # 障害状態（呼び出しブロック）
    HALF_OPEN = "half_open"  # 試行状態（限定的な呼び出し許可）


class CacheCircuitBreaker:
    """
    キャッシュ操作用サーキットブレーカー

    失敗が連続して発生した場合に、一時的に操作をブロックし
    システム全体への影響を防ぎます。
    """

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
        half_open_max_calls: Optional[int] = None,
        config=None,
    ):
        """
        Args:
            failure_threshold: 失敗しきい値
            recovery_timeout: 回復タイムアウト（秒）
            half_open_max_calls: HALF_OPEN状態での最大コール数
            config: キャッシュ設定
        """
        if config is None:
            config = get_cache_config()

        self.failure_threshold = failure_threshold or config.failure_threshold
        self.recovery_timeout = recovery_timeout or config.recovery_timeout
        self.half_open_max_calls = half_open_max_calls or config.half_open_max_calls

        # 状態管理
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0

        # スレッドセーフティ
        self._lock = threading.RLock()

        logger.debug(
            f"Circuit breaker initialized: threshold={self.failure_threshold}, "
            f"timeout={self.recovery_timeout}, half_open_max={self.half_open_max_calls}"
        )

    @property
    def state(self) -> CircuitBreakerState:
        """現在の状態を取得"""
        with self._lock:
            self._update_state()
            return self._state

    @property
    def failure_count(self) -> int:
        """現在の失敗回数を取得"""
        with self._lock:
            return self._failure_count

    def _update_state(self) -> None:
        """状態を更新（内部メソッド、ロック必須）"""
        current_time = time.time()

        if (
            self._state == CircuitBreakerState.OPEN
            and current_time - self._last_failure_time >= self.recovery_timeout
        ):
            # OPEN状態で回復タイムアウトが経過した場合
            self._state = CircuitBreakerState.HALF_OPEN
            self._half_open_calls = 0
            logger.info("Circuit breaker transitioned to HALF_OPEN")

    def is_call_allowed(self) -> bool:
        """呼び出しが許可されるかチェック"""
        with self._lock:
            self._update_state()

            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """成功を記録"""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                # HALF_OPEN状態での成功
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("Circuit breaker recovered to CLOSED")
            elif self._state == CircuitBreakerState.CLOSED:
                # 連続失敗カウンターをリセット
                if self._failure_count > 0:
                    self._failure_count = 0

    def record_failure(self) -> None:
        """失敗を記録"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # HALF_OPEN状態での失敗
                self._state = CircuitBreakerState.OPEN
                self._half_open_calls = 0
                logger.warning("Circuit breaker failed during HALF_OPEN, back to OPEN")
            elif self._state == CircuitBreakerState.CLOSED:
                # CLOSED状態での失敗チェック
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker opened due to {self._failure_count} failures "
                        f"(threshold: {self.failure_threshold})"
                    )

    def call(self, func, *args, **kwargs):
        """
        サーキットブレーカー保護付きで関数を呼び出し

        Args:
            func: 呼び出す関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の戻り値

        Raises:
            CacheCircuitBreakerError: 回路が開いている場合
        """
        if not self.is_call_allowed():
            raise CacheCircuitBreakerError(
                f"Circuit breaker is {self.state.value}, calls not allowed"
            )

        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def reset(self) -> None:
        """サーキットブレーカーをリセット"""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0
            self._half_open_calls = 0

        logger.info("Circuit breaker manually reset")

    def get_stats(self) -> dict:
        """統計情報を取得"""
        with self._lock:
            self._update_state()
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "time_until_half_open": max(
                    0, self.recovery_timeout - (time.time() - self._last_failure_time)
                )
                if self._state == CircuitBreakerState.OPEN
                else 0,
            }

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"CacheCircuitBreaker(state={self._state.value}, "
                f"failures={self._failure_count}/{self.failure_threshold})"
            )


# グローバルサーキットブレーカー管理
_global_circuit_breaker: Optional[CacheCircuitBreaker] = None
_breaker_lock = threading.RLock()


def get_cache_circuit_breaker() -> CacheCircuitBreaker:
    """
    グローバルキャッシュサーキットブレーカーを取得

    Returns:
        CacheCircuitBreaker: サーキットブレーカーインスタンス
    """
    global _global_circuit_breaker

    if _global_circuit_breaker is None:
        with _breaker_lock:
            if _global_circuit_breaker is None:
                config = get_cache_config()
                if config.enable_circuit_breaker:
                    _global_circuit_breaker = CacheCircuitBreaker(config=config)
                else:
                    # サーキットブレーカーが無効の場合はダミー実装
                    _global_circuit_breaker = _DummyCircuitBreaker()

    return _global_circuit_breaker


def set_cache_circuit_breaker(breaker: CacheCircuitBreaker) -> None:
    """グローバルサーキットブレーカーを設定"""
    global _global_circuit_breaker

    with _breaker_lock:
        _global_circuit_breaker = breaker


def reset_cache_circuit_breaker() -> None:
    """グローバルサーキットブレーカーをリセット（テスト用）"""
    global _global_circuit_breaker

    with _breaker_lock:
        if _global_circuit_breaker:
            _global_circuit_breaker.reset()
        _global_circuit_breaker = None


class _DummyCircuitBreaker:
    """サーキットブレーカー無効時のダミー実装"""

    @property
    def state(self):
        return CircuitBreakerState.CLOSED

    @property
    def failure_count(self):
        return 0

    def is_call_allowed(self):
        return True

    def record_success(self):
        pass

    def record_failure(self):
        pass

    def call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def reset(self):
        pass

    def get_stats(self):
        return {
            "state": "disabled",
            "failure_count": 0,
            "failure_threshold": 0,
            "recovery_timeout": 0,
            "half_open_max_calls": 0,
            "half_open_calls": 0,
            "last_failure_time": 0,
            "time_until_half_open": 0,
        }

    def __repr__(self):
        return "DummyCircuitBreaker(disabled)"
