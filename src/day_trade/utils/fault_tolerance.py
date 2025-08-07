#!/usr/bin/env python3
"""
フォールトトレランス（障害許容）システム

ネットワークエラー、API制限、データ不整合への対応
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from .logging_config import get_context_logger

logger = get_context_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """リトライ設定クラス"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter


class CircuitBreakerConfig:
    """サーキットブレーカー設定クラス"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout


class CircuitBreaker:
    """サーキットブレーカー実装"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """デコレータ実装"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"サーキットブレーカー {self.name} - HALF_OPEN状態")
                else:
                    raise Exception(
                        f"サーキットブレーカー {self.name} - OPEN状態（呼び出し拒否）"
                    )

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """リセット試行判定"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.config.recovery_timeout

    def _on_success(self):
        """成功時の処理"""
        if self.state == "HALF_OPEN":
            logger.info(f"サーキットブレーカー {self.name} - 復旧成功、CLOSED状態")
            self.state = "CLOSED"
        self.failure_count = 0

    def _on_failure(self):
        """失敗時の処理"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"サーキットブレーカー {self.name} - OPEN状態（{self.failure_count}回失敗）"
            )


def with_retry(
    config: RetryConfig = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    リトライデコレータ

    Args:
        config: リトライ設定
        exceptions: リトライ対象の例外クラス
        on_retry: リトライ時に実行するコールバック
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # 最後の試行で失敗
                        logger.error(f"関数 {func.__name__} - 全リトライ失敗: {e}")
                        raise e

                    # リトライ待機
                    delay = min(
                        config.base_delay * (config.backoff_factor**attempt),
                        config.max_delay,
                    )

                    if config.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5  # ±25%のジッター

                    logger.warning(
                        f"関数 {func.__name__} - リトライ {attempt + 1}/{config.max_attempts} "
                        f"({delay:.2f}秒後): {e}"
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    time.sleep(delay)

            # 到達しないはずだが、安全のため
            raise last_exception

        return wrapper

    return decorator


def with_fallback(fallback_func: Callable[..., T], exceptions: tuple = (Exception,)):
    """
    フォールバック処理デコレータ

    Args:
        fallback_func: フォールバック関数
        exceptions: フォールバック対象の例外クラス
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"関数 {func.__name__} 失敗、フォールバック実行: {e}")
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"フォールバック関数も失敗: {fallback_error}")
                    raise e  # 元の例外を再発生

        return wrapper

    return decorator


def with_timeout(timeout_seconds: float):
    """
    タイムアウト処理デコレータ

    Args:
        timeout_seconds: タイムアウト時間（秒）
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"関数 {func.__name__} がタイムアウト（{timeout_seconds}秒）"
                )

            # Windowsでは利用できない場合がある
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))

                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # タイマー解除
                    return result
                finally:
                    signal.signal(signal.SIGALRM, old_handler)

            except AttributeError:
                # Windows環境等でSIGALRMが利用できない場合
                logger.warning("タイムアウト処理が利用できない環境です")
                return func(*args, **kwargs)

        return wrapper

    return decorator


class HealthChecker:
    """ヘルスチェッククラス"""

    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_check_results: Dict[str, bool] = {}
        self.last_check_times: Dict[str, float] = {}

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """ヘルスチェック関数登録"""
        self.health_checks[name] = check_func
        logger.info(f"ヘルスチェック登録: {name}")

    def run_health_checks(self) -> Dict[str, bool]:
        """全ヘルスチェック実行"""
        results = {}
        current_time = time.time()

        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.last_check_results[name] = result
                self.last_check_times[name] = current_time

                status = "OK" if result else "NG"
                logger.info(f"ヘルスチェック {name}: {status}")

            except Exception as e:
                results[name] = False
                self.last_check_results[name] = False
                self.last_check_times[name] = current_time
                logger.error(f"ヘルスチェック {name} 実行エラー: {e}")

        return results

    def get_system_health(self) -> Dict[str, Any]:
        """システム全体の健全性取得"""
        check_results = self.run_health_checks()

        total_checks = len(check_results)
        passed_checks = sum(check_results.values())
        health_ratio = passed_checks / max(1, total_checks)

        return {
            "overall_health": health_ratio >= 0.8,  # 80%以上で健全
            "health_ratio": health_ratio,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "individual_results": check_results,
            "last_check_time": time.time(),
        }


# グローバルヘルスチェッカー
_global_health_checker = None


def get_health_checker() -> HealthChecker:
    """グローバルヘルスチェッカー取得"""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


# 実用的なデフォルト設定
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3, base_delay=1.0, backoff_factor=2.0, max_delay=30.0, jitter=True
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=5, base_delay=0.5, backoff_factor=1.5, max_delay=10.0, jitter=True
)


if __name__ == "__main__":
    # テスト実行
    print("=== フォールトトレランステスト ===")

    # リトライテスト
    @with_retry(DEFAULT_RETRY_CONFIG, exceptions=(ValueError,))
    def flaky_function(success_rate: float = 0.3):
        import random

        if random.random() < success_rate:
            return "成功"
        else:
            raise ValueError("ランダム失敗")

    # サーキットブレーカーテスト
    circuit_breaker = CircuitBreaker(
        "test_cb", CircuitBreakerConfig(failure_threshold=2)
    )

    @circuit_breaker
    def circuit_test_function(should_fail: bool = False):
        if should_fail:
            raise Exception("意図的な失敗")
        return "成功"

    # ヘルスチェックテスト
    health_checker = get_health_checker()

    def dummy_health_check():
        return True

    health_checker.register_health_check("dummy_check", dummy_health_check)

    # 実行
    try:
        result = flaky_function(0.8)
        print(f"リトライテスト結果: {result}")
    except Exception as e:
        print(f"リトライテスト失敗: {e}")

    # ヘルスチェック実行
    health_status = health_checker.get_system_health()
    print(f"システムヘルス: {health_status}")

    print("フォールトトレランステスト完了")
