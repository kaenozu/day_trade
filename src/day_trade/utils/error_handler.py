#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合エラーハンドリングシステム

システム全体のエラー処理を一元化し、適切な回復処理を行います。
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Dict, Type, Union
from dataclasses import dataclass
from datetime import datetime
import threading
from enum import Enum


class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """エラー情報"""
    timestamp: datetime
    error_type: str
    message: str
    traceback_str: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    retry_count: int = 0
    resolved: bool = False


class RetryPolicy:
    """リトライポリシー"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def get_delay(self, retry_count: int) -> float:
        """リトライ遅延時間を計算"""
        delay = self.base_delay * (self.backoff_factor ** retry_count)
        return min(delay, self.max_delay)


class ErrorHandler:
    """
    統合エラーハンドリングクラス
    
    エラーの記録、分析、回復処理を一元管理
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: Dict[str, ErrorInfo] = {}
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
        
        # エラー別のリトライポリシー
        self.retry_policies: Dict[Type[Exception], RetryPolicy] = {
            ConnectionError: RetryPolicy(max_retries=5, base_delay=2.0),
            TimeoutError: RetryPolicy(max_retries=3, base_delay=1.0),
            FileNotFoundError: RetryPolicy(max_retries=2, base_delay=0.5),
            PermissionError: RetryPolicy(max_retries=1, base_delay=1.0),
            MemoryError: RetryPolicy(max_retries=1, base_delay=5.0),
        }
        
        self.logger.info("統合エラーハンドリングシステム初期化完了")
    
    def register_error(self, 
                      exception: Exception,
                      context: Optional[Dict[str, Any]] = None,
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> str:
        """エラーを登録"""
        error_id = f"{type(exception).__name__}_{int(time.time())}"
        
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            message=str(exception),
            traceback_str=traceback.format_exc(),
            severity=severity,
            context=context or {},
        )
        
        with self.lock:
            self.error_history[error_id] = error_info
            self.error_counts[type(exception).__name__] = (
                self.error_counts.get(type(exception).__name__, 0) + 1
            )
        
        # ログ出力
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[severity]
        
        self.logger.log(
            log_level,
            f"エラー登録 [{error_id}] {type(exception).__name__}: {exception}",
            extra={"error_id": error_id, "context": context}
        )
        
        return error_id
    
    def get_retry_policy(self, exception: Exception) -> RetryPolicy:
        """例外に対応するリトライポリシーを取得"""
        for exc_type, policy in self.retry_policies.items():
            if isinstance(exception, exc_type):
                return policy
        
        # デフォルトポリシー
        return RetryPolicy()
    
    def should_retry(self, exception: Exception, retry_count: int) -> bool:
        """リトライすべきかを判定"""
        policy = self.get_retry_policy(exception)
        return retry_count < policy.max_retries
    
    def get_error_stats(self) -> Dict[str, Any]:
        """エラー統計を取得"""
        with self.lock:
            total_errors = len(self.error_history)
            recent_errors = sum(
                1 for error in self.error_history.values()
                if (datetime.now() - error.timestamp).total_seconds() < 3600
            )
            
            severity_counts = {}
            for error in self.error_history.values():
                severity_counts[error.severity.value] = (
                    severity_counts.get(error.severity.value, 0) + 1
                )
            
            return {
                "total_errors": total_errors,
                "recent_errors_1h": recent_errors,
                "error_counts_by_type": dict(self.error_counts),
                "severity_counts": severity_counts,
                "most_common_errors": sorted(
                    self.error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    def clear_old_errors(self, max_age_hours: int = 24):
        """古いエラー情報をクリア"""
        cutoff_time = datetime.now()
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - max_age_hours)
        
        with self.lock:
            old_ids = [
                error_id for error_id, error_info in self.error_history.items()
                if error_info.timestamp < cutoff_time
            ]
            
            for error_id in old_ids:
                del self.error_history[error_id]
        
        if old_ids:
            self.logger.info(f"古いエラー情報を削除: {len(old_ids)}件")


# グローバルエラーハンドラー
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """グローバルエラーハンドラーを取得"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    retry: bool = True,
    fallback_value: Any = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    エラーハンドリングデコレータ
    
    Args:
        severity: エラー重要度
        retry: リトライするかどうか
        fallback_value: エラー時の戻り値
        context: エラーコンテキスト
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            retry_count = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    # エラー登録
                    error_context = dict(context or {})
                    error_context.update({
                        "function": func.__name__,
                        "args": str(args)[:200],  # 長すぎる場合は切り詰め
                        "kwargs": str(kwargs)[:200],
                        "retry_count": retry_count
                    })
                    
                    error_id = handler.register_error(e, error_context, severity)
                    
                    # リトライ判定
                    if retry and handler.should_retry(e, retry_count):
                        retry_count += 1
                        policy = handler.get_retry_policy(e)
                        delay = policy.get_delay(retry_count - 1)
                        
                        handler.logger.info(
                            f"リトライ実行 [{error_id}] {retry_count}/{policy.max_retries} "
                            f"(遅延: {delay:.1f}秒)"
                        )
                        
                        time.sleep(delay)
                        continue
                    
                    # リトライ上限または非対象の場合
                    handler.logger.error(
                        f"エラー処理完了 [{error_id}] 戻り値: {fallback_value}"
                    )
                    
                    if fallback_value is not None:
                        return fallback_value
                    
                    # 元の例外を再発生
                    raise
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_value: Any = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    安全な関数実行
    
    Args:
        func: 実行する関数
        default_value: エラー時のデフォルト値
        context: エラーコンテキスト
        
    Returns:
        関数の戻り値またはデフォルト値
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = get_error_handler()
        error_context = dict(context or {})
        error_context.update({
            "function": func.__name__ if hasattr(func, '__name__') else str(func),
            "safe_execute": True
        })
        
        handler.register_error(e, error_context, ErrorSeverity.LOW)
        return default_value


class CircuitBreaker:
    """
    サーキットブレーカーパターン実装
    
    連続したエラーが発生した場合、一定時間処理を停止
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if self._should_attempt_reset():
                        self.state = "HALF_OPEN"
                        self.logger.info("サーキットブレーカー: HALF_OPEN状態に移行")
                    else:
                        raise Exception("サーキットブレーカーが開いています")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                
                except self.expected_exception as e:
                    self._on_failure()
                    raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """リセットを試行すべきかを判定"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """成功時の処理"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("サーキットブレーカー: CLOSED状態に復帰")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """失敗時の処理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"サーキットブレーカー: OPEN状態に移行 "
                f"(失敗回数: {self.failure_count})"
            )


# よく使用されるデコレータのエイリアス
retry_on_error = lambda **kwargs: handle_errors(retry=True, **kwargs)
log_errors = lambda **kwargs: handle_errors(retry=False, **kwargs)
critical_section = lambda **kwargs: handle_errors(
    severity=ErrorSeverity.CRITICAL, **kwargs
)