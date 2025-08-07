"""
統合エラーハンドリングシステム

Phase 4: エラーハンドリング統合とデータベース層統合の一環として、
分散していたエラーハンドリングパターンを統一し、保守性を向上させます。

主な機能:
- 全モジュール共通のエラーハンドリングパターン統一
- データベース、API、分析エラーの一元管理
- エラー分類とリカバリ戦略の自動適用
- 構造化ログ出力とメトリクス収集
- ユーザーフレンドリーなエラーメッセージ生成
"""

import functools
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..config.unified_config import get_unified_config_manager
from ..utils.exceptions import (
    AnalysisError,
    APIError,
    DatabaseError,
    NetworkError,
    TradingError,
    ValidationError,
    handle_database_exception,
    handle_network_exception,
)
from ..utils.logging_config import get_context_logger
from ..utils.unified_utils import (
    CircuitBreaker,
    ThreadSafeCounter,
    retry_on_failure,
)

logger = get_context_logger(__name__, component="unified_error_handler")

T = TypeVar("T")


class ErrorSeverity(Enum):
    """エラー重要度レベル"""

    CRITICAL = "critical"  # システム全体に影響
    HIGH = "high"  # 機能停止
    MEDIUM = "medium"  # 一部機能の問題
    LOW = "low"  # 軽微な問題
    INFO = "info"  # 情報レベル


class ErrorCategory(Enum):
    """エラーカテゴリー"""

    DATABASE = "database"
    API = "api"
    NETWORK = "network"
    ANALYSIS = "analysis"
    TRADING = "trading"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    USER_INPUT = "user_input"


class RecoveryAction(Enum):
    """リカバリアクション"""

    RETRY = "retry"  # リトライ実行
    FALLBACK = "fallback"  # フォールバック処理
    CIRCUIT_BREAK = "circuit_break"  # サーキットブレーカー
    USER_INPUT = "user_input"  # ユーザー入力要求
    RESTART = "restart"  # システム再起動
    IGNORE = "ignore"  # 無視
    ESCALATE = "escalate"  # エスカレーション


class UnifiedErrorContext:
    """統合エラーコンテキスト情報"""

    def __init__(
        self,
        error: Exception,
        operation: str,
        component: str,
        user_data: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        recovery_hints: Optional[Dict[str, Any]] = None,
    ):
        self.error = error
        self.operation = operation
        self.component = component
        self.user_data = user_data or {}
        self.system_state = system_state or {}
        self.recovery_hints = recovery_hints or {}
        self.timestamp = time.time()
        self.thread_id = threading.get_ident()

        # エラー分類の自動判定
        self.severity = self._classify_severity()
        self.category = self._classify_category()
        self.suggested_actions = self._suggest_recovery_actions()

    def _classify_severity(self) -> ErrorSeverity:
        """エラー重要度を分類"""
        if isinstance(self.error, (DatabaseError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(self.error, (APIError, TradingError)):
            return ErrorSeverity.HIGH
        elif isinstance(self.error, (AnalysisError, NetworkError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(self.error, ValidationError):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM

    def _classify_category(self) -> ErrorCategory:
        """エラーカテゴリーを分類"""
        if isinstance(self.error, DatabaseError):
            return ErrorCategory.DATABASE
        elif isinstance(self.error, APIError):
            return ErrorCategory.API
        elif isinstance(self.error, NetworkError):
            return ErrorCategory.NETWORK
        elif isinstance(self.error, AnalysisError):
            return ErrorCategory.ANALYSIS
        elif isinstance(self.error, TradingError):
            return ErrorCategory.TRADING
        elif isinstance(self.error, ValidationError):
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.SYSTEM

    def _suggest_recovery_actions(self) -> List[RecoveryAction]:
        """推奨リカバリアクションを提案"""
        actions = []

        if self.category == ErrorCategory.NETWORK:
            actions.extend([RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK])
        elif self.category == ErrorCategory.DATABASE:
            actions.extend([RecoveryAction.RETRY, RecoveryAction.FALLBACK])
        elif self.category == ErrorCategory.API:
            actions.extend([RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK])
        elif self.category == ErrorCategory.VALIDATION:
            actions.append(RecoveryAction.USER_INPUT)
        elif self.severity == ErrorSeverity.CRITICAL:
            actions.append(RecoveryAction.ESCALATE)
        else:
            actions.append(RecoveryAction.RETRY)

        return actions

    def to_dict(self) -> Dict[str, Any]:
        """コンテキスト情報を辞書形式で取得"""
        return {
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "operation": self.operation,
            "component": self.component,
            "severity": self.severity.value,
            "category": self.category.value,
            "suggested_actions": [action.value for action in self.suggested_actions],
            "user_data": self.user_data,
            "system_state": self.system_state,
            "recovery_hints": self.recovery_hints,
        }


class ErrorMetrics:
    """エラーメトリクス収集"""

    def __init__(self):
        self._error_counts = ThreadSafeCounter()
        self._recovery_attempts = ThreadSafeCounter()
        self._successful_recoveries = ThreadSafeCounter()
        self._category_counts = {
            category: ThreadSafeCounter() for category in ErrorCategory
        }
        self._severity_counts = {
            severity: ThreadSafeCounter() for severity in ErrorSeverity
        }
        self._lock = threading.RLock()

    def record_error(self, context: UnifiedErrorContext) -> None:
        """エラー発生を記録"""
        self._error_counts.increment()
        self._category_counts[context.category].increment()
        self._severity_counts[context.severity].increment()

    def record_recovery_attempt(self, action: RecoveryAction) -> None:
        """リカバリ試行を記録"""
        self._recovery_attempts.increment()

    def record_successful_recovery(self, action: RecoveryAction) -> None:
        """リカバリ成功を記録"""
        self._successful_recoveries.increment()

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            total_errors = self._error_counts.get()
            total_recoveries = self._recovery_attempts.get()
            successful_recoveries = self._successful_recoveries.get()

            category_stats = {
                cat.value: counter.get()
                for cat, counter in self._category_counts.items()
            }

            severity_stats = {
                sev.value: counter.get()
                for sev, counter in self._severity_counts.items()
            }

            return {
                "total_errors": total_errors,
                "total_recovery_attempts": total_recoveries,
                "successful_recoveries": successful_recoveries,
                "recovery_success_rate": (
                    successful_recoveries / max(total_recoveries, 1)
                ),
                "category_distribution": category_stats,
                "severity_distribution": severity_stats,
            }


class UnifiedErrorHandler:
    """統合エラーハンドリングシステム"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: 統合設定マネージャー
        """
        self.config_manager = config_manager or get_unified_config_manager()
        self.metrics = ErrorMetrics()
        self._circuit_breakers = {}
        self._lock = threading.RLock()

        # 設定読み込み
        self._load_config()

    def _load_config(self):
        """設定を読み込み"""
        try:
            perf_config = self.config_manager.get_performance_config()
            self.max_retry_attempts = perf_config.max_retry_attempts
            self.circuit_breaker_threshold = getattr(
                perf_config, "circuit_breaker_threshold", 5
            )
            self.circuit_breaker_timeout = getattr(
                perf_config, "circuit_breaker_timeout", 60.0
            )
        except Exception as e:
            logger.warning(f"設定読み込みに失敗、デフォルト値を使用: {e}")
            self.max_retry_attempts = 3
            self.circuit_breaker_threshold = 5
            self.circuit_breaker_timeout = 60.0

    def _get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """コンポーネント用サーキットブレーカーを取得"""
        with self._lock:
            if component not in self._circuit_breakers:
                self._circuit_breakers[component] = CircuitBreaker(
                    failure_threshold=self.circuit_breaker_threshold,
                    recovery_timeout=self.circuit_breaker_timeout,
                )
            return self._circuit_breakers[component]

    def handle_error(
        self,
        error: Exception,
        operation: str,
        component: str,
        user_data: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        recovery_hints: Optional[Dict[str, Any]] = None,
    ) -> UnifiedErrorContext:
        """
        統合エラーハンドリング

        Args:
            error: 発生したエラー
            operation: 実行中の操作
            component: エラーが発生したコンポーネント
            user_data: ユーザー関連データ
            system_state: システム状態情報
            recovery_hints: リカバリヒント

        Returns:
            統合エラーコンテキスト
        """
        # エラーコンテキストを作成
        context = UnifiedErrorContext(
            error=error,
            operation=operation,
            component=component,
            user_data=user_data,
            system_state=system_state,
            recovery_hints=recovery_hints,
        )

        # メトリクス記録
        self.metrics.record_error(context)

        # 構造化ログ出力
        self._log_error_with_context(context)

        return context

    def _log_error_with_context(self, context: UnifiedErrorContext) -> None:
        """構造化ログでエラー情報を出力"""
        log_data = {
            "error_context": context.to_dict(),
            "component": context.component,
            "operation": context.operation,
            "severity": context.severity.value,
            "category": context.category.value,
        }

        if context.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
            logger.error(
                f"{context.severity.value.upper()} error in {context.component}",
                extra=log_data,
                exc_info=True,
            )
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Warning in {context.component}", extra={log_data})
        else:
            logger.info(f"Info: {context.component}", extra={log_data})

    def attempt_recovery(
        self,
        context: UnifiedErrorContext,
        recovery_function: Callable[..., T],
        *args,
        **kwargs,
    ) -> Optional[T]:
        """
        自動リカバリ試行

        Args:
            context: エラーコンテキスト
            recovery_function: リカバリ実行関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            リカバリ結果、またはNone（失敗時）
        """
        for action in context.suggested_actions:
            self.metrics.record_recovery_attempt(action)

            try:
                result = self._execute_recovery_action(
                    action, context, recovery_function, *args, **kwargs
                )

                if result is not None:
                    self.metrics.record_successful_recovery(action)
                    logger.info(
                        f"Recovery successful using {action.value}",
                        extra={
                            "component": context.component,
                            "operation": context.operation,
                            "recovery_action": action.value,
                        },
                    )
                    return result

            except Exception as recovery_error:
                logger.warning(
                    f"Recovery action {action.value} failed: {recovery_error}",
                    extra={
                        "component": context.component,
                        "original_error": str(context.error),
                        "recovery_error": str(recovery_error),
                    },
                )

        return None

    def _execute_recovery_action(
        self,
        action: RecoveryAction,
        context: UnifiedErrorContext,
        recovery_function: Callable[..., T],
        *args,
        **kwargs,
    ) -> Optional[T]:
        """個別リカバリアクションの実行"""

        if action == RecoveryAction.RETRY:
            return self._execute_retry(context, recovery_function, *args, **kwargs)

        elif action == RecoveryAction.CIRCUIT_BREAK:
            circuit_breaker = self._get_circuit_breaker(context.component)
            return circuit_breaker.call(recovery_function, *args, **kwargs)

        elif action == RecoveryAction.FALLBACK:
            return self._execute_fallback(context, recovery_function, *args, **kwargs)

        else:
            # その他のアクションは現在未実装
            return None

    def _execute_retry(
        self,
        context: UnifiedErrorContext,
        recovery_function: Callable[..., T],
        *args,
        **kwargs,
    ) -> Optional[T]:
        """リトライ実行"""

        @retry_on_failure(
            max_retries=self.max_retry_attempts,
            delay=1.0,
            backoff_factor=2.0,
            exceptions=(type(context.error),),
        )
        def retry_wrapper():
            return recovery_function(*args, **kwargs)

        try:
            return retry_wrapper()
        except Exception:
            return None

    def _execute_fallback(
        self,
        context: UnifiedErrorContext,
        recovery_function: Callable[..., T],
        *args,
        **kwargs,
    ) -> Optional[T]:
        """フォールバック実行"""
        fallback_hint = context.recovery_hints.get("fallback_function")
        if fallback_hint and callable(fallback_hint):
            try:
                return fallback_hint(*args, **kwargs)
            except Exception as fallback_error:
                logger.debug(f"Fallback failed: {fallback_error}")

        return None


# デコレーター関数


def unified_error_handling(
    operation: str,
    component: str,
    auto_recovery: bool = True,
    fallback_function: Optional[Callable] = None,
):
    """
    統合エラーハンドリングデコレーター

    Args:
        operation: 操作名
        component: コンポーネント名
        auto_recovery: 自動リカバリを試行するか
        fallback_function: フォールバック関数
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = get_unified_error_handler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # エラーコンテキストを作成
                recovery_hints = {}
                if fallback_function:
                    recovery_hints["fallback_function"] = fallback_function

                context = handler.handle_error(
                    error=e,
                    operation=operation,
                    component=component,
                    recovery_hints=recovery_hints,
                )

                # 自動リカバリ試行
                if auto_recovery:
                    recovery_result = handler.attempt_recovery(
                        context, func, *args, **kwargs
                    )
                    if recovery_result is not None:
                        return recovery_result

                # リカバリ失敗時は元のエラーを再発生
                raise e

        return wrapper

    return decorator


# データベース専用エラーハンドリング


@contextmanager
def unified_database_session(
    session_factory,
    operation: str = "database_operation",
    auto_commit: bool = True,
):
    """
    統合データベースセッション管理

    Args:
        session_factory: SQLAlchemyセッションファクトリ
        operation: 操作名
        auto_commit: 自動コミットを行うか
    """
    handler = get_unified_error_handler()
    session = session_factory()

    try:
        yield session

        if auto_commit:
            session.commit()

    except Exception as e:
        session.rollback()

        # SQLAlchemy例外を統一例外に変換
        if hasattr(e, "__module__") and "sqlalchemy" in str(e.__module__):
            converted_error = handle_database_exception(e)
        else:
            converted_error = e

        # エラーハンドリング
        handler.handle_error(
            error=converted_error,
            operation=operation,
            component="database",
            system_state={"session_info": str(session.info)},
        )

        # データベースエラーは通常リカバリが困難なので、そのまま再発生
        raise converted_error from e

    finally:
        session.close()


# グローバルインスタンス管理

_global_error_handler: Optional[UnifiedErrorHandler] = None
_handler_lock = threading.RLock()


def get_unified_error_handler() -> UnifiedErrorHandler:
    """統合エラーハンドラーのグローバルインスタンスを取得"""
    global _global_error_handler

    if _global_error_handler is None:
        with _handler_lock:
            if _global_error_handler is None:
                _global_error_handler = UnifiedErrorHandler()

    return _global_error_handler


def set_unified_error_handler(handler: UnifiedErrorHandler) -> None:
    """統合エラーハンドラーのグローバルインスタンスを設定"""
    global _global_error_handler

    with _handler_lock:
        _global_error_handler = handler


# 便利関数


def handle_api_error(
    error: Exception,
    api_name: str,
    endpoint: str = "",
    user_data: Optional[Dict] = None,
) -> UnifiedErrorContext:
    """API エラー専用ハンドリング"""
    handler = get_unified_error_handler()

    # ネットワーク例外の変換
    if hasattr(error, "__module__") and "requests" in str(error.__module__):
        converted_error = handle_network_exception(error)
    else:
        converted_error = error

    return handler.handle_error(
        error=converted_error,
        operation=f"api_call_{endpoint}" if endpoint else "api_call",
        component=f"api_{api_name}",
        user_data=user_data,
        system_state={"endpoint": endpoint},
    )


def handle_analysis_error(
    error: Exception,
    analysis_type: str,
    input_data: Optional[Dict] = None,
) -> UnifiedErrorContext:
    """分析エラー専用ハンドリング"""
    handler = get_unified_error_handler()

    return handler.handle_error(
        error=error,
        operation=f"analysis_{analysis_type}",
        component="analysis_engine",
        user_data=input_data,
    )


def get_error_handler_stats() -> Dict[str, Any]:
    """エラーハンドラー統計情報を取得"""
    handler = get_unified_error_handler()
    return handler.metrics.get_stats()
