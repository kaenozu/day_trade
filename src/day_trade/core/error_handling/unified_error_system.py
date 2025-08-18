"""
統一エラーハンドリングシステム

システム全体で一貫したエラー処理を提供
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback
import asyncio
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATA_ACCESS = "data_access"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"


class RecoveryAction(Enum):
    """回復アクション"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """エラーコンテキスト"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    component_name: str = ""
    operation_name: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class ErrorDetails:
    """エラー詳細情報"""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    stacktrace: Optional[str] = None
    inner_exception: Optional[Exception] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ApplicationError(Exception):
    """アプリケーション基底例外"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        inner_exception: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None,
        **metadata
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.inner_exception = inner_exception
        self.recovery_suggestions = recovery_suggestions or []
        self.metadata = metadata
        
        # スタックトレース保存
        self.stacktrace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "error_id": self.context.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "component_name": self.context.component_name,
            "operation_name": self.context.operation_name,
            "user_id": self.context.user_id,
            "correlation_id": self.context.correlation_id,
            "recovery_suggestions": self.recovery_suggestions,
            "metadata": self.metadata
        }


class ValidationError(ApplicationError):
    """バリデーションエラー"""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.VALIDATION,
            ErrorSeverity.LOW,
            **kwargs
        )
        self.field = field


class BusinessLogicError(ApplicationError):
    """ビジネスロジックエラー"""
    
    def __init__(self, message: str, rule_name: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.BUSINESS_LOGIC,
            ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.rule_name = rule_name


class BusinessRuleViolationError(BusinessLogicError):
    """ビジネスルール違反エラー"""
    
    def __init__(self, message: str, rule_name: str = None, **kwargs):
        super().__init__(message, rule_name, **kwargs)


class InsufficientDataError(ApplicationError):
    """データ不足エラー"""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.DATA_ACCESS,
            ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.data_type = data_type


class DataAccessError(ApplicationError):
    """データアクセスエラー"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.DATA_ACCESS,
            ErrorSeverity.HIGH,
            **kwargs
        )
        self.operation = operation


class ExternalServiceError(ApplicationError):
    """外部サービスエラー"""
    
    def __init__(self, message: str, service_name: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorSeverity.HIGH,
            **kwargs
        )
        self.service_name = service_name


class SystemError(ApplicationError):
    """システムエラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.SYSTEM,
            ErrorSeverity.CRITICAL,
            **kwargs
        )


class ErrorRecoveryStrategy(ABC):
    """エラー回復戦略"""
    
    @abstractmethod
    def can_handle(self, error: ApplicationError) -> bool:
        """処理可能判定"""
        pass
    
    @abstractmethod
    async def recover(self, error: ApplicationError, operation: Callable) -> Any:
        """回復処理"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """戦略名"""
        pass


class RetryStrategy(ErrorRecoveryStrategy):
    """リトライ戦略"""
    
    def __init__(self, max_attempts: int = 3, delay_seconds: float = 1.0):
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
    
    def can_handle(self, error: ApplicationError) -> bool:
        """一時的なエラーのみリトライ"""
        return error.category in [
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.DATA_ACCESS
        ] and error.severity != ErrorSeverity.CRITICAL
    
    async def recover(self, error: ApplicationError, operation: Callable) -> Any:
        """リトライ実行"""
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    raise e
                await asyncio.sleep(self.delay_seconds * (2 ** attempt))
    
    @property
    def strategy_name(self) -> str:
        return "retry"


class FallbackStrategy(ErrorRecoveryStrategy):
    """フォールバック戦略"""
    
    def __init__(self, fallback_value: Any = None):
        self.fallback_value = fallback_value
    
    def can_handle(self, error: ApplicationError) -> bool:
        """外部サービスエラーをフォールバック"""
        return error.category == ErrorCategory.EXTERNAL_SERVICE
    
    async def recover(self, error: ApplicationError, operation: Callable) -> Any:
        """フォールバック値返却"""
        logger.warning(f"Using fallback for error: {error.message}")
        return self.fallback_value
    
    @property
    def strategy_name(self) -> str:
        return "fallback"


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """サーキットブレーカー戦略"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._is_open = False
    
    def can_handle(self, error: ApplicationError) -> bool:
        """外部サービスエラーを対象"""
        return error.category == ErrorCategory.EXTERNAL_SERVICE
    
    async def recover(self, error: ApplicationError, operation: Callable) -> Any:
        """サーキットブレーカー処理"""
        current_time = datetime.now()
        
        # サーキット開放中
        if self._is_open:
            if (current_time - self._last_failure_time).total_seconds() > self.timeout_seconds:
                # タイムアウト後、半開状態で試行
                self._is_open = False
                self._failure_count = 0
            else:
                raise ApplicationError(
                    "Circuit breaker is open",
                    ErrorCategory.SYSTEM,
                    ErrorSeverity.HIGH
                )
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            
            # 成功時リセット
            self._failure_count = 0
            return result
            
        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = current_time
            
            if self._failure_count >= self.failure_threshold:
                self._is_open = True
                logger.error(f"Circuit breaker opened after {self._failure_count} failures")
            
            raise e
    
    @property
    def strategy_name(self) -> str:
        return "circuit_breaker"


class ErrorReporter:
    """エラー報告"""
    
    def __init__(self):
        self._handlers: List[Callable[[ApplicationError], None]] = []
    
    def add_handler(self, handler: Callable[[ApplicationError], None]) -> None:
        """ハンドラー追加"""
        self._handlers.append(handler)
    
    async def report(self, error: ApplicationError) -> None:
        """エラー報告"""
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error)
                else:
                    handler(error)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")


class ErrorAnalytics:
    """エラー分析"""
    
    def __init__(self):
        self._error_history: List[ApplicationError] = []
        self._error_patterns: Dict[str, int] = {}
    
    def record_error(self, error: ApplicationError) -> None:
        """エラー記録"""
        self._error_history.append(error)
        
        # パターン記録
        pattern = f"{error.category.value}:{error.message}"
        self._error_patterns[pattern] = self._error_patterns.get(pattern, 0) + 1
    
    def get_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """エラー傾向分析"""
        cutoff_time = datetime.now().replace(microsecond=0) - timedelta(hours=hours)
        recent_errors = [e for e in self._error_history if e.context.timestamp > cutoff_time]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "categories": category_counts,
            "severities": severity_counts,
            "most_common_patterns": sorted(
                self._error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class UnifiedErrorHandler:
    """統一エラーハンドラー"""
    
    def __init__(self):
        self._strategies: List[ErrorRecoveryStrategy] = []
        self._reporter = ErrorReporter()
        self._analytics = ErrorAnalytics()
        
        # デフォルト戦略追加
        self.add_strategy(RetryStrategy())
        self.add_strategy(FallbackStrategy())
        self.add_strategy(CircuitBreakerStrategy())
    
    def add_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """回復戦略追加"""
        self._strategies.append(strategy)
    
    def add_error_handler(self, handler: Callable[[ApplicationError], None]) -> None:
        """エラーハンドラー追加"""
        self._reporter.add_handler(handler)
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        operation: Optional[Callable] = None
    ) -> Any:
        """エラー処理"""
        # ApplicationErrorに変換
        if isinstance(error, ApplicationError):
            app_error = error
        else:
            app_error = ApplicationError(
                str(error),
                ErrorCategory.SYSTEM,
                ErrorSeverity.HIGH,
                context=context,
                inner_exception=error
            )
        
        # 分析記録
        self._analytics.record_error(app_error)
        
        # 報告
        await self._reporter.report(app_error)
        
        # 回復試行
        if operation:
            for strategy in self._strategies:
                if strategy.can_handle(app_error):
                    try:
                        return await strategy.recover(app_error, operation)
                    except Exception as recovery_error:
                        logger.warning(f"Recovery strategy {strategy.strategy_name} failed: {recovery_error}")
        
        # 回復不可能
        raise app_error
    
    @contextmanager
    def error_context(
        self,
        component_name: str,
        operation_name: str,
        **context_data
    ):
        """エラーコンテキスト"""
        context = ErrorContext(
            component_name=component_name,
            operation_name=operation_name,
            **context_data
        )
        
        try:
            yield context
        except Exception as e:
            # 同期的なエラー処理のみ実行
            app_error = e if isinstance(e, ApplicationError) else ApplicationError(
                str(e),
                ErrorCategory.SYSTEM,
                ErrorSeverity.HIGH,
                context=context,
                inner_exception=e
            )
            self._analytics.record_error(app_error)
            raise
    
    def get_analytics(self) -> Dict[str, Any]:
        """分析データ取得"""
        return self._analytics.get_error_trends()


# グローバルエラーハンドラー
global_error_handler = UnifiedErrorHandler()


def error_boundary(
    component_name: str = "",
    operation_name: str = "",
    fallback_value: Any = None,
    suppress_errors: bool = False
):
    """エラー境界デコレーター"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component_name=component_name,
                        operation_name=operation_name or func.__name__
                    )
                    
                    if suppress_errors:
                        try:
                            await global_error_handler.handle_error(e, context)
                        except Exception:
                            # エラーハンドリング自体が失敗した場合はログに記録
                            logger.error(f"Error handler failed: {e}")
                        return fallback_value
                    else:
                        try:
                            await global_error_handler.handle_error(e, context, func)
                        except Exception:
                            # エラーハンドリングが失敗した場合は元の例外を再発生
                            raise e
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component_name=component_name,
                        operation_name=operation_name or func.__name__
                    )
                    
                    if suppress_errors:
                        # 同期関数では非同期エラーハンドラを呼び出さず、
                        # 基本的なエラー記録のみ実行
                        app_error = e if isinstance(e, ApplicationError) else ApplicationError(
                            str(e),
                            ErrorCategory.SYSTEM,
                            ErrorSeverity.HIGH,
                            context=context,
                            inner_exception=e
                        )
                        global_error_handler._analytics.record_error(app_error)
                        return fallback_value
                    else:
                        raise e
            return sync_wrapper
    
    return decorator