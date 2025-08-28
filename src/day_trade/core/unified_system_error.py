"""
統一システムエラーハンドリング

全アーキテクチャコンポーネント共通のエラーハンドリングシステム。
重複したエラークラスを統合し、一貫性のあるエラー処理を提供。
"""

import asyncio
import json
import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Type
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


# ================================
# 統一エラー分類システム
# ================================

class ErrorSeverity(Enum):
    """エラー重要度"""
    INFO = "info"
    LOW = "low"
    WARNING = "warning"
    MEDIUM = "medium"
    HIGH = "high"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    DATA_ACCESS = "data_access"


class ErrorRecoveryAction(Enum):
    """エラー回復アクション"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"
    ALERT = "alert"


# ================================
# 統一エラークラス階層
# ================================

@dataclass
class ErrorContext:
    """エラーコンテキスト"""
    operation: str = ""
    user_id: str = ""
    session_id: str = ""
    request_id: str = ""
    component: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'component': self.component,
            'additional_data': self.additional_data
        }


class UnifiedSystemError(Exception):
    """統一システムエラー基底クラス"""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.BUSINESS_LOGIC,
        recovery_action: ErrorRecoveryAction = ErrorRecoveryAction.LOG_ONLY,
        context: ErrorContext = None,
        cause: Exception = None,
        details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.recovery_action = recovery_action
        self.context = context or ErrorContext()
        self.cause = cause
        self.details = details or {}
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()
        
    def _generate_error_code(self) -> str:
        """エラーコード生成"""
        class_name = self.__class__.__name__
        timestamp = int(self.timestamp.timestamp())
        return f"{class_name}_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'recovery_action': self.recovery_action.value,
            'context': self.context.to_dict(),
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None,
            'stack_trace': self.stack_trace
        }
    
    def is_recoverable(self) -> bool:
        """回復可能かどうか"""
        return self.recovery_action in [
            ErrorRecoveryAction.RETRY,
            ErrorRecoveryAction.FALLBACK
        ]
    
    def should_alert(self) -> bool:
        """アラート対象かどうか"""
        return self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] or \
               self.recovery_action == ErrorRecoveryAction.ALERT


# ================================
# 具体的なエラークラス
# ================================

class ValidationError(UnifiedSystemError):
    """検証エラー"""
    
    def __init__(self, message: str, field_name: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            recovery_action=ErrorRecoveryAction.LOG_ONLY,
            details={'field_name': field_name},
            **kwargs
        )


class BusinessLogicError(UnifiedSystemError):
    """ビジネスロジックエラー"""
    
    def __init__(self, message: str, rule_name: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            recovery_action=ErrorRecoveryAction.LOG_ONLY,
            details={'rule_name': rule_name},
            **kwargs
        )


class InfrastructureError(UnifiedSystemError):
    """インフラストラクチャエラー"""
    
    def __init__(self, message: str, service_name: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INFRASTRUCTURE,
            recovery_action=ErrorRecoveryAction.RETRY,
            details={'service_name': service_name},
            **kwargs
        )


class SecurityError(UnifiedSystemError):
    """セキュリティエラー"""
    
    def __init__(self, message: str, security_context: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            recovery_action=ErrorRecoveryAction.ALERT,
            details={'security_context': security_context},
            **kwargs
        )


class ExternalServiceError(UnifiedSystemError):
    """外部サービスエラー"""
    
    def __init__(self, message: str, service_name: str = "", status_code: int = 0, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            recovery_action=ErrorRecoveryAction.CIRCUIT_BREAK,
            details={'service_name': service_name, 'status_code': status_code},
            **kwargs
        )


class ConfigurationError(UnifiedSystemError):
    """設定エラー"""
    
    def __init__(self, message: str, config_key: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recovery_action=ErrorRecoveryAction.ESCALATE,
            details={'config_key': config_key},
            **kwargs
        )


class DataAccessError(UnifiedSystemError):
    """データアクセスエラー"""
    
    def __init__(self, message: str, table_name: str = "", operation: str = "", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_ACCESS,
            recovery_action=ErrorRecoveryAction.RETRY,
            details={'table_name': table_name, 'operation': operation},
            **kwargs
        )


class PerformanceError(UnifiedSystemError):
    """パフォーマンスエラー"""
    
    def __init__(self, message: str, threshold: float = 0.0, actual_value: float = 0.0, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PERFORMANCE,
            recovery_action=ErrorRecoveryAction.LOG_ONLY,
            details={'threshold': threshold, 'actual_value': actual_value},
            **kwargs
        )


# ================================
# エラー回復戦略
# ================================

class ErrorRecoveryStrategy(ABC):
    """エラー回復戦略基底クラス"""
    
    @abstractmethod
    async def execute(self, error: UnifiedSystemError, context: Dict[str, Any]) -> Any:
        """回復戦略実行"""
        pass


class RetryStrategy(ErrorRecoveryStrategy):
    """リトライ戦略"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
    
    async def execute(self, error: UnifiedSystemError, context: Dict[str, Any]) -> Any:
        """リトライ実行"""
        operation = context.get('operation')
        if not operation:
            raise error
        
        last_exception = error
        current_delay = self.delay
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(current_delay)
                return await operation()
            except Exception as e:
                last_exception = e
                current_delay *= self.backoff_factor
                logging.warning(f"リトライ {attempt + 1}/{self.max_retries} 失敗: {e}")
        
        raise last_exception


class FallbackStrategy(ErrorRecoveryStrategy):
    """フォールバック戦略"""
    
    def __init__(self, fallback_function: Callable):
        self.fallback_function = fallback_function
    
    async def execute(self, error: UnifiedSystemError, context: Dict[str, Any]) -> Any:
        """フォールバック実行"""
        logging.info(f"フォールバック実行: {error.message}")
        
        if asyncio.iscoroutinefunction(self.fallback_function):
            return await self.fallback_function(error, context)
        else:
            return self.fallback_function(error, context)


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """サーキットブレーカー戦略"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    async def execute(self, error: UnifiedSystemError, context: Dict[str, Any]) -> Any:
        """サーキットブレーカー実行"""
        with self._lock:
            if self.state == "OPEN":
                if self._should_try_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise error
            
            operation = context.get('operation')
            if not operation:
                raise error
            
            try:
                result = await operation()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_try_reset(self) -> bool:
        """リセット試行判定"""
        if not self.last_failure_time:
            return True
        return (datetime.now() - self.last_failure_time).seconds > self.timeout
    
    def _on_success(self):
        """成功時処理"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """失敗時処理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# ================================
# 統一エラーハンドラー
# ================================

class UnifiedErrorHandler:
    """統一エラーハンドラー"""
    
    def __init__(self):
        self.strategies: Dict[ErrorRecoveryAction, ErrorRecoveryStrategy] = {}
        self.error_log: List[UnifiedSystemError] = []
        self.max_log_size = 10000
        self._lock = threading.Lock()
        self.custom_handlers: Dict[Type[UnifiedSystemError], Callable] = {}
        
        # デフォルト戦略設定
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """デフォルト戦略設定"""
        self.strategies[ErrorRecoveryAction.RETRY] = RetryStrategy()
        self.strategies[ErrorRecoveryAction.FALLBACK] = FallbackStrategy(self._default_fallback)
        self.strategies[ErrorRecoveryAction.CIRCUIT_BREAK] = CircuitBreakerStrategy()
    
    async def _default_fallback(self, error: UnifiedSystemError, context: Dict[str, Any]) -> Any:
        """デフォルトフォールバック"""
        logging.warning(f"デフォルトフォールバック実行: {error.message}")
        return None
    
    def register_strategy(self, action: ErrorRecoveryAction, strategy: ErrorRecoveryStrategy):
        """回復戦略登録"""
        self.strategies[action] = strategy
        logging.info(f"回復戦略登録: {action.value}")
    
    def register_custom_handler(self, error_type: Type[UnifiedSystemError], handler: Callable):
        """カスタムハンドラー登録"""
        self.custom_handlers[error_type] = handler
        logging.info(f"カスタムハンドラー登録: {error_type.__name__}")
    
    async def handle_error(self, error: Exception, context: ErrorContext = None, operation: Callable = None) -> Any:
        """エラー処理"""
        # UnifiedSystemErrorに変換
        if not isinstance(error, UnifiedSystemError):
            unified_error = UnifiedSystemError(
                message=str(error),
                cause=error,
                context=context or ErrorContext()
            )
        else:
            unified_error = error
            if context:
                unified_error.context = context
        
        # エラーログに追加
        with self._lock:
            self.error_log.append(unified_error)
            if len(self.error_log) > self.max_log_size:
                self.error_log.pop(0)
        
        # ログ出力
        await self._log_error(unified_error)
        
        # アラート判定
        if unified_error.should_alert():
            await self._send_alert(unified_error)
        
        # カスタムハンドラーチェック
        error_type = type(unified_error)
        if error_type in self.custom_handlers:
            try:
                return await self._execute_custom_handler(unified_error)
            except Exception as e:
                logging.error(f"カスタムハンドラーエラー: {e}")
        
        # 回復戦略実行
        recovery_strategy = self.strategies.get(unified_error.recovery_action)
        if recovery_strategy and operation:
            try:
                return await recovery_strategy.execute(unified_error, {'operation': operation})
            except Exception as e:
                logging.error(f"回復戦略実行エラー: {e}")
                raise unified_error
        
        # 回復不可能な場合
        if unified_error.severity == ErrorSeverity.CRITICAL:
            logging.critical(f"致命的エラー: {unified_error.message}")
        
        raise unified_error
    
    async def _log_error(self, error: UnifiedSystemError):
        """エラーログ出力"""
        log_data = {
            'timestamp': error.timestamp.isoformat(),
            'error_code': error.error_code,
            'message': error.message,
            'severity': error.severity.value,
            'category': error.category.value,
            'context': error.context.to_dict()
        }
        
        log_message = f"エラー発生: {json.dumps(log_data, ensure_ascii=False)}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logging.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logging.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logging.warning(log_message)
        else:
            logging.info(log_message)
    
    async def _send_alert(self, error: UnifiedSystemError):
        """アラート送信"""
        # 実際の実装では外部アラートサービスに送信
        alert_data = {
            'error_code': error.error_code,
            'message': error.message,
            'severity': error.severity.value,
            'timestamp': error.timestamp.isoformat(),
            'context': error.context.to_dict()
        }
        
        logging.critical(f"🚨 アラート送信: {json.dumps(alert_data, ensure_ascii=False)}")
    
    async def _execute_custom_handler(self, error: UnifiedSystemError) -> Any:
        """カスタムハンドラー実行"""
        handler = self.custom_handlers[type(error)]
        
        if asyncio.iscoroutinefunction(handler):
            return await handler(error)
        else:
            return handler(error)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得"""
        with self._lock:
            if not self.error_log:
                return {}
            
            total_errors = len(self.error_log)
            severity_counts = {}
            category_counts = {}
            recent_errors = 0
            
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for error in self.error_log:
                # 重要度別カウント
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                
                # カテゴリ別カウント
                category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
                
                # 最近のエラー
                if error.timestamp > cutoff_time:
                    recent_errors += 1
            
            return {
                'total_errors': total_errors,
                'recent_errors_1h': recent_errors,
                'severity_distribution': severity_counts,
                'category_distribution': category_counts,
                'error_rate_per_hour': recent_errors,
                'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
            }


# ================================
# エラーハンドリングデコレーター
# ================================

def error_handler(
    error_context: ErrorContext = None,
    fallback_value: Any = None,
    max_retries: int = 0
):
    """エラーハンドリングデコレーター"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = UnifiedErrorHandler()
            context = error_context or ErrorContext(operation=func.__name__)
            
            async def operation():
                return await func(*args, **kwargs)
            
            try:
                return await handler.handle_error(None, context, operation)
            except Exception as e:
                if fallback_value is not None:
                    logging.warning(f"フォールバック値を返却: {fallback_value}")
                    return fallback_value
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同期関数用の簡単な実装
            try:
                return func(*args, **kwargs)
            except Exception as e:
                unified_error = UnifiedSystemError(str(e), cause=e)
                logging.error(f"エラー発生: {unified_error.to_dict()}")
                
                if fallback_value is not None:
                    return fallback_value
                raise unified_error
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ================================
# グローバルエラーハンドラー
# ================================

_global_error_handler = UnifiedErrorHandler()

def get_global_error_handler() -> UnifiedErrorHandler:
    """グローバルエラーハンドラー取得"""
    return _global_error_handler

async def handle_global_error(error: Exception, context: ErrorContext = None) -> Any:
    """グローバルエラー処理"""
    return await _global_error_handler.handle_error(error, context)


# エクスポート
__all__ = [
    # エラー分類
    'ErrorSeverity', 'ErrorCategory', 'ErrorRecoveryAction', 'ErrorContext',
    
    # エラークラス
    'UnifiedSystemError', 'ValidationError', 'BusinessLogicError', 'InfrastructureError',
    'SecurityError', 'ExternalServiceError', 'ConfigurationError', 'DataAccessError', 'PerformanceError',
    
    # 回復戦略
    'ErrorRecoveryStrategy', 'RetryStrategy', 'FallbackStrategy', 'CircuitBreakerStrategy',
    
    # エラーハンドラー
    'UnifiedErrorHandler', 'error_handler',
    
    # グローバル関数
    'get_global_error_handler', 'handle_global_error'
]