"""
強化されたエラーハンドリングシステム

カスタム例外、回復機能、ログ統合を提供
"""

import traceback
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps

from ...utils.logging_config import get_context_logger, log_error_with_context

logger = get_context_logger(__name__)


class ErrorSeverity(Enum):
    """エラー重要度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    NETWORK = "network"
    DATABASE = "database"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    CONFIGURATION = "configuration"
    DATA = "data"


@dataclass
class ErrorContext:
    """エラーコンテキスト情報"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_message: Optional[str] = None


class BaseApplicationError(Exception):
    """アプリケーション基底例外"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.user_message = user_message
        self.recovery_hint = recovery_hint
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(self.timestamp.timestamp())}"


class NetworkError(BaseApplicationError):
    """ネットワーク関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            **kwargs
        )


class DatabaseError(BaseApplicationError):
    """データベース関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            **kwargs
        )


class CalculationError(BaseApplicationError):
    """計算関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CALCULATION,
            **kwargs
        )


class ValidationError(BaseApplicationError):
    """バリデーション関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class BusinessRuleViolationError(BaseApplicationError):
    """ビジネスルール違反エラー"""
    
    def __init__(self, message: str, rule_name: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        self.rule_name = rule_name


class InsufficientDataError(BaseApplicationError):
    """データ不足エラー"""
    
    def __init__(self, message: str, required_data: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        self.required_data = required_data


class BusinessLogicError(BaseApplicationError):
    """ビジネスロジック関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )


class ConfigurationError(BaseApplicationError):
    """設定関連エラー"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class RecoveryStrategy(ABC):
    """回復戦略の抽象基底クラス"""
    
    @abstractmethod
    def can_recover(self, error: BaseApplicationError) -> bool:
        """回復可能かチェック"""
        pass
    
    @abstractmethod
    def recover(self, error: BaseApplicationError, context: Dict[str, Any]) -> Any:
        """回復処理実行"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """戦略名"""
        pass


class RetryStrategy(RecoveryStrategy):
    """リトライ戦略"""
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self._attempt_counts: Dict[str, int] = {}
    
    def can_recover(self, error: BaseApplicationError) -> bool:
        """リトライ可能かチェック"""
        return (
            error.category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_API] and
            self._attempt_counts.get(error.error_id, 0) < self.max_attempts
        )
    
    def recover(self, error: BaseApplicationError, context: Dict[str, Any]) -> Any:
        """リトライ実行"""
        import time
        
        attempt = self._attempt_counts.get(error.error_id, 0) + 1
        self._attempt_counts[error.error_id] = attempt
        
        logger.info(f"Retrying operation (attempt {attempt}/{self.max_attempts})")
        time.sleep(self.delay * attempt)  # 指数バックオフ
        
        # コンテキストから元の関数を再実行
        original_func = context.get('original_func')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        if original_func:
            return original_func(*args, **kwargs)
        
        raise error
    
    @property
    def strategy_name(self) -> str:
        return "retry"


class FallbackStrategy(RecoveryStrategy):
    """フォールバック戦略"""
    
    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func
    
    def can_recover(self, error: BaseApplicationError) -> bool:
        """フォールバック可能かチェック"""
        return error.category in [
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL_API,
            ErrorCategory.CALCULATION
        ]
    
    def recover(self, error: BaseApplicationError, context: Dict[str, Any]) -> Any:
        """フォールバック実行"""
        logger.warning(f"Using fallback strategy for {error.category.value} error")
        
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        return self.fallback_func(*args, **kwargs)
    
    @property
    def strategy_name(self) -> str:
        return "fallback"


class ErrorHandler:
    """統一エラーハンドラー"""
    
    def __init__(self):
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy) -> None:
        """回復戦略追加"""
        self.recovery_strategies.append(strategy)
        logger.info(f"Added recovery strategy: {strategy.strategy_name}")
    
    def handle_error(
        self,
        error: Union[BaseApplicationError, Exception],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        エラーハンドリング
        
        Args:
            error: 発生したエラー
            context: エラーコンテキスト
            
        Returns:
            Any: 回復処理の結果（ある場合）
            
        Raises:
            BaseApplicationError: 回復できない場合
        """
        # 標準例外をアプリケーション例外に変換
        if not isinstance(error, BaseApplicationError):
            error = BaseApplicationError(
                message=str(error),
                category=ErrorCategory.SYSTEM,
                details={"original_type": type(error).__name__}
            )
        
        # エラーコンテキスト作成
        error_context = ErrorContext(
            error_id=error.error_id,
            timestamp=error.timestamp,
            severity=error.severity,
            category=error.category,
            message=error.message,
            details=error.details,
            stack_trace=traceback.format_exc(),
            user_message=error.user_message
        )
        
        # エラー履歴記録
        self.error_history.append(error_context)
        self.error_counts[error.category.value] = (
            self.error_counts.get(error.category.value, 0) + 1
        )
        
        # ログ出力
        self._log_error(error, error_context)
        
        # 回復戦略試行
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error):
                try:
                    logger.info(f"Attempting recovery with {strategy.strategy_name}")
                    result = strategy.recover(error, context or {})
                    
                    error_context.recovery_attempted = True
                    error_context.recovery_successful = True
                    
                    logger.info(f"Recovery successful with {strategy.strategy_name}")
                    return result
                    
                except Exception as recovery_error:
                    logger.warning(
                        f"Recovery failed with {strategy.strategy_name}: {recovery_error}"
                    )
                    error_context.recovery_attempted = True
                    continue
        
        # 回復できない場合はエラーを再発生
        raise error
    
    def _log_error(self, error: BaseApplicationError, context: ErrorContext) -> None:
        """エラーログ出力"""
        log_data = {
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "details": error.details
        }
        
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR]:
            log_error_with_context(error.message, **log_data)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(error.message, extra=log_data)
        else:
            logger.info(error.message, extra=log_data)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """エラー統計取得"""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_category": self.error_counts.copy(),
            "recent_errors": [
                {
                    "error_id": ctx.error_id,
                    "timestamp": ctx.timestamp.isoformat(),
                    "category": ctx.category.value,
                    "message": ctx.message,
                    "recovery_successful": ctx.recovery_successful
                }
                for ctx in self.error_history[-10:]  # 最新10件
            ]
        }


# グローバルエラーハンドラー
global_error_handler = ErrorHandler()

# デフォルト回復戦略を追加
global_error_handler.add_recovery_strategy(RetryStrategy(max_attempts=3, delay=1.0))


def error_handled(
    error_types: Union[Type[Exception], tuple] = Exception,
    recovery_strategies: Optional[List[RecoveryStrategy]] = None
):
    """
    エラーハンドリングデコレーター
    
    Args:
        error_types: キャッチする例外タイプ
        recovery_strategies: 個別の回復戦略
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                context = {
                    'original_func': func,
                    'args': args,
                    'kwargs': kwargs
                }
                
                # 個別戦略がある場合は一時的に追加
                handler = global_error_handler
                if recovery_strategies:
                    temp_handler = ErrorHandler()
                    for strategy in recovery_strategies:
                        temp_handler.add_recovery_strategy(strategy)
                    handler = temp_handler
                
                return handler.handle_error(e, context)
        
        return wrapper
    return decorator


@contextmanager
def error_context(**kwargs):
    """エラーコンテキストマネージャー"""
    try:
        yield
    except Exception as e:
        context = kwargs
        global_error_handler.handle_error(e, context)