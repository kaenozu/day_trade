"""
Risk Management Exceptions
リスク管理例外システム

統一例外クラスと標準化されたエラーハンドリング
"""

from .error_codes import (
    ErrorCategory,
    ErrorCode,
    create_error_response,
    format_error_message,
)
from .error_handlers import (
    AsyncErrorHandler,
    DefaultErrorHandler,
    ErrorHandler,
    RiskAnalysisErrorHandler,
)
from .risk_exceptions import (
    AlertError,
    AnalysisError,
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ConfigurationError,
    ConflictError,
    DataIntegrityError,
    ExternalServiceError,
    RateLimitError,
    ResourceNotFoundError,
    RiskManagementError,
    SecurityError,
    TimeoutError,
    ValidationError,
)

__all__ = [
    # 例外クラス
    "RiskManagementError",
    "ConfigurationError",
    "ValidationError",
    "AnalysisError",
    "CacheError",
    "AlertError",
    "SecurityError",
    "TimeoutError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    "DataIntegrityError",
    "ExternalServiceError",
    "ResourceNotFoundError",
    "ConflictError",
    # エラーコード
    "ErrorCode",
    "ErrorCategory",
    "create_error_response",
    "format_error_message",
    # エラーハンドラー
    "ErrorHandler",
    "AsyncErrorHandler",
    "RiskAnalysisErrorHandler",
    "DefaultErrorHandler",
]
