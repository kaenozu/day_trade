"""
Risk Management Exceptions
リスク管理例外システム

統一例外クラスと標準化されたエラーハンドリング
"""

from .risk_exceptions import (
    RiskManagementError,
    ConfigurationError,
    ValidationError,
    AnalysisError,
    CacheError,
    AlertError,
    SecurityError,
    TimeoutError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    DataIntegrityError,
    ExternalServiceError,
    ResourceNotFoundError,
    ConflictError
)

from .error_codes import (
    ErrorCode,
    ErrorCategory,
    create_error_response,
    format_error_message
)

from .error_handlers import (
    ErrorHandler,
    AsyncErrorHandler,
    RiskAnalysisErrorHandler,
    DefaultErrorHandler
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
    "DefaultErrorHandler"
]
