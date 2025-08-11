#!/usr/bin/env python3
"""
Risk Management Exception Classes
リスク管理例外クラス

統一された例外階層と詳細なエラー情報を提供
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorSeverity(Enum):
    """エラー重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskManagementError(Exception):
    """リスク管理システム基底例外"""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        self.timestamp = datetime.now()

        # 因果関係の連鎖を保持
        if cause:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


class ConfigurationError(RiskManagementError):
    """設定関連エラー"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "CONFIG_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.HIGH),
            **kwargs,
        )
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value

        if config_key:
            self.details.update(
                {
                    "config_key": config_key,
                    "expected_type": expected_type,
                    "actual_value": str(actual_value)
                    if actual_value is not None
                    else None,
                }
            )


class ValidationError(RiskManagementError):
    """データ検証エラー"""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "VALIDATION_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_rules = validation_rules or []

        self.details.update(
            {
                "field_name": field_name,
                "invalid_value": str(invalid_value)
                if invalid_value is not None
                else None,
                "validation_rules": validation_rules,
            }
        )


class AnalysisError(RiskManagementError):
    """リスク分析エラー"""

    def __init__(
        self,
        message: str,
        analyzer_name: Optional[str] = None,
        request_id: Optional[str] = None,
        analysis_stage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "ANALYSIS_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.HIGH),
            **kwargs,
        )
        self.analyzer_name = analyzer_name
        self.request_id = request_id
        self.analysis_stage = analysis_stage

        self.details.update(
            {
                "analyzer_name": analyzer_name,
                "request_id": request_id,
                "analysis_stage": analysis_stage,
            }
        )


class CacheError(RiskManagementError):
    """キャッシュ操作エラー"""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        cache_provider: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "CACHE_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.LOW),
            **kwargs,
        )
        self.cache_key = cache_key
        self.operation = operation
        self.cache_provider = cache_provider

        self.details.update(
            {
                "cache_key": cache_key,
                "operation": operation,
                "cache_provider": cache_provider,
            }
        )


class AlertError(RiskManagementError):
    """アラート関連エラー"""

    def __init__(
        self,
        message: str,
        alert_id: Optional[str] = None,
        channel: Optional[str] = None,
        notification_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "ALERT_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
        self.alert_id = alert_id
        self.channel = channel
        self.notification_type = notification_type

        self.details.update(
            {
                "alert_id": alert_id,
                "channel": channel,
                "notification_type": notification_type,
            }
        )


class SecurityError(RiskManagementError):
    """セキュリティ関連エラー"""

    def __init__(
        self, message: str, security_context: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "SECURITY_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.CRITICAL),
            **kwargs,
        )
        self.security_context = security_context or {}
        self.details.update({"security_context": self.security_context})


class AuthenticationError(SecurityError):
    """認証エラー"""

    def __init__(self, message: str, username: Optional[str] = None, **kwargs):
        super().__init__(
            message, error_code=kwargs.get("error_code", "AUTH_ERROR"), **kwargs
        )
        self.username = username
        self.details.update({"username": username})


class AuthorizationError(SecurityError):
    """認可エラー"""

    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message, error_code=kwargs.get("error_code", "AUTHZ_ERROR"), **kwargs
        )
        self.required_permission = required_permission
        self.resource = resource

        self.details.update(
            {"required_permission": required_permission, "resource": resource}
        )


class TimeoutError(RiskManagementError):
    """タイムアウトエラー"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "TIMEOUT_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.HIGH),
            **kwargs,
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds

        self.details.update(
            {"operation": operation, "timeout_seconds": timeout_seconds}
        )


class RateLimitError(RiskManagementError):
    """レート制限エラー"""

    def __init__(
        self,
        message: str,
        rate_limit: Optional[int] = None,
        retry_after_seconds: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "RATE_LIMIT_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
        self.rate_limit = rate_limit
        self.retry_after_seconds = retry_after_seconds

        self.details.update(
            {"rate_limit": rate_limit, "retry_after_seconds": retry_after_seconds}
        )


class DataIntegrityError(RiskManagementError):
    """データ整合性エラー"""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "DATA_INTEGRITY_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.HIGH),
            **kwargs,
        )
        self.data_source = data_source
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum

        self.details.update(
            {
                "data_source": data_source,
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
            }
        )


class ExternalServiceError(RiskManagementError):
    """外部サービスエラー"""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "EXTERNAL_SERVICE_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.HIGH),
            **kwargs,
        )
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body

        self.details.update(
            {
                "service_name": service_name,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_body": response_body[:500]
                if response_body
                else None,  # 長いレスポンスは切り詰め
            }
        )


class ResourceNotFoundError(RiskManagementError):
    """リソース未発見エラー"""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "RESOURCE_NOT_FOUND"),
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id

        self.details.update(
            {"resource_type": resource_type, "resource_id": resource_id}
        )


class ConflictError(RiskManagementError):
    """競合エラー"""

    def __init__(
        self,
        message: str,
        conflicting_resource: Optional[str] = None,
        current_version: Optional[str] = None,
        expected_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "CONFLICT_ERROR"),
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
        self.conflicting_resource = conflicting_resource
        self.current_version = current_version
        self.expected_version = expected_version

        self.details.update(
            {
                "conflicting_resource": conflicting_resource,
                "current_version": current_version,
                "expected_version": expected_version,
            }
        )


# 例外作成ヘルパー関数


def create_validation_error(
    field_name: str,
    value: Any,
    message: Optional[str] = None,
    rules: Optional[List[str]] = None,
) -> ValidationError:
    """検証エラー作成ヘルパー"""
    if message is None:
        message = f"Invalid value for field '{field_name}': {value}"

    return ValidationError(
        message=message,
        field_name=field_name,
        invalid_value=value,
        validation_rules=rules,
    )


def create_analysis_error(
    analyzer_name: str, request_id: str, stage: str, cause: Exception
) -> AnalysisError:
    """分析エラー作成ヘルパー"""
    return AnalysisError(
        message=f"Analysis failed in {stage} stage of {analyzer_name}",
        analyzer_name=analyzer_name,
        request_id=request_id,
        analysis_stage=stage,
        cause=cause,
    )


def create_timeout_error(operation: str, timeout_seconds: float) -> TimeoutError:
    """タイムアウトエラー作成ヘルパー"""
    return TimeoutError(
        message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
        operation=operation,
        timeout_seconds=timeout_seconds,
    )


def create_external_service_error(
    service_name: str,
    endpoint: str,
    status_code: int,
    response_body: Optional[str] = None,
) -> ExternalServiceError:
    """外部サービスエラー作成ヘルパー"""
    return ExternalServiceError(
        message=f"External service '{service_name}' returned error {status_code}",
        service_name=service_name,
        endpoint=endpoint,
        status_code=status_code,
        response_body=response_body,
    )
