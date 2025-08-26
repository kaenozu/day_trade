"""
キャッシュ関連の定数と例外定義

安全で効率的なキャッシュ操作のための基本的な定数と例外クラスを提供します。
"""

from typing import Any, Dict, Optional


class CacheConstants:
    """キャッシュ関連の定数定義"""

    # デフォルト値
    DEFAULT_MAX_KEY_LENGTH = 1000
    DEFAULT_MAX_VALUE_SIZE_MB = 10
    DEFAULT_MAX_RECURSION_DEPTH = 10
    DEFAULT_MIN_RECURSION_DEPTH = 3
    DEFAULT_MAX_ADAPTIVE_DEPTH = 20
    DEFAULT_SERIALIZATION_TIMEOUT = 5.0
    DEFAULT_LOCK_TIMEOUT = 1.0

    # キャッシュ統計
    DEFAULT_MAX_OPERATION_HISTORY = 1000
    DEFAULT_HIT_RATE_WINDOW_SIZE = 100
    DEFAULT_STATS_HISTORY_CLEANUP_SECONDS = 300  # 5分

    # TTLキャッシュ
    DEFAULT_TTL_CACHE_SIZE = 5000
    DEFAULT_TTL_SECONDS = 600  # 10分
    DEFAULT_CLEANUP_FREQUENCY = 100
    DEFAULT_CLEANUP_THRESHOLD_RATIO = 0.8

    # 高性能キャッシュ
    DEFAULT_HIGH_PERF_CACHE_SIZE = 10000
    DEFAULT_HIGH_PERF_CLEANUP_RATIO = 0.7

    # エラー処理
    MAX_COUNTER_VALUE = 2**63 - 1  # 64bit符号付き整数の最大値
    ERROR_PENALTY_MULTIPLIER = 50

    # シリアライゼーション
    MAX_SET_SORT_ATTEMPTS = 3
    CHARSET_DETECTION_CONFIDENCE_THRESHOLD = 0.7


class CacheError(Exception):
    """キャッシュ操作エラーの基底例外クラス"""

    def __init__(
        self,
        message: str,
        error_code: str = "CACHE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: 詳細情報（オプション）
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class CacheCircuitBreakerError(CacheError):
    """キャッシュサーキットブレーカーエラー"""

    def __init__(self, message: str, circuit_state: str, failure_count: int):
        """
        Args:
            message: エラーメッセージ
            circuit_state: サーキットブレーカーの状態
            failure_count: 失敗回数
        """
        super().__init__(
            message,
            "CACHE_CIRCUIT_BREAKER",
            {"circuit_state": circuit_state, "failure_count": failure_count},
        )
        self.circuit_state = circuit_state
        self.failure_count = failure_count


class CacheTimeoutError(CacheError):
    """キャッシュタイムアウトエラー"""

    def __init__(self, message: str, timeout_seconds: float, operation: str):
        """
        Args:
            message: エラーメッセージ
            timeout_seconds: タイムアウト秒数
            operation: タイムアウトした操作
        """
        super().__init__(
            message,
            "CACHE_TIMEOUT",
            {"timeout_seconds": timeout_seconds, "operation": operation},
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class CacheKeyError(CacheError):
    """キャッシュキー関連エラー"""

    def __init__(self, message: str, key: str, operation: str):
        """
        Args:
            message: エラーメッセージ
            key: 問題のあるキー
            operation: エラーが発生した操作
        """
        super().__init__(
            message,
            "CACHE_KEY_ERROR",
            {"key": key, "operation": operation},
        )
        self.key = key
        self.operation = operation


class CacheSerializationError(CacheError):
    """キャッシュシリアライゼーションエラー"""

    def __init__(self, message: str, object_type: str, original_error: Exception):
        """
        Args:
            message: エラーメッセージ
            object_type: シリアライゼーションに失敗したオブジェクトの型
            original_error: 元の例外
        """
        super().__init__(
            message,
            "CACHE_SERIALIZATION_ERROR",
            {
                "object_type": object_type,
                "original_error": str(original_error),
                "original_error_type": type(original_error).__name__,
            },
        )
        self.object_type = object_type
        self.original_error = original_error


class CacheValidationError(CacheError):
    """キャッシュバリデーションエラー"""

    def __init__(self, message: str, validation_type: str, value: Any):
        """
        Args:
            message: エラーメッセージ
            validation_type: バリデーションの種類
            value: バリデーションに失敗した値
        """
        super().__init__(
            message,
            "CACHE_VALIDATION_ERROR",
            {"validation_type": validation_type, "value": str(value)[:100]},
        )
        self.validation_type = validation_type
        self.value = value


class CacheConfigurationError(CacheError):
    """キャッシュ設定エラー"""

    def __init__(self, message: str, config_key: str, config_value: Any):
        """
        Args:
            message: エラーメッセージ
            config_key: 問題のある設定キー
            config_value: 問題のある設定値
        """
        super().__init__(
            message,
            "CACHE_CONFIGURATION_ERROR",
            {"config_key": config_key, "config_value": str(config_value)},
        )
        self.config_key = config_key
        self.config_value = config_value