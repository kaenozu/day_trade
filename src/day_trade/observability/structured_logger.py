"""
APM・オブザーバビリティ統合基盤 - Issue #442 Phase 2
構造化ログ・相関ID・コンテキスト追跡

このモジュールは以下を提供します:
- JSON形式の構造化ログ
- 分散トレーシングとの連携
- 相関IDによるリクエスト追跡
- セキュリティとパフォーマンス考慮
"""

import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .telemetry_config import get_logger as get_base_logger


class LogLevel(Enum):
    """ログレベル定義"""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"  # HFT用極詳細ログ


class EventCategory(Enum):
    """イベントカテゴリ定義"""

    TRADE_EXECUTION = "trade_execution"
    MARKET_DATA = "market_data"
    API_ACCESS = "api_access"
    SECURITY = "security"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    ERROR = "error"
    METRICS = "metrics"


class StructuredLogger:
    """
    構造化ログクラス

    Features:
    - JSON形式構造化出力
    - トレーシング情報自動付与
    - 相関ID管理
    - HFT対応高速ログ
    - セキュリティ情報マスキング
    """

    def __init__(
        self,
        name: str = "day-trade",
        service_version: str = "1.0.0",
        environment: str = None,
    ):
        self.name = name
        self.service_version = service_version
        self.environment = environment or os.getenv("ENVIRONMENT", "production")

        # スレッドローカル情報
        self._thread_local = threading.local()

        # 基本ログ設定
        self._base_logger = self._setup_base_logger()

        # HFT最適化設定
        self.hft_mode = os.getenv("HFT_MODE", "false").lower() == "true"
        self.enable_trace_logs = (
            os.getenv("ENABLE_TRACE_LOGS", "false").lower() == "true"
        )

        # ログサンプリング設定（HFT時）
        self._sampling_rate = float(os.getenv("LOG_SAMPLING_RATE", "1.0"))
        self._sample_counter = 0

    def _setup_base_logger(self) -> structlog.BoundLogger:
        """基本ログ設定"""

        def add_service_context(logger, method_name, event_dict):
            """サービスコンテキスト追加"""
            event_dict.update(
                {
                    "service": {
                        "name": self.name,
                        "version": self.service_version,
                        "environment": self.environment,
                    },
                    "host": {
                        "name": os.getenv("HOSTNAME", "unknown"),
                        "ip": os.getenv("HOST_IP", "unknown"),
                    },
                }
            )
            return event_dict

        def add_trace_context(logger, method_name, event_dict):
            """OpenTelemetryトレースコンテキスト追加"""
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                ctx = span.get_span_context()
                event_dict["trace"] = {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "trace_flags": format(ctx.trace_flags, "02x"),
                }
            return event_dict

        def add_correlation_context(logger, method_name, event_dict):
            """相関ID追加"""
            correlation_id = self.get_correlation_id()
            if correlation_id:
                event_dict["correlation_id"] = correlation_id

            # リクエストIDも追加
            request_id = self.get_request_id()
            if request_id:
                event_dict["request_id"] = request_id

            return event_dict

        def add_timestamp_context(logger, method_name, event_dict):
            """詳細タイムスタンプ情報追加"""
            now = datetime.now(timezone.utc)
            event_dict.update(
                {
                    "timestamp": now.isoformat(),
                    "@timestamp": now.isoformat(),
                    "timestamp_unix": now.timestamp(),
                    "timestamp_unix_ns": time.time_ns(),  # HFT用ナノ秒精度
                }
            )
            return event_dict

        def mask_sensitive_data(logger, method_name, event_dict):
            """機密情報マスキング"""
            sensitive_fields = [
                "password",
                "token",
                "api_key",
                "secret",
                "auth_token",
                "credit_card",
                "ssn",
                "account_number",
            ]

            def mask_recursive(obj):
                if isinstance(obj, dict):
                    return {
                        key: (
                            "***MASKED***"
                            if key.lower() in sensitive_fields
                            else mask_recursive(value)
                        )
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [mask_recursive(item) for item in obj]
                elif isinstance(obj, str):
                    # クレジットカード番号パターン
                    import re

                    cc_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
                    obj = re.sub(cc_pattern, "****-****-****-****", obj)

                    # メールアドレス部分マスキング
                    email_pattern = (
                        r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
                    )
                    obj = re.sub(email_pattern, r"***@\2", obj)

                return obj

            return mask_recursive(event_dict)

        # structlog設定
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                add_service_context,
                add_trace_context,
                add_correlation_context,
                add_timestamp_context,
                mask_sensitive_data,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(self.name)

    # === 相関ID管理 ===

    def set_correlation_id(self, correlation_id: str = None) -> str:
        """相関ID設定"""
        if not hasattr(self._thread_local, "correlation_id") or correlation_id:
            self._thread_local.correlation_id = correlation_id or str(uuid.uuid4())
        return self._thread_local.correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """相関ID取得"""
        return getattr(self._thread_local, "correlation_id", None)

    def set_request_id(self, request_id: str = None) -> str:
        """リクエストID設定"""
        if not hasattr(self._thread_local, "request_id") or request_id:
            self._thread_local.request_id = request_id or str(uuid.uuid4())
        return self._thread_local.request_id

    def get_request_id(self) -> Optional[str]:
        """リクエストID取得"""
        return getattr(self._thread_local, "request_id", None)

    # === HFT対応サンプリング ===

    def _should_log(self, level: LogLevel) -> bool:
        """ログ出力判定（サンプリング考慮）"""
        # エラー・クリティカルは常にログ出力
        if level in [LogLevel.CRITICAL, LogLevel.ERROR]:
            return True

        # HFT モードでのサンプリング
        if self.hft_mode and self._sampling_rate < 1.0:
            self._sample_counter += 1
            return (self._sample_counter % int(1.0 / self._sampling_rate)) == 0

        # TRACE ログは設定で制御
        if level == LogLevel.TRACE and not self.enable_trace_logs:
            return False

        return True

    # === ログ出力メソッド ===

    def log(
        self,
        level: LogLevel,
        message: str,
        category: EventCategory = EventCategory.SYSTEM,
        **kwargs,
    ) -> None:
        """汎用ログ出力"""
        if not self._should_log(level):
            return

        # イベント情報追加
        event_data = {
            "event": {
                "category": category.value,
                "action": kwargs.pop("action", None),
                "outcome": kwargs.pop("outcome", "unknown"),
                "duration_ms": kwargs.pop("duration_ms", None),
            },
            **kwargs,
        }

        # ログレベル別出力
        logger_method = getattr(self._base_logger, level.value.lower())
        logger_method(message, **event_data)

    def critical(self, message: str, **kwargs) -> None:
        """クリティカルログ"""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def error(self, message: str, error: Exception = None, **kwargs) -> None:
        """エラーログ"""
        if error:
            kwargs.update(
                {
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "stack_trace": (
                            str(error.__traceback__) if error.__traceback__ else None
                        ),
                    }
                }
            )

            # OpenTelemetryスパンにも例外記録
            span = trace.get_current_span()
            if span:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))

        self.log(LogLevel.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """警告ログ"""
        self.log(LogLevel.WARNING, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """情報ログ"""
        self.log(LogLevel.INFO, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """デバッグログ"""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def trace(self, message: str, **kwargs) -> None:
        """トレースログ（HFT用）"""
        self.log(LogLevel.TRACE, message, **kwargs)

    # === 特化ログメソッド ===

    def log_trade_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        latency_us: float,
        success: bool,
        pnl: Optional[float] = None,
        **kwargs,
    ) -> None:
        """取引実行ログ"""
        self.log(
            LogLevel.INFO,
            "Trade executed",
            category=EventCategory.TRADE_EXECUTION,
            action="execute_trade",
            outcome="success" if success else "failure",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            latency_microseconds=latency_us,
            pnl=pnl,
            **kwargs,
        )

    def log_api_access(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        client_ip: str = None,
        user_agent: str = None,
        **kwargs,
    ) -> None:
        """APIアクセスログ"""
        self.log(
            LogLevel.INFO,
            f"{method} {path}",
            category=EventCategory.API_ACCESS,
            action="api_request",
            outcome="success" if status_code < 400 else "failure",
            http={"method": method, "path": path, "status_code": status_code},
            response_time_ms=response_time_ms,
            client_ip=client_ip,
            user_agent=user_agent,
            **kwargs,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        source_ip: str = None,
        user_id: str = None,
        **kwargs,
    ) -> None:
        """セキュリティイベントログ"""
        self.log(
            LogLevel.WARNING if severity == "medium" else LogLevel.ERROR,
            message,
            category=EventCategory.SECURITY,
            action=event_type,
            outcome="detected",
            security={
                "event_type": event_type,
                "severity": severity,
                "source_ip": source_ip,
                "user_id": user_id,
            },
            **kwargs,
        )

    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str = None,
        tags: Dict[str, str] = None,
        **kwargs,
    ) -> None:
        """パフォーマンスメトリクスログ"""
        self.log(
            LogLevel.INFO,
            f"Performance metric: {metric_name}",
            category=EventCategory.PERFORMANCE,
            action="record_metric",
            metric={
                "name": metric_name,
                "value": metric_value,
                "unit": metric_unit,
                "tags": tags or {},
            },
            **kwargs,
        )

    # === コンテキストマネージャー ===

    @contextmanager
    def correlation_context(self, correlation_id: str = None):
        """相関IDコンテキスト"""
        old_correlation_id = self.get_correlation_id()
        new_correlation_id = self.set_correlation_id(correlation_id)

        try:
            yield new_correlation_id
        finally:
            if old_correlation_id:
                self.set_correlation_id(old_correlation_id)
            else:
                if hasattr(self._thread_local, "correlation_id"):
                    delattr(self._thread_local, "correlation_id")

    @contextmanager
    def request_context(self, request_id: str = None):
        """リクエストIDコンテキスト"""
        old_request_id = self.get_request_id()
        new_request_id = self.set_request_id(request_id)

        try:
            yield new_request_id
        finally:
            if old_request_id:
                self.set_request_id(old_request_id)
            else:
                if hasattr(self._thread_local, "request_id"):
                    delattr(self._thread_local, "request_id")

    @contextmanager
    def operation_context(
        self, operation_name: str, correlation_id: str = None, **context_data
    ):
        """操作コンテキスト（相関ID + 操作情報）"""
        start_time = time.perf_counter()
        start_time_ns = time.time_ns()

        with self.correlation_context(correlation_id) as cid:
            self.info(
                f"Operation started: {operation_name}",
                action="operation_start",
                operation=operation_name,
                **context_data,
            )

            try:
                yield cid

                # 成功時
                duration_ms = (time.perf_counter() - start_time) * 1000
                duration_ns = time.time_ns() - start_time_ns

                self.info(
                    f"Operation completed: {operation_name}",
                    action="operation_complete",
                    operation=operation_name,
                    outcome="success",
                    duration_ms=duration_ms,
                    duration_ns=duration_ns,
                    **context_data,
                )

            except Exception as e:
                # エラー時
                duration_ms = (time.perf_counter() - start_time) * 1000

                self.error(
                    f"Operation failed: {operation_name}",
                    error=e,
                    action="operation_failed",
                    operation=operation_name,
                    outcome="failure",
                    duration_ms=duration_ms,
                    **context_data,
                )
                raise


# グローバル構造化ログ
_global_logger: Optional[StructuredLogger] = None


def get_structured_logger(
    name: str = "day-trade", service_version: str = "1.0.0"
) -> StructuredLogger:
    """グローバル構造化ログ取得"""
    global _global_logger

    if _global_logger is None:
        _global_logger = StructuredLogger(name, service_version)

    return _global_logger


def set_correlation_id(correlation_id: str = None) -> str:
    """グローバル相関ID設定"""
    logger = get_structured_logger()
    return logger.set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """グローバル相関ID取得"""
    logger = get_structured_logger()
    return logger.get_correlation_id()


# 便利関数
def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    latency_us: float,
    success: bool,
    **kwargs,
):
    """取引ログ便利関数"""
    logger = get_structured_logger()
    logger.log_trade_execution(
        symbol, side, quantity, price, latency_us, success, **kwargs
    )


def log_api(method: str, path: str, status: int, response_time: float, **kwargs):
    """APIログ便利関数"""
    logger = get_structured_logger()
    logger.log_api_access(method, path, status, response_time, **kwargs)


def log_security(event_type: str, severity: str, message: str, **kwargs):
    """セキュリティログ便利関数"""
    logger = get_structured_logger()
    logger.log_security_event(event_type, severity, message, **kwargs)
