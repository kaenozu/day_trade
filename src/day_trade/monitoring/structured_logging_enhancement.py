"""
構造化ログ強化システム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

構造化ログの徹底、ログ品質向上、コンテキスト追跡、
ログスキーマ管理によるログ分析の高度化。
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from pythonjsonlogger import jsonlogger

    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

try:
    import contextvars

    CONTEXTVARS_AVAILABLE = True
except ImportError:
    CONTEXTVARS_AVAILABLE = False


class LogSeverity(Enum):
    """ログ重要度（構造化）"""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    ALERT = "ALERT"
    EMERGENCY = "EMERGENCY"


class LogCategory(Enum):
    """ログカテゴリ"""

    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    DEBUG_INFO = "debug"
    MONITORING = "monitoring"


class LogSchemaVersion(Enum):
    """ログスキーマバージョン"""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class LogContext:
    """ログコンテキスト"""

    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    environment: str = "production"
    service_name: str = "day_trade"
    service_version: str = "1.0.0"


@dataclass
class StructuredLogEntry:
    """構造化ログエントリ"""

    timestamp: str
    level: str
    severity: int  # RFC 5424 severity levels
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: str
    thread_name: str
    process_id: int
    hostname: str

    # コンテキスト情報
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None

    # ビジネス情報
    category: Optional[str] = None
    subcategory: Optional[str] = None
    event_type: Optional[str] = None

    # 技術情報
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    error_stack_trace: Optional[str] = None

    # メタデータ
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)

    # スキーマ情報
    schema_version: str = LogSchemaVersion.V2_0.value
    log_format: str = "structured_json"

    # 環境情報
    environment: str = "production"
    service_name: str = "day_trade"
    service_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ContextualLogger:
    """コンテキスト付きロガー"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context_stack = []

        if CONTEXTVARS_AVAILABLE:
            self.trace_id_var = contextvars.ContextVar("trace_id", default=None)
            self.span_id_var = contextvars.ContextVar("span_id", default=None)
            self.user_id_var = contextvars.ContextVar("user_id", default=None)
            self.session_id_var = contextvars.ContextVar("session_id", default=None)
            self.operation_var = contextvars.ContextVar("operation", default=None)
            self.component_var = contextvars.ContextVar("component", default=None)

    def _get_current_context(self) -> LogContext:
        """現在のコンテキスト取得"""
        if CONTEXTVARS_AVAILABLE:
            return LogContext(
                trace_id=self.trace_id_var.get() or self._generate_trace_id(),
                span_id=self.span_id_var.get(),
                user_id=self.user_id_var.get(),
                session_id=self.session_id_var.get(),
                operation=self.operation_var.get(),
                component=self.component_var.get(),
            )
        else:
            # Fallback for systems without contextvars
            return LogContext(trace_id=self._generate_trace_id())

    def _generate_trace_id(self) -> str:
        """トレースID生成"""
        return str(uuid.uuid4())

    def _generate_span_id(self) -> str:
        """スパンID生成"""
        return str(uuid.uuid4())[:8]

    @contextmanager
    def context(self, **context_updates):
        """コンテキストマネージャー"""
        if CONTEXTVARS_AVAILABLE:
            # コンテキスト変数を一時的に設定
            tokens = []
            for key, value in context_updates.items():
                if hasattr(self, f"{key}_var"):
                    context_var = getattr(self, f"{key}_var")
                    token = context_var.set(value)
                    tokens.append((context_var, token))

            try:
                yield
            finally:
                # コンテキストをリセット
                for context_var, token in tokens:
                    context_var.reset(token)
        else:
            # Fallback implementation
            old_context = self._context_stack.copy() if self._context_stack else {}
            self._context_stack.append(context_updates)
            try:
                yield
            finally:
                if self._context_stack:
                    self._context_stack.pop()

    def _create_structured_log(
        self,
        level: LogSeverity,
        message: str,
        category: LogCategory = LogCategory.APPLICATION,
        **kwargs,
    ) -> StructuredLogEntry:
        """構造化ログエントリ作成"""

        # 呼び出し元情報取得
        frame = sys._getframe(2)  # 2レベル上のフレーム

        # 現在のコンテキスト取得
        context = self._get_current_context()

        # 重要度レベルマッピング
        severity_mapping = {
            LogSeverity.TRACE: 7,
            LogSeverity.DEBUG: 7,
            LogSeverity.INFO: 6,
            LogSeverity.NOTICE: 5,
            LogSeverity.WARNING: 4,
            LogSeverity.ERROR: 3,
            LogSeverity.CRITICAL: 2,
            LogSeverity.ALERT: 1,
            LogSeverity.EMERGENCY: 0,
        }

        log_entry = StructuredLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            severity=severity_mapping.get(level, 6),
            message=message,
            logger_name=self.logger.name,
            module=frame.f_code.co_filename.split(os.sep)[-1],
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=str(threading.get_ident()),
            thread_name=threading.current_thread().name,
            process_id=os.getpid(),
            hostname=os.uname().nodename if hasattr(os, "uname") else "unknown",
            # コンテキスト
            trace_id=context.trace_id,
            span_id=context.span_id,
            user_id=context.user_id,
            session_id=context.session_id,
            request_id=context.request_id,
            correlation_id=context.correlation_id,
            operation=context.operation,
            component=context.component,
            # カテゴリ
            category=category.value,
            # 環境情報
            environment=context.environment,
            service_name=context.service_name,
            service_version=context.service_version,
        )

        # 追加フィールド設定
        for key, value in kwargs.items():
            if hasattr(log_entry, key):
                setattr(log_entry, key, value)
            else:
                log_entry.extra_data[key] = value

        return log_entry

    def trace(self, message: str, **kwargs):
        """TRACEレベルログ"""
        log_entry = self._create_structured_log(LogSeverity.TRACE, message, **kwargs)
        self.logger.debug(log_entry.to_json())

    def debug(self, message: str, **kwargs):
        """DEBUGレベルログ"""
        log_entry = self._create_structured_log(LogSeverity.DEBUG, message, **kwargs)
        self.logger.debug(log_entry.to_json())

    def info(self, message: str, **kwargs):
        """INFOレベルログ"""
        log_entry = self._create_structured_log(LogSeverity.INFO, message, **kwargs)
        self.logger.info(log_entry.to_json())

    def notice(self, message: str, **kwargs):
        """NOTICEレベルログ"""
        log_entry = self._create_structured_log(LogSeverity.NOTICE, message, **kwargs)
        self.logger.info(log_entry.to_json())

    def warning(self, message: str, **kwargs):
        """WARNINGレベルログ"""
        log_entry = self._create_structured_log(LogSeverity.WARNING, message, **kwargs)
        self.logger.warning(log_entry.to_json())

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """ERRORレベルログ"""
        if exception:
            kwargs.update(
                {
                    "error_type": type(exception).__name__,
                    "error_code": getattr(exception, "code", None),
                    "error_stack_trace": traceback.format_exc(),
                }
            )

        log_entry = self._create_structured_log(LogSeverity.ERROR, message, **kwargs)
        self.logger.error(log_entry.to_json())

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """CRITICALレベルログ"""
        if exception:
            kwargs.update(
                {
                    "error_type": type(exception).__name__,
                    "error_code": getattr(exception, "code", None),
                    "error_stack_trace": traceback.format_exc(),
                }
            )

        log_entry = self._create_structured_log(LogSeverity.CRITICAL, message, **kwargs)
        self.logger.critical(log_entry.to_json())

    def business_event(self, event_type: str, message: str, **kwargs):
        """ビジネスイベントログ"""
        kwargs.update({"category": LogCategory.BUSINESS, "event_type": event_type})
        log_entry = self._create_structured_log(LogSeverity.INFO, message, **kwargs)
        self.logger.info(log_entry.to_json())

    def security_event(
        self,
        event_type: str,
        message: str,
        severity: LogSeverity = LogSeverity.WARNING,
        **kwargs,
    ):
        """セキュリティイベントログ"""
        kwargs.update({"category": LogCategory.SECURITY, "event_type": event_type})
        log_entry = self._create_structured_log(severity, message, **kwargs)

        if severity in [LogSeverity.ERROR, LogSeverity.CRITICAL]:
            self.logger.error(log_entry.to_json())
        elif severity == LogSeverity.WARNING:
            self.logger.warning(log_entry.to_json())
        else:
            self.logger.info(log_entry.to_json())

    def audit_event(self, event_type: str, message: str, **kwargs):
        """監査イベントログ"""
        kwargs.update({"category": LogCategory.AUDIT, "event_type": event_type})
        log_entry = self._create_structured_log(LogSeverity.INFO, message, **kwargs)
        self.logger.info(log_entry.to_json())

    def performance_event(
        self, operation: str, duration: float, message: str, **kwargs
    ):
        """パフォーマンスイベントログ"""
        kwargs.update(
            {
                "category": LogCategory.PERFORMANCE,
                "operation": operation,
                "duration": duration,
                "metrics": {"duration_ms": duration * 1000},
            }
        )
        log_entry = self._create_structured_log(LogSeverity.INFO, message, **kwargs)
        self.logger.info(log_entry.to_json())


class StructuredLoggingEnhancementSystem:
    """構造化ログ強化システム"""

    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.loggers: Dict[str, ContextualLogger] = {}
        self._setup_root_logger()

    def _setup_root_logger(self):
        """ルートロガー設定"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))

        # 既存ハンドラーをクリア
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 構造化JSON出力ハンドラー設定
        handler = logging.StreamHandler(sys.stdout)

        if JSON_LOGGER_AVAILABLE:
            # pythonjsonloggerを使用
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={
                    "asctime": "timestamp",
                    "name": "logger_name",
                    "levelname": "level",
                },
            )
        else:
            # 標準フォーマッター（構造化されていない）
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    def get_logger(self, name: str) -> ContextualLogger:
        """コンテキスト付きロガー取得"""
        if name not in self.loggers:
            base_logger = logging.getLogger(name)
            self.loggers[name] = ContextualLogger(base_logger)
        return self.loggers[name]

    def create_trading_logger(self) -> ContextualLogger:
        """取引用ロガー作成"""
        logger = self.get_logger("day_trade.trading")
        return logger

    def create_ml_logger(self) -> ContextualLogger:
        """ML用ロガー作成"""
        logger = self.get_logger("day_trade.ml")
        return logger

    def create_security_logger(self) -> ContextualLogger:
        """セキュリティ用ロガー作成"""
        logger = self.get_logger("day_trade.security")
        return logger

    def create_api_logger(self) -> ContextualLogger:
        """API用ロガー作成"""
        logger = self.get_logger("day_trade.api")
        return logger

    def create_performance_logger(self) -> ContextualLogger:
        """パフォーマンス用ロガー作成"""
        logger = self.get_logger("day_trade.performance")
        return logger


class LogQualityValidator:
    """ログ品質バリデーター"""

    def __init__(self):
        self.required_fields = {
            "timestamp",
            "level",
            "message",
            "logger_name",
            "trace_id",
            "service_name",
            "environment",
        }

        self.quality_rules = [
            self._validate_required_fields,
            self._validate_timestamp_format,
            self._validate_log_level,
            self._validate_message_content,
            self._validate_trace_id_format,
            self._validate_structured_data,
        ]

    def validate_log_entry(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """ログエントリ品質検証"""
        validation_result = {
            "is_valid": True,
            "quality_score": 100.0,
            "violations": [],
            "suggestions": [],
        }

        for rule in self.quality_rules:
            try:
                rule_result = rule(log_data)
                if not rule_result["is_valid"]:
                    validation_result["is_valid"] = False
                    validation_result["violations"].extend(rule_result["violations"])
                    validation_result["suggestions"].extend(rule_result["suggestions"])

                # 品質スコア計算（減点方式）
                validation_result["quality_score"] -= rule_result.get("penalty", 0)

            except Exception as e:
                validation_result["violations"].append(
                    f"バリデーションルール実行エラー: {e}"
                )
                validation_result["quality_score"] -= 10

        validation_result["quality_score"] = max(0, validation_result["quality_score"])
        return validation_result

    def _validate_required_fields(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """必須フィールド検証"""
        missing_fields = self.required_fields - set(log_data.keys())

        if missing_fields:
            return {
                "is_valid": False,
                "violations": [f"必須フィールド不足: {', '.join(missing_fields)}"],
                "suggestions": [
                    f"必須フィールドを追加してください: {', '.join(missing_fields)}"
                ],
                "penalty": len(missing_fields) * 15,
            }

        return {"is_valid": True, "penalty": 0}

    def _validate_timestamp_format(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """タイムスタンプ形式検証"""
        timestamp = log_data.get("timestamp", "")

        try:
            # ISO 8601形式かチェック
            if timestamp:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return {"is_valid": True, "penalty": 0}
            else:
                return {
                    "is_valid": False,
                    "violations": ["タイムスタンプが設定されていません"],
                    "suggestions": ["ISO 8601形式のタイムスタンプを設定してください"],
                    "penalty": 10,
                }
        except ValueError:
            return {
                "is_valid": False,
                "violations": ["タイムスタンプ形式が不正です"],
                "suggestions": [
                    "ISO 8601形式（例: 2024-01-01T12:00:00Z）を使用してください"
                ],
                "penalty": 10,
            }

    def _validate_log_level(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """ログレベル検証"""
        level = log_data.get("level", "").upper()
        valid_levels = {e.value for e in LogSeverity}

        if level not in valid_levels:
            return {
                "is_valid": False,
                "violations": [f"無効なログレベル: {level}"],
                "suggestions": [
                    f"有効なログレベルを使用してください: {', '.join(valid_levels)}"
                ],
                "penalty": 5,
            }

        return {"is_valid": True, "penalty": 0}

    def _validate_message_content(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """メッセージ内容検証"""
        message = log_data.get("message", "")

        violations = []
        suggestions = []
        penalty = 0

        if not message or not message.strip():
            violations.append("メッセージが空です")
            suggestions.append("意味のあるメッセージを記述してください")
            penalty += 15

        elif len(message) < 5:
            violations.append("メッセージが短すぎます")
            suggestions.append("より詳細なメッセージを記述してください")
            penalty += 5

        elif len(message) > 1000:
            violations.append("メッセージが長すぎます")
            suggestions.append(
                "メッセージを簡潔にするか、詳細は構造化データに含めてください"
            )
            penalty += 5

        # 機密情報チェック
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "key",
            "credit_card",
            "ssn",
        ]
        if any(pattern in message.lower() for pattern in sensitive_patterns):
            violations.append("機密情報の可能性があります")
            suggestions.append("機密情報はログに出力せず、マスキングしてください")
            penalty += 20

        if violations:
            return {
                "is_valid": False,
                "violations": violations,
                "suggestions": suggestions,
                "penalty": penalty,
            }

        return {"is_valid": True, "penalty": 0}

    def _validate_trace_id_format(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """トレースID形式検証"""
        trace_id = log_data.get("trace_id", "")

        if not trace_id:
            return {
                "is_valid": False,
                "violations": ["トレースIDが設定されていません"],
                "suggestions": ["リクエスト追跡のためトレースIDを設定してください"],
                "penalty": 10,
            }

        # UUID形式かチェック（簡単な検証）
        if len(trace_id) < 8:
            return {
                "is_valid": False,
                "violations": ["トレースID形式が不正です"],
                "suggestions": ["UUID形式のトレースIDを使用してください"],
                "penalty": 5,
            }

        return {"is_valid": True, "penalty": 0}

    def _validate_structured_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """構造化データ検証"""
        violations = []
        suggestions = []
        penalty = 0

        # ネストレベルチェック
        if self._get_max_depth(log_data) > 5:
            violations.append("ネストレベルが深すぎます")
            suggestions.append("構造化データのネストを5レベル以下に制限してください")
            penalty += 5

        # フィールド数チェック
        if len(log_data) > 50:
            violations.append("フィールド数が多すぎます")
            suggestions.append(
                "不要なフィールドを削除し、関連するデータをグループ化してください"
            )
            penalty += 5

        if violations:
            return {
                "is_valid": False,
                "violations": violations,
                "suggestions": suggestions,
                "penalty": penalty,
            }

        return {"is_valid": True, "penalty": 0}

    def _get_max_depth(self, data: Any, current_depth: int = 0) -> int:
        """データの最大深度取得"""
        if not isinstance(data, dict):
            return current_depth

        if not data:
            return current_depth

        return max(
            self._get_max_depth(value, current_depth + 1) for value in data.values()
        )


# Factory functions
def create_structured_logging_system(
    log_level: str = "INFO",
) -> StructuredLoggingEnhancementSystem:
    """構造化ログ強化システム作成"""
    return StructuredLoggingEnhancementSystem(log_level)


def create_log_quality_validator() -> LogQualityValidator:
    """ログ品質バリデーター作成"""
    return LogQualityValidator()


# グローバルインスタンス
_structured_logging_system = None
_log_quality_validator = None


def get_structured_logging_system() -> StructuredLoggingEnhancementSystem:
    """グローバル構造化ログシステム取得"""
    global _structured_logging_system
    if _structured_logging_system is None:
        _structured_logging_system = create_structured_logging_system()
    return _structured_logging_system


def get_log_quality_validator() -> LogQualityValidator:
    """グローバルログ品質バリデーター取得"""
    global _log_quality_validator
    if _log_quality_validator is None:
        _log_quality_validator = create_log_quality_validator()
    return _log_quality_validator


if __name__ == "__main__":
    # テスト実行
    def test_structured_logging_enhancement():
        print("=== 構造化ログ強化システムテスト ===")

        try:
            # 構造化ログシステム初期化
            logging_system = create_structured_logging_system("DEBUG")
            validator = create_log_quality_validator()

            print("\n1. 構造化ログシステム初期化完了")

            # 各種ロガー作成
            trading_logger = logging_system.create_trading_logger()
            ml_logger = logging_system.create_ml_logger()
            security_logger = logging_system.create_security_logger()
            api_logger = logging_system.create_api_logger()

            print(f"\n2. 専用ロガー作成完了: {len(logging_system.loggers)}個")

            # コンテキスト付きログテスト
            print("\n3. コンテキスト付きログテスト...")

            with trading_logger.context(
                user_id="user123",
                operation="stock_purchase",
                component="trading_engine",
            ):
                trading_logger.info(
                    "株式購入注文を処理しました",
                    symbol="7203",
                    quantity=100,
                    price=2500,
                    order_type="market",
                )

                trading_logger.business_event(
                    "order_placed",
                    "新規注文が配置されました",
                    symbol="7203",
                    side="buy",
                    amount=250000,
                )

            # MLログテスト
            print("\n4. MLログテスト...")

            with ml_logger.context(
                user_id="system",
                operation="model_inference",
                component="lstm_predictor",
            ):
                start_time = time.time()

                # 模擬推論処理
                time.sleep(0.1)

                duration = time.time() - start_time

                ml_logger.performance_event(
                    "lstm_inference",
                    duration,
                    "LSTM株価予測を実行しました",
                    model="lstm_v2",
                    symbol="7203",
                    prediction_accuracy=0.87,
                    input_features=20,
                )

            # セキュリティログテスト
            print("\n5. セキュリティログテスト...")

            security_logger.security_event(
                "failed_login",
                "ログイン試行が失敗しました",
                LogSeverity.WARNING,
                user_id="unknown",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0...",
                attempt_count=3,
            )

            security_logger.audit_event(
                "permission_change",
                "ユーザー権限が変更されました",
                user_id="admin",
                target_user="user123",
                old_role="user",
                new_role="trader",
            )

            # エラーログテスト
            print("\n6. エラーログテスト...")

            try:
                # 意図的にエラーを発生
                raise ValueError("テスト用のエラーです")
            except Exception as e:
                api_logger.error(
                    "API呼び出し中にエラーが発生しました",
                    exception=e,
                    endpoint="/api/trades",
                    method="POST",
                    status_code=500,
                )

            # ログ品質検証テスト
            print("\n7. ログ品質検証テスト...")

            # 良質なログサンプル
            good_log = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "message": "株式注文が正常に処理されました",
                "logger_name": "day_trade.trading",
                "trace_id": str(uuid.uuid4()),
                "service_name": "day_trade",
                "environment": "production",
                "user_id": "user123",
                "operation": "place_order",
                "symbol": "7203",
            }

            good_validation = validator.validate_log_entry(good_log)
            print(
                f"   良質ログ: 有効={good_validation['is_valid']}, スコア={good_validation['quality_score']}"
            )

            # 問題のあるログサンプル
            bad_log = {
                "level": "INVALID",
                "message": "",
                "password": "secret123",  # 機密情報
            }

            bad_validation = validator.validate_log_entry(bad_log)
            print(
                f"   問題ログ: 有効={bad_validation['is_valid']}, スコア={bad_validation['quality_score']}"
            )
            print(f"   違反: {len(bad_validation['violations'])}件")

            if bad_validation["violations"]:
                for violation in bad_validation["violations"][:3]:
                    print(f"     - {violation}")

            # パフォーマンステスト
            print("\n8. パフォーマンステスト...")

            start_time = time.time()

            for i in range(100):
                trading_logger.info(
                    f"パフォーマンステストメッセージ {i}",
                    test_iteration=i,
                    batch_id="perf_test_001",
                )

            duration = time.time() - start_time
            print(f"   100件ログ出力時間: {duration:.3f}秒")
            print(f"   スループット: {100 / duration:.1f}件/秒")

            print("\n✅ 構造化ログ強化システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    test_structured_logging_enhancement()
