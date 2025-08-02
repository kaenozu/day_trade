"""
構造化ロギング設定モジュール

アプリケーション全体で使用する構造化ロギング機能を提供。
JSON形式での出力、フィルタリング、ログレベル管理を統一。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from structlog.types import Processor


class LoggingConfig:
    """ロギング設定管理クラス"""

    def __init__(self):
        self.is_configured = False
        self.log_level = self._get_log_level()
        self.log_format = self._get_log_format()

    def _get_log_level(self) -> str:
        """環境変数からログレベルを取得"""
        return os.getenv("LOG_LEVEL", "INFO").upper()

    def _get_log_format(self) -> str:
        """環境変数からログフォーマットを取得"""
        return os.getenv("LOG_FORMAT", "json").lower()

    def configure_logging(self) -> None:
        """構造化ロギングを設定"""
        if self.is_configured:
            return

        # 標準ログレベルの設定
        log_level = getattr(logging, self.log_level, logging.INFO)

        # structlogの設定
        processors = self._get_processors()

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # 標準ライブラリのloggingの設定
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
        )

        # サードパーティライブラリのログレベル調整
        self._configure_third_party_logging()

        self.is_configured = True

    def _get_processors(self) -> list[Processor]:
        """ログプロセッサーを取得"""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        # 開発環境か本番環境かでフォーマットを変更
        if self._is_development():
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=True)
            ])
        else:
            processors.extend([
                structlog.processors.JSONRenderer()
            ])

        return processors

    def _is_development(self) -> bool:
        """開発環境かどうかを判定"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        return env in ("development", "dev", "local")

    def _configure_third_party_logging(self) -> None:
        """サードパーティライブラリのログレベルを調整"""
        # SQLAlchemyのログを制限
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)

        # urllib3のログを制限
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        # requests関連のログを制限
        logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)

    def get_logger(self, name: str) -> Any:
        """構造化ロガーを取得"""
        if not self.is_configured:
            self.configure_logging()
        return structlog.get_logger(name)


# グローバルインスタンス
_logging_config = LoggingConfig()


def setup_logging() -> None:
    """ロギング設定を初期化"""
    _logging_config.configure_logging()


def get_logger(name: str = None) -> Any:
    """構造化ロガーを取得"""
    if name is None:
        # 呼び出し元のモジュール名を自動取得
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')

    return _logging_config.get_logger(name)


class ContextLogger:
    """コンテキスト情報を保持するロガーラッパー"""

    def __init__(self, logger: Any, context: Dict[str, Any] = None):
        self.logger = logger
        self.context = context or {}

    def bind(self, **kwargs) -> "ContextLogger":
        """新しいコンテキストでロガーを作成"""
        new_context = {**self.context, **kwargs}
        return ContextLogger(self.logger, new_context)

    def info(self, message: str, **kwargs) -> None:
        """情報ログ出力"""
        self.logger.info(message, **{**self.context, **kwargs})

    def warning(self, message: str, **kwargs) -> None:
        """警告ログ出力"""
        self.logger.warning(message, **{**self.context, **kwargs})

    def error(self, message: str, **kwargs) -> None:
        """エラーログ出力"""
        self.logger.error(message, **{**self.context, **kwargs})

    def debug(self, message: str, **kwargs) -> None:
        """デバッグログ出力"""
        self.logger.debug(message, **{**self.context, **kwargs})

    def critical(self, message: str, **kwargs) -> None:
        """クリティカルログ出力"""
        self.logger.critical(message, **{**self.context, **kwargs})


def get_context_logger(name: str = None, **context) -> ContextLogger:
    """コンテキスト付きロガーを取得"""
    logger = get_logger(name)
    return ContextLogger(logger, context)


# 便利な関数群
def log_function_call(func_name: str, **kwargs) -> None:
    """関数呼び出しをログ出力"""
    logger = get_logger()
    logger.info("Function called", function=func_name, **kwargs)


def log_error_with_context(error: Exception, context: Dict[str, Any] = None) -> None:
    """エラー情報をコンテキスト付きでログ出力"""
    logger = get_logger()
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {})
    }
    logger.error("Error occurred", **error_context)


def log_performance_metric(metric_name: str, value: float, unit: str = "ms", **kwargs) -> None:
    """パフォーマンスメトリクスをログ出力"""
    logger = get_logger()
    logger.info(
        "Performance metric",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )


def log_business_event(event_name: str, **kwargs) -> None:
    """ビジネスイベントをログ出力"""
    logger = get_logger()
    logger.info("Business event", event_name=event_name, **kwargs)


def log_database_operation(operation: str, table: str, **kwargs) -> None:
    """データベース操作をログ出力"""
    logger = get_logger()
    logger.info(
        "Database operation",
        operation=operation,
        table=table,
        **kwargs
    )


def log_api_call(api_name: str, method: str, url: str, status_code: int = None, **kwargs) -> None:
    """API呼び出しをログ出力"""
    logger = get_logger()
    logger.info(
        "API call",
        api_name=api_name,
        method=method,
        url=url,
        status_code=status_code,
        **kwargs
    )


def log_security_event(event_type: str, severity: str = "info", **kwargs) -> None:
    """セキュリティイベントをログ出力"""
    logger = get_logger()

    log_data = {
        "event_type": event_type,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }

    if severity == "critical":
        logger.critical("Security event", **log_data)
    elif severity == "warning":
        logger.warning("Security event", **log_data)
    else:
        logger.info("Security event", **log_data)


def log_user_action(action: str, user_id: str = None, **kwargs) -> None:
    """ユーザーアクションをログ出力"""
    logger = get_logger()
    logger.info(
        "User action",
        action=action,
        user_id=user_id or "anonymous",
        timestamp=datetime.now().isoformat(),
        **kwargs
    )


def log_system_health(component: str, status: str, **metrics) -> None:
    """システムヘルス情報をログ出力"""
    logger = get_logger()
    logger.info(
        "System health",
        component=component,
        status=status,
        timestamp=datetime.now().isoformat(),
        **metrics
    )


class AlertThreshold:
    """アラート閾値設定"""

    def __init__(self):
        self.error_count_threshold = int(os.getenv("ALERT_ERROR_THRESHOLD", "10"))
        self.response_time_threshold = float(os.getenv("ALERT_RESPONSE_TIME_THRESHOLD", "5.0"))
        self.memory_usage_threshold = float(os.getenv("ALERT_MEMORY_THRESHOLD", "80.0"))


class LoggingAlert:
    """ログベースアラート機能"""

    def __init__(self):
        self.thresholds = AlertThreshold()
        self.error_count = 0
        self.alert_enabled = os.getenv("ENABLE_LOGGING_ALERTS", "false").lower() == "true"

    def check_error_threshold(self) -> None:
        """エラー数閾値チェック"""
        if not self.alert_enabled:
            return

        self.error_count += 1
        if self.error_count >= self.thresholds.error_count_threshold:
            log_security_event(
                "error_threshold_exceeded",
                severity="warning",
                error_count=self.error_count,
                threshold=self.thresholds.error_count_threshold
            )
            self.error_count = 0  # リセット

    def check_performance_threshold(self, metric: str, value: float) -> None:
        """パフォーマンス閾値チェック"""
        if not self.alert_enabled:
            return

        threshold_exceeded = False

        if metric == "response_time" and value > self.thresholds.response_time_threshold:
            threshold_exceeded = True
        elif metric == "memory_usage" and value > self.thresholds.memory_usage_threshold:
            threshold_exceeded = True

        if threshold_exceeded:
            log_security_event(
                "performance_threshold_exceeded",
                severity="warning",
                metric=metric,
                value=value,
                threshold=getattr(self.thresholds, f"{metric}_threshold")
            )


# グローバルアラートインスタンス
_alert_manager = LoggingAlert()


def setup_logging_with_alerts() -> None:
    """アラート機能付きロギング設定"""
    setup_logging()

    # カスタムハンドラーでアラート機能を統合
    if _alert_manager.alert_enabled:
        logger = get_logger()
        logger.info("ログベースアラート機能を有効化",
                   error_threshold=_alert_manager.thresholds.error_count_threshold,
                   response_time_threshold=_alert_manager.thresholds.response_time_threshold,
                   memory_threshold=_alert_manager.thresholds.memory_usage_threshold)


def enhanced_log_error_with_context(error: Exception, context: Dict[str, Any] = None) -> None:
    """エラー情報をコンテキスト付きでログ出力（アラート機能付き）"""
    log_error_with_context(error, context)
    _alert_manager.check_error_threshold()


def enhanced_log_performance_metric(metric_name: str, value: float, unit: str = "ms", **kwargs) -> None:
    """パフォーマンスメトリクスをログ出力（アラート機能付き）"""
    log_performance_metric(metric_name, value, unit, **kwargs)

    # 特定のメトリクスに対してアラートチェック
    if metric_name in ["response_time", "memory_usage"]:
        _alert_manager.check_performance_threshold(metric_name, value)
