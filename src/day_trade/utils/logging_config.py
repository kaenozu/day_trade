"""
構造化ロギング設定モジュール

アプリケーション全体で使用する構造化ロギング機能を提供。
JSON形式での出力、フィルタリング、ログレベル管理を統一。
"""

import logging
import os
import sys
from typing import Any, Dict

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class ContextLogger:
    """コンテキスト付きロガー"""

    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        if STRUCTLOG_AVAILABLE:
            self.logger = logger.bind(**(context or {}))
        else:
            self.logger = logger
            self.context = context or {}

    def bind(self, **kwargs) -> "ContextLogger":
        """コンテキストを追加したロガーを作成"""
        if STRUCTLOG_AVAILABLE:
            return ContextLogger(self.logger.bind(**kwargs))
        else:
            new_context = {**self.context, **kwargs}
            return ContextLogger(self.logger, new_context)

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """コンテキスト付きでログ出力"""
        if STRUCTLOG_AVAILABLE:
            self.logger.log(level, msg, *args, **kwargs)
        else:
            extra = kwargs.get("extra", {})
            extra.update(self.context)

            # Loggerが認識しないキーワード引数を除去
            logger_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["extra", "exc_info", "stack_info", "stacklevel"]
            }
            logger_kwargs["extra"] = extra

            self.logger.log(level, msg, *args, **logger_kwargs)

    def info(self, msg: str, *args, **kwargs):
        """インフォログ出力"""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """警告ログ出力"""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """エラーログ出力"""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """デバッグログ出力"""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """クリティカルログ出力"""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


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
        return os.getenv("LOG_FORMAT", "simple").lower()

    def configure_logging(self) -> None:
        """基本ロギングを設定"""
        if self.is_configured:
            return

        if STRUCTLOG_AVAILABLE:
            # Structlogの設定
            processors = [
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_logger_oid, # Python 3.12+ の場合
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.CallsiteParameterAdder(
                    {
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    }
                ),
            ]

            if self.log_format == "json":
                processors.append(structlog.processors.JSONRenderer())
            else:
                processors.append(structlog.dev.ConsoleRenderer())

            structlog.configure(
                processors=processors,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

            # 標準ロガーをstructlogにフック
            logging.basicConfig(
                format="%(message)s", stream=sys.stdout, level=self.log_level
            )
        else:
            # Structlogが利用できない場合のフォールバック
            logging.basicConfig(
                level=getattr(logging, self.log_level, logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )

        self.is_configured = True

    # グローバルロギング設定インスタンス
_logging_config = LoggingConfig()


def setup_logging():
    """ロギング設定を初期化"""
    _logging_config.configure_logging()


def get_logger(name: str) -> logging.Logger:
    """
    標準ロガーを取得

    Args:
        name: ロガー名

    Returns:
        設定済みロガー
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


def get_context_logger(name: str, component: str = None, **kwargs) -> ContextLogger:
    """コンテキスト付きロガーを取得"""
    logger_name = f"{name}.{component}" if component else name
    logger = get_logger(logger_name)

    return ContextLogger(logger, kwargs)


def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """
    コンテキスト付きでエラーをログ出力

    Args:
        error: エラーオブジェクト
        context: コンテキスト情報
    """
    logger = logging.getLogger(__name__)
    logger.error(f"Error occurred: {error}. Context: {context}", exc_info=True)


def log_database_operation(operation: str, duration: float = 0.0, **kwargs) -> None:
    """データベース操作ログ出力"""
    logger = get_context_logger(__name__, component="database")

    # durationが数値でない場合は0.0にフォールバック
    try:
        duration_float = float(duration)
    except (ValueError, TypeError):
        duration_float = 0.0

    log_data = {"operation": operation, "duration": duration_float, **kwargs}

    if duration_float > 1.0:
        logger.warning(f"Slow database operation: {operation}", extra=log_data)
    else:
        logger.debug(f"Database operation: {operation}", extra=log_data)


def log_business_event(event: str, details: Dict[str, Any] = None) -> None:
    """ビジネスイベントログ出力"""
    logger = get_context_logger(__name__, component="business")

    log_data = {"event": event, "details": details or {}}

    logger.info(f"Business event: {event}", extra=log_data)


def get_performance_logger(name: str = None) -> logging.Logger:
    """パフォーマンス測定用ロガーを取得"""
    logger_name = f"{name}.performance" if name else "performance"
    return get_context_logger(logger_name, component="performance")


def log_api_call(
    endpoint: str,
    method: str = "GET",
    duration: float = 0.0,
    status_code: int = None,
    **kwargs,
) -> None:
    """API呼び出しログ出力"""
    logger = get_context_logger(__name__, component="api")

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "duration": duration,
        "status_code": status_code,
        **kwargs,
    }

    if status_code and status_code >= 400:
        logger.error(f"API call failed: {method} {endpoint}", extra=log_data)
    elif duration > 2.0:
        logger.warning(f"Slow API call: {method} {endpoint}", extra=log_data)
    else:
        logger.debug(f"API call: {method} {endpoint}", extra=log_data)


def log_performance_metric(
    metric_name: str, value: float, unit: str = "", **kwargs
) -> None:
    """パフォーマンスメトリクスログ出力"""
    logger = get_performance_logger(__name__)

    log_data = {"metric_name": metric_name, "value": value, "unit": unit, **kwargs}

    logger.info(f"Performance metric: {metric_name}={value}{unit}", extra=log_data)

