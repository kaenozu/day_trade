"""
構造化ロギング設定モジュール

アプリケーション全体で使用する構造化ロギング機能を提供。
JSON形式での出力、フィルタリング、ログレベル管理を統一。
"""

import atexit
import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Any, Dict

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class ContextLogger:
    """コンテキスト付きロガー - Issue #625対応: フォールバック簡素化"""

    def __init__(self, logger, context: Dict[str, Any] = None):
        # Issue #624, #625対応: structlogロガーかどうかを正しく判定
        if STRUCTLOG_AVAILABLE and hasattr(logger, 'bind'):
            # structlogのBoundLoggerの場合
            self.logger = logger.bind(**(context or {}))
            self.is_structlog = True
        else:
            # 標準ロガーまたはstructlog未使用の場合
            self.logger = logger
            self.context = context or {}
            self.is_structlog = False

    def bind(self, **kwargs) -> "ContextLogger":
        """コンテキストを追加したロガーを作成"""
        if self.is_structlog:
            return ContextLogger(self.logger.bind(**kwargs))
        else:
            new_context = {**self.context, **kwargs}
            return ContextLogger(self.logger, new_context)

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """コンテキスト付きでログ出力 - Issue #625対応: 簡素化"""
        if self.is_structlog:
            # structlogの場合は直接ログ出力
            self.logger.log(level, msg, *args, **kwargs)
        else:
            # 標準ロガーの場合はextraにコンテキストを追加
            extra = kwargs.get("extra", {})
            extra.update(self.context)

            # Issue #625対応: 安全なキーワード引数フィルタリング
            safe_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in ["extra", "exc_info", "stack_info", "stacklevel"]
            }
            safe_kwargs["extra"] = extra

            self.logger.log(level, msg, *args, **safe_kwargs)

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
    """ロギング設定管理クラス - Issue #631対応: 外部設定ファイル拡張"""

    def __init__(self):
        self.is_configured = False
        self.log_level = self._get_log_level()
        self.log_format = self._get_log_format()
        self.config_file = self._get_config_file_path()
        
    def _get_log_level(self) -> str:
        """環境変数からログレベルを取得 - Issue #631対応"""
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        # 有効なログレベルかどうかチェック
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return level if level in valid_levels else 'INFO'

    def _get_log_format(self) -> str:
        """環境変数からログフォーマットを取得 - Issue #631対応"""
        format_type = os.getenv("LOG_FORMAT", "simple").lower()
        # 有効なフォーマットかどうかチェック
        valid_formats = ['simple', 'json', 'detailed']
        return format_type if format_type in valid_formats else 'simple'
        
    def _get_config_file_path(self) -> str:
        """設定ファイルパスを取得 - Issue #631対応"""
        return os.getenv("LOGGING_CONFIG_FILE", "")

    def configure_logging(self) -> None:
        """基本ロギングを設定"""
        if self.is_configured:
            return

        if STRUCTLOG_AVAILABLE:
            # Structlogの設定
            processors = [
                structlog.stdlib.add_logger_name,
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

            # Issue #632対応: Python版本依存structlog設定簡素化
            try:
                if hasattr(structlog.stdlib, "add_logger_oid"):
                    processors.insert(1, structlog.stdlib.add_logger_oid)
            except AttributeError:
                # Python版本の互換性問題がある場合はスキップ
                pass

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
            # 非同期ロギングのためにQueueHandlerを使用

            log_queue = Queue(-1)  # 無制限のキュー
            queue_handler = QueueHandler(log_queue)

            # 標準のStreamHandlerを設定
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(
                logging.Formatter("%(message)s")
            )  # structlogがフォーマットするため、ここではシンプルに

            # Issue #626対応: QueueHandler終了堅牢性向上
            queue_listener = QueueListener(log_queue, stream_handler)
            queue_listener.start()  # リスナーを開始

            # アプリケーション終了時の安全な停止処理
            def safe_queue_listener_stop():
                try:
                    queue_listener.stop()
                except Exception:
                    # 終了時のエラーは無視（デバッグ時以外）
                    pass
            
            atexit.register(safe_queue_listener_stop)

            logging.basicConfig(
                format="%(message)s", handlers=[queue_handler], level=self.log_level
            )
        else:
            # Structlogが利用できない場合のフォールバック
            # 非同期ロギングのためにQueueHandlerを使用
            log_queue = Queue(-1)  # 無制限のキュー
            queue_handler = QueueHandler(log_queue)

            # 標準のStreamHandlerを設定
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

            # Issue #626対応: QueueHandler終了堅牢性向上
            queue_listener = QueueListener(log_queue, stream_handler)
            queue_listener.start()  # リスナーを開始

            # アプリケーション終了時の安全な停止処理
            def safe_queue_listener_stop():
                try:
                    queue_listener.stop()
                except Exception:
                    # 終了時のエラーは無視（デバッグ時以外）
                    pass
            
            atexit.register(safe_queue_listener_stop)

            logging.basicConfig(
                level=getattr(logging, self.log_level, logging.INFO),
                handlers=[queue_handler],
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
    """コンテキスト付きロガーを取得 - Issue #627対応: 命名階層構造改善"""
    # Issue #627対応: 階層的ロガー名の正規化
    if component:
        logger_name = f"{name}.{component}"
    else:
        logger_name = name
    
    logger = get_logger(logger_name)
    return ContextLogger(logger, kwargs)


def log_error_with_context(error: Exception, context: Dict[str, Any], source_module: str = None):
    """
    コンテキスト付きでエラーをログ出力 - Issue #628対応

    Args:
        error: エラーオブジェクト
        context: コンテキスト情報
        source_module: エラー発生源のモジュール名（Noneの場合は呼び出し元を自動検出）
    """
    # Issue #628対応: エラーの実際の発生源を特定
    if source_module is None:
        # 呼び出し元のフレーム情報を取得
        import inspect
        frame = inspect.currentframe()
        try:
            # 呼び出し元のフレームを取得（2つ上のフレーム）
            caller_frame = frame.f_back.f_back if frame.f_back else frame.f_back
            if caller_frame:
                source_module = caller_frame.f_globals.get('__name__', __name__)
            else:
                source_module = __name__
        finally:
            del frame  # メモリリーク防止

    logger = logging.getLogger(source_module)

    # エラー情報を詳細化
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'source_module': source_module,
        'context': context
    }

    logger.error(f"Error in {source_module}: {type(error).__name__}: {error}. Context: {context}",
                extra=error_info, exc_info=True)


def _safe_duration_conversion(duration) -> float:
    """安全なduration型変換 - Issue #630対応"""
    try:
        return float(duration)
    except (ValueError, TypeError):
        return 0.0


def log_database_operation(operation: str, duration: float = 0.0, **kwargs) -> None:
    """データベース操作ログ出力 - Issue #629, #630対応"""
    logger = get_context_logger(__name__, component="database")

    # Issue #630対応: duration型ハンドリング改善
    duration_float = _safe_duration_conversion(duration)

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


def get_caller_info(skip_frames: int = 2) -> Dict[str, Any]:
    """
    呼び出し元の詳細情報を取得 - Issue #628対応

    Args:
        skip_frames: スキップするフレーム数（デフォルト2：この関数と直接の呼び出し元をスキップ）

    Returns:
        呼び出し元情報の辞書
    """
    import inspect

    try:
        # フレームスタックを取得
        stack = inspect.stack()

        if len(stack) > skip_frames:
            caller_frame = stack[skip_frames]
            return {
                'module_name': caller_frame.frame.f_globals.get('__name__', 'unknown'),
                'function_name': caller_frame.function,
                'filename': caller_frame.filename,
                'line_number': caller_frame.lineno,
                'code_context': caller_frame.code_context[0].strip() if caller_frame.code_context else None
            }
        else:
            return {
                'module_name': 'unknown',
                'function_name': 'unknown',
                'filename': 'unknown',
                'line_number': 0,
                'code_context': None
            }
    except Exception as e:
        # フレーム取得に失敗した場合のフォールバック
        return {
            'module_name': 'error_getting_caller_info',
            'function_name': 'unknown',
            'filename': 'unknown',
            'line_number': 0,
            'code_context': None,
            'caller_info_error': str(e)
        }


def log_error_with_enhanced_context(error: Exception, context: Dict[str, Any] = None, include_caller_info: bool = True):
    """
    拡張コンテキスト付きエラーログ出力 - Issue #628対応

    Args:
        error: エラーオブジェクト
        context: 追加のコンテキスト情報
        include_caller_info: 呼び出し元情報を含めるかどうか
    """
    context = context or {}

    # 呼び出し元情報を取得
    if include_caller_info:
        caller_info = get_caller_info(skip_frames=2)
        source_module = caller_info['module_name']
    else:
        caller_info = {}
        source_module = __name__

    logger = logging.getLogger(source_module)

    # 拡張エラー情報を構築（LogRecordと重複するキーを避ける）
    enhanced_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'source_module': source_module,
        'context': context,
        'caller_module': caller_info.get('module_name'),
        'caller_function': caller_info.get('function_name'),
        'caller_file': caller_info.get('filename'),  # filenameではなくcaller_fileを使用
        'caller_line': caller_info.get('line_number'),
        'caller_context': caller_info.get('code_context')
    }

    # ログメッセージを構築
    location_info = f"{caller_info.get('filename', 'unknown')}:{caller_info.get('line_number', 0)}" if include_caller_info else source_module
    message = f"Error in {location_info} [{caller_info.get('function_name', 'unknown')}]: {type(error).__name__}: {error}"

    logger.error(message, extra=enhanced_info, exc_info=True)
