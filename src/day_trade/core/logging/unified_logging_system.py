"""
統一ロギングシステム

システム全体で一貫したログ管理を提供
"""

import logging
import logging.handlers
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import threading
import queue
import time
import traceback
import sys
import os
from contextlib import contextmanager

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class LogLevel(Enum):
    """ログレベル"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """ログフォーマット"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogEntry:
    """ログエントリ"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None


class LogDestination(ABC):
    """ログ出力先基底クラス"""

    @abstractmethod
    def write_log(self, entry: LogEntry) -> bool:
        """ログ出力"""
        pass

    @abstractmethod
    def close(self) -> None:
        """出力先クローズ"""
        pass


class ConsoleLogDestination(LogDestination):
    """コンソールログ出力先"""

    def __init__(self, format_type: LogFormat = LogFormat.TEXT):
        self.format_type = format_type

    def write_log(self, entry: LogEntry) -> bool:
        """コンソールにログ出力"""
        try:
            if self.format_type == LogFormat.JSON:
                log_data = {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.name,
                    "logger": entry.logger_name,
                    "message": entry.message,
                    "component": entry.component,
                    "operation": entry.operation
                }
                if entry.extra_data:
                    log_data.update(entry.extra_data)

                print(json.dumps(log_data, ensure_ascii=False))
            else:
                # テキスト形式
                log_line = (
                    f"{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[{entry.level.name}] "
                    f"{entry.logger_name} - {entry.message}"
                )

                if entry.component:
                    log_line += f" [component={entry.component}]"
                if entry.operation:
                    log_line += f" [operation={entry.operation}]"

                print(log_line)

            return True
        except Exception:
            return False

    def close(self) -> None:
        """コンソール出力先クローズ"""
        pass


class FileLogDestination(LogDestination):
    """ファイルログ出力先"""

    def __init__(
        self,
        file_path: str,
        format_type: LogFormat = LogFormat.TEXT,
        max_size_mb: int = 100,
        backup_count: int = 5,
        encoding: str = "utf-8"
    ):
        self.file_path = Path(file_path)
        self.format_type = format_type
        self.encoding = encoding

        # ディレクトリ作成
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # ローテーションハンドラー設定
        self.handler = logging.handlers.RotatingFileHandler(
            filename=str(self.file_path),
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding=encoding
        )

        self._lock = threading.Lock()

    def write_log(self, entry: LogEntry) -> bool:
        """ファイルにログ出力"""
        try:
            with self._lock:
                if self.format_type == LogFormat.JSON:
                    log_data = {
                        "timestamp": entry.timestamp.isoformat(),
                        "level": entry.level.name,
                        "logger": entry.logger_name,
                        "message": entry.message,
                        "component": entry.component,
                        "operation": entry.operation,
                        "user_id": entry.user_id,
                        "session_id": entry.session_id,
                        "correlation_id": entry.correlation_id
                    }

                    if entry.extra_data:
                        log_data["extra"] = entry.extra_data

                    if entry.exception_info:
                        log_data["exception"] = entry.exception_info

                    log_line = json.dumps(log_data, ensure_ascii=False) + "\n"
                else:
                    # テキスト形式
                    log_line = (
                        f"{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"[{entry.level.name}] "
                        f"{entry.logger_name} - {entry.message}"
                    )

                    if entry.component:
                        log_line += f" [component={entry.component}]"
                    if entry.operation:
                        log_line += f" [operation={entry.operation}]"
                    if entry.correlation_id:
                        log_line += f" [correlation_id={entry.correlation_id}]"

                    if entry.exception_info:
                        log_line += f"\n{entry.exception_info}"

                    log_line += "\n"

                # ファイル書き込み
                with open(self.file_path, 'a', encoding=self.encoding) as f:
                    f.write(log_line)

                return True
        except Exception:
            return False

    def close(self) -> None:
        """ファイル出力先クローズ"""
        self.handler.close()


class DatabaseLogDestination(LogDestination):
    """データベースログ出力先"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connection = None
        self._lock = threading.Lock()

    def write_log(self, entry: LogEntry) -> bool:
        """データベースにログ出力"""
        # データベース実装は実際のDBライブラリに依存
        # ここでは仮実装
        return True

    def close(self) -> None:
        """データベース接続クローズ"""
        if self._connection:
            self._connection.close()


class LogFilter:
    """ログフィルター"""

    def __init__(self):
        self._filters: List[Callable[[LogEntry], bool]] = []

    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """フィルター追加"""
        self._filters.append(filter_func)

    def should_log(self, entry: LogEntry) -> bool:
        """ログ出力判定"""
        for filter_func in self._filters:
            if not filter_func(entry):
                return False
        return True


class LogAggregator:
    """ログ集約"""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._log_counts: Dict[str, Dict[str, int]] = {}
        self._last_reset = time.time()
        self._lock = threading.Lock()

    def should_suppress(self, entry: LogEntry) -> bool:
        """ログ抑制判定"""
        current_time = time.time()

        with self._lock:
            # ウィンドウリセット
            if current_time - self._last_reset > self.window_seconds:
                self._log_counts.clear()
                self._last_reset = current_time

            # カウント更新
            key = f"{entry.logger_name}:{entry.level.name}:{entry.message}"
            if key not in self._log_counts:
                self._log_counts[key] = {"count": 0, "first_seen": current_time}

            self._log_counts[key]["count"] += 1

            # 抑制判定（同じログが10回以上）
            return self._log_counts[key]["count"] > 10


class AsyncLogProcessor:
    """非同期ログ処理"""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._destinations: List[LogDestination] = []
        self._filter = LogFilter()
        self._aggregator = LogAggregator()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._started = False

    def add_destination(self, destination: LogDestination) -> None:
        """出力先追加"""
        self._destinations.append(destination)

    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """フィルター追加"""
        self._filter.add_filter(filter_func)

    def start(self) -> None:
        """非同期処理開始"""
        if self._started:
            return

        self._started = True
        self._worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """非同期処理停止"""
        if not self._started:
            return

        self._shutdown_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # 残りのログを処理
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                self._write_to_destinations(entry)
            except queue.Empty:
                break

        # 出力先クローズ
        for destination in self._destinations:
            destination.close()

        self._started = False

    def enqueue_log(self, entry: LogEntry) -> bool:
        """ログエンキュー"""
        if not self._started:
            self.start()

        try:
            self._queue.put_nowait(entry)
            return True
        except queue.Full:
            # キューフルの場合、古いエントリを削除して追加
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(entry)
                return True
            except (queue.Empty, queue.Full):
                return False

    def _process_logs(self) -> None:
        """ログ処理ワーカー"""
        while not self._shutdown_event.is_set():
            try:
                entry = self._queue.get(timeout=1.0)

                # フィルター適用
                if not self._filter.should_log(entry):
                    continue

                # 集約チェック
                if self._aggregator.should_suppress(entry):
                    continue

                # 出力先に書き込み
                self._write_to_destinations(entry)

            except queue.Empty:
                continue
            except Exception as e:
                # エラーログの無限ループを避けるため、stderrに出力
                print(f"Log processing error: {e}", file=sys.stderr)

    def _write_to_destinations(self, entry: LogEntry) -> None:
        """出力先書き込み"""
        for destination in self._destinations:
            try:
                destination.write_log(entry)
            except Exception as e:
                print(f"Log destination error: {e}", file=sys.stderr)


class UnifiedLogger:
    """統一ログガー"""

    def __init__(self, name: str, component: str = ""):
        self.name = name
        self.component = component
        self._processor: Optional[AsyncLogProcessor] = None
        self._context_data: Dict[str, Any] = {}

    def set_processor(self, processor: AsyncLogProcessor) -> None:
        """プロセッサー設定"""
        self._processor = processor

    def set_context(self, **context_data) -> None:
        """コンテキストデータ設定"""
        self._context_data.update(context_data)

    def clear_context(self) -> None:
        """コンテキストデータクリア"""
        self._context_data.clear()

    @contextmanager
    def context(self, **context_data):
        """一時コンテキスト"""
        old_context = self._context_data.copy()
        try:
            self._context_data.update(context_data)
            yield
        finally:
            self._context_data = old_context

    def _log(
        self,
        level: LogLevel,
        message: str,
        operation: str = "",
        exception: Optional[Exception] = None,
        **extra_data
    ) -> None:
        """ログ出力"""
        if not self._processor:
            return

        # 例外情報取得
        exception_info = None
        if exception:
            exception_info = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))

        # ログエントリ作成
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            logger_name=self.name,
            message=message,
            component=self.component,
            operation=operation,
            extra_data={**self._context_data, **extra_data},
            exception_info=exception_info
        )

        # コンテキストからユーザーID等を設定
        entry.user_id = self._context_data.get("user_id")
        entry.session_id = self._context_data.get("session_id")
        entry.correlation_id = self._context_data.get("correlation_id")

        self._processor.enqueue_log(entry)

    def trace(self, message: str, **kwargs) -> None:
        """TRACEログ"""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """DEBUGログ"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """INFOログ"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """WARNINGログ"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """ERRORログ"""
        self._log(LogLevel.ERROR, message, exception=exception, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """CRITICALログ"""
        self._log(LogLevel.CRITICAL, message, exception=exception, **kwargs)


class UnifiedLoggingSystem:
    """統一ロギングシステム"""

    def __init__(self):
        self._processor = AsyncLogProcessor()
        self._loggers: Dict[str, UnifiedLogger] = {}
        self._global_context: Dict[str, Any] = {}

        # デフォルト出力先追加
        self._processor.add_destination(ConsoleLogDestination())

    def configure(
        self,
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.TEXT,
        max_file_size_mb: int = 100,
        backup_count: int = 5
    ) -> None:
        """ロギングシステム設定"""
        # ファイル出力追加
        if log_file:
            file_destination = FileLogDestination(
                file_path=log_file,
                format_type=format_type,
                max_size_mb=max_file_size_mb,
                backup_count=backup_count
            )
            self._processor.add_destination(file_destination)

        # レベルフィルター追加
        self._processor.add_filter(lambda entry: entry.level.value >= log_level.value)

    def get_logger(self, name: str, component: str = "") -> UnifiedLogger:
        """ロガー取得"""
        if name not in self._loggers:
            logger = UnifiedLogger(name, component)
            logger.set_processor(self._processor)
            logger.set_context(**self._global_context)
            self._loggers[name] = logger

        return self._loggers[name]

    def set_global_context(self, **context_data) -> None:
        """グローバルコンテキスト設定"""
        self._global_context.update(context_data)

        # 既存ロガーにも適用
        for logger in self._loggers.values():
            logger.set_context(**context_data)

    def start(self) -> None:
        """ロギングシステム開始"""
        self._processor.start()

    def stop(self) -> None:
        """ロギングシステム停止"""
        self._processor.stop()

    def add_destination(self, destination: LogDestination) -> None:
        """出力先追加"""
        self._processor.add_destination(destination)

    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """フィルター追加"""
        self._processor.add_filter(filter_func)


# グローバルロギングシステム
global_logging_system = UnifiedLoggingSystem()


def get_logger(name: str, component: str = "") -> UnifiedLogger:
    """ロガー取得関数"""
    return global_logging_system.get_logger(name, component)


def configure_logging(
    log_file: str = "logs/day_trade.log",
    log_level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.TEXT
) -> None:
    """ロギング設定関数"""
    global_logging_system.configure(
        log_file=log_file,
        log_level=log_level,
        format_type=format_type
    )
    global_logging_system.start()


# 便利なデコレーター
def log_execution(
    logger_name: str = None,
    component: str = "",
    operation: str = "",
    log_args: bool = False,
    log_result: bool = False
):
    """実行ログデコレーター"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__, component)
            operation_name = operation or func.__name__

            start_time = time.time()

            try:
                logger.info(f"Starting {operation_name}", operation=operation_name)

                if log_args:
                    logger.debug(f"Arguments: args={args}, kwargs={kwargs}", operation=operation_name)

                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                logger.info(
                    f"Completed {operation_name} in {execution_time:.3f}s",
                    operation=operation_name,
                    execution_time=execution_time
                )

                if log_result:
                    logger.debug(f"Result: {result}", operation=operation_name)

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {operation_name} after {execution_time:.3f}s",
                    operation=operation_name,
                    exception=e,
                    execution_time=execution_time
                )
                raise

        return wrapper
    return decorator