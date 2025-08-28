#!/usr/bin/env python3
"""
統一ログシステム

全システム統一のログ出力、構造化ログ、ログ集約機能を提供します。
"""

import asyncio
import json
import logging
import logging.handlers
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import traceback
import sys
import os

from .base import BaseComponent, BaseConfig, HealthStatus, SystemStatus
from .unified_system_error import ErrorSeverity

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """ログレベル"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class LogFormat(Enum):
    """ログフォーマット"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogEntry:
    """ログエントリ"""
    timestamp: str
    level: str
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, str]] = None


@dataclass
class LoggingConfig:
    """ログ設定"""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    output_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_enabled: bool = True
    file_enabled: bool = True
    remote_enabled: bool = False
    remote_endpoint: Optional[str] = None
    buffer_size: int = 1000
    flush_interval: float = 5.0
    include_traceback: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: ['password', 'token', 'secret', 'key'])


class LogContext:
    """ログコンテキスト（スレッドローカル）"""
    _local = threading.local()
    
    @classmethod
    def set_request_id(cls, request_id: str):
        """リクエストID設定"""
        cls._local.request_id = request_id
    
    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """リクエストID取得"""
        return getattr(cls._local, 'request_id', None)
    
    @classmethod
    def set_user_id(cls, user_id: str):
        """ユーザーID設定"""
        cls._local.user_id = user_id
    
    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """ユーザーID取得"""
        return getattr(cls._local, 'user_id', None)
    
    @classmethod
    def set_session_id(cls, session_id: str):
        """セッションID設定"""
        cls._local.session_id = session_id
    
    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """セッションID取得"""
        return getattr(cls._local, 'session_id', None)
    
    @classmethod
    def set_component(cls, component: str):
        """コンポーネント設定"""
        cls._local.component = component
    
    @classmethod
    def get_component(cls) -> Optional[str]:
        """コンポーネント取得"""
        return getattr(cls._local, 'component', None)
    
    @classmethod
    def set_operation(cls, operation: str):
        """オペレーション設定"""
        cls._local.operation = operation
    
    @classmethod
    def get_operation(cls) -> Optional[str]:
        """オペレーション取得"""
        return getattr(cls._local, 'operation', None)
    
    @classmethod
    def clear(cls):
        """コンテキストクリア"""
        for attr in ['request_id', 'user_id', 'session_id', 'component', 'operation']:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)


class StructuredLogFormatter(logging.Formatter):
    """構造化ログフォーマッター"""
    
    def __init__(self, config: LoggingConfig):
        super().__init__()
        self.config = config
    
    def format(self, record: logging.LogRecord) -> str:
        """ログレコードフォーマット"""
        try:
            # 基本ログエントリ作成
            log_entry = self._create_log_entry(record)
            
            # フォーマット別出力
            if self.config.format == LogFormat.JSON:
                return self._format_json(log_entry)
            elif self.config.format == LogFormat.STRUCTURED:
                return self._format_structured(log_entry)
            else:
                return self._format_text(log_entry)
                
        except Exception as e:
            # フォーマット失敗時はフォールバック
            return f"LOG_FORMAT_ERROR: {record.getMessage()} | Format Error: {e}"
    
    def _create_log_entry(self, record: logging.LogRecord) -> LogEntry:
        """ログエントリ作成"""
        # タイムスタンプ
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp = dt.isoformat()
        
        # 例外情報
        exception_info = None
        if record.exc_info and self.config.include_traceback:
            exc_type, exc_value, exc_traceback = record.exc_info
            exception_info = {
                'type': exc_type.__name__ if exc_type else None,
                'message': str(exc_value) if exc_value else None,
                'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            }
        
        # 追加データ
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                extra_data[key] = self._sanitize_value(key, value)
        
        return LogEntry(
            timestamp=timestamp,
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            request_id=LogContext.get_request_id(),
            user_id=LogContext.get_user_id(),
            session_id=LogContext.get_session_id(),
            component=LogContext.get_component(),
            operation=LogContext.get_operation(),
            duration_ms=extra_data.pop('duration_ms', None),
            tags=extra_data.pop('tags', {}),
            extra_data=extra_data,
            exception_info=exception_info
        )
    
    def _sanitize_value(self, key: str, value: Any) -> Any:
        """値サニタイズ"""
        # 機密情報マスク
        if any(sensitive in key.lower() for sensitive in self.config.sensitive_fields):
            return "[MASKED]"
        
        # 複雑なオブジェクトの文字列化
        if isinstance(value, (dict, list)):
            try:
                return json.loads(json.dumps(value, default=str))
            except:
                return str(value)
        
        return value
    
    def _format_json(self, log_entry: LogEntry) -> str:
        """JSON形式フォーマット"""
        try:
            # None値を除外
            data = {k: v for k, v in asdict(log_entry).items() if v is not None}
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except Exception:
            return json.dumps({
                'timestamp': log_entry.timestamp,
                'level': log_entry.level,
                'message': log_entry.message,
                'error': 'JSON_SERIALIZATION_FAILED'
            })
    
    def _format_structured(self, log_entry: LogEntry) -> str:
        """構造化形式フォーマット"""
        parts = [
            f"[{log_entry.timestamp}]",
            f"[{log_entry.level}]",
            f"[{log_entry.component or log_entry.logger_name}]"
        ]
        
        if log_entry.request_id:
            parts.append(f"[req:{log_entry.request_id[:8]}]")
        
        if log_entry.operation:
            parts.append(f"[op:{log_entry.operation}]")
        
        if log_entry.duration_ms:
            parts.append(f"[{log_entry.duration_ms:.2f}ms]")
        
        parts.append(log_entry.message)
        
        result = " ".join(parts)
        
        # 例外情報追加
        if log_entry.exception_info:
            result += f"\n{log_entry.exception_info['traceback']}"
        
        return result
    
    def _format_text(self, log_entry: LogEntry) -> str:
        """テキスト形式フォーマット"""
        return (f"{log_entry.timestamp} - {log_entry.level} - "
                f"{log_entry.logger_name} - {log_entry.message}")


class LogBuffer:
    """ログバッファ"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[LogEntry] = []
        self.lock = threading.RLock()
    
    def add(self, log_entry: LogEntry):
        """ログ追加"""
        with self.lock:
            self.buffer.append(log_entry)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)  # 古いログを削除
    
    def flush(self) -> List[LogEntry]:
        """バッファフラッシュ"""
        with self.lock:
            entries = self.buffer.copy()
            self.buffer.clear()
            return entries
    
    def size(self) -> int:
        """バッファサイズ"""
        with self.lock:
            return len(self.buffer)


class RemoteLogHandler(logging.Handler):
    """リモートログハンドラー"""
    
    def __init__(self, endpoint: str, buffer_size: int = 100, flush_interval: float = 5.0):
        super().__init__()
        self.endpoint = endpoint
        self.buffer = LogBuffer(buffer_size)
        self.flush_interval = flush_interval
        self.flush_task = None
        self._running = False
    
    def start(self):
        """ハンドラー開始"""
        if not self._running:
            self._running = True
            self.flush_task = asyncio.create_task(self._flush_periodically())
    
    def stop(self):
        """ハンドラー停止"""
        self._running = False
        if self.flush_task:
            self.flush_task.cancel()
    
    def emit(self, record: logging.LogRecord):
        """ログ送出"""
        try:
            formatter = self.formatter
            if formatter:
                formatted = formatter.format(record)
                # ここでは簡略化のため、バッファに追加のみ
                # 実際の実装では、ログエントリを作成してバッファに追加
                pass
        except Exception:
            self.handleError(record)
    
    async def _flush_periodically(self):
        """定期フラッシュ"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                entries = self.buffer.flush()
                if entries:
                    await self._send_logs(entries)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # ログ送信失敗は内部ログに記録
                print(f"Remote log flush failed: {e}")
    
    async def _send_logs(self, entries: List[LogEntry]):
        """ログ送信"""
        # 実際の実装では HTTP POST 等でリモートエンドポイントに送信
        pass


class UnifiedLoggingSystem(BaseComponent):
    """統一ログシステム"""
    
    def __init__(self, name: str = "unified_logging_system"):
        super().__init__(name, BaseConfig(name=name))
        self.config = LoggingConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        self.remote_handler: Optional[RemoteLogHandler] = None
        self._lock = threading.RLock()
    
    async def start(self) -> bool:
        """ログシステム開始"""
        try:
            await self._setup_logging()
            
            if self.remote_handler:
                self.remote_handler.start()
            
            self.status = SystemStatus.RUNNING
            logger.info("Unified Logging System started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Unified Logging System: {e}")
            return False
    
    async def stop(self) -> bool:
        """ログシステム停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            if self.remote_handler:
                self.remote_handler.stop()
            
            # ハンドラークリーンアップ
            for handler in self.handlers:
                handler.close()
            
            self.status = SystemStatus.STOPPED
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Unified Logging System: {e}")
            return False
    
    def configure(self, config: LoggingConfig):
        """ログ設定"""
        self.config = config
        if self.status == SystemStatus.RUNNING:
            # 再設定
            asyncio.create_task(self._setup_logging())
    
    async def _setup_logging(self):
        """ログ設定"""
        with self._lock:
            # 既存ハンドラークリア
            for handler in self.handlers:
                handler.close()
            self.handlers.clear()
            
            # フォーマッター作成
            formatter = StructuredLogFormatter(self.config)
            
            # コンソールハンドラー
            if self.config.console_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                console_handler.setLevel(getattr(logging, self.config.level.value))
                self.handlers.append(console_handler)
            
            # ファイルハンドラー
            if self.config.file_enabled and self.config.output_file:
                # ディレクトリ作成
                log_path = Path(self.config.output_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    self.config.output_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(getattr(logging, self.config.level.value))
                self.handlers.append(file_handler)
            
            # リモートハンドラー
            if self.config.remote_enabled and self.config.remote_endpoint:
                self.remote_handler = RemoteLogHandler(
                    self.config.remote_endpoint,
                    self.config.buffer_size,
                    self.config.flush_interval
                )
                self.remote_handler.setFormatter(formatter)
                self.remote_handler.setLevel(getattr(logging, self.config.level.value))
                self.handlers.append(self.remote_handler)
            
            # ルートロガー設定
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            for handler in self.handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(getattr(logging, self.config.level.value))
    
    def get_logger(self, name: str) -> logging.Logger:
        """ロガー取得"""
        with self._lock:
            if name not in self.loggers:
                logger_instance = logging.getLogger(name)
                self.loggers[name] = logger_instance
            return self.loggers[name]
    
    def log_performance(self, operation: str, duration_ms: float, 
                       component: str = None, **kwargs):
        """パフォーマンスログ"""
        perf_logger = self.get_logger('performance')
        
        # コンテキスト設定
        if component:
            LogContext.set_component(component)
        LogContext.set_operation(operation)
        
        try:
            perf_logger.info(
                f"Performance: {operation} completed in {duration_ms:.2f}ms",
                extra={'duration_ms': duration_ms, 'tags': {'type': 'performance'}, **kwargs}
            )
        finally:
            LogContext.clear()
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          details: Dict[str, Any] = None):
        """セキュリティイベントログ"""
        security_logger = self.get_logger('security')
        
        if user_id:
            LogContext.set_user_id(user_id)
        LogContext.set_component('security')
        LogContext.set_operation(event_type)
        
        try:
            security_logger.warning(
                f"Security event: {event_type}",
                extra={'tags': {'type': 'security'}, 'event_details': details or {}}
            )
        finally:
            LogContext.clear()
    
    def log_business_event(self, event_type: str, component: str, 
                          data: Dict[str, Any] = None):
        """ビジネスイベントログ"""
        business_logger = self.get_logger('business')
        
        LogContext.set_component(component)
        LogContext.set_operation(event_type)
        
        try:
            business_logger.info(
                f"Business event: {event_type}",
                extra={'tags': {'type': 'business'}, 'event_data': data or {}}
            )
        finally:
            LogContext.clear()
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # ハンドラー状態チェック
            for handler in self.handlers:
                if hasattr(handler, 'stream') and handler.stream.closed:
                    return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY


# ユーティリティ関数
def setup_component_logging(component_name: str, logging_system: UnifiedLoggingSystem):
    """コンポーネントログ設定"""
    LogContext.set_component(component_name)
    return logging_system.get_logger(component_name)


def log_execution_time(func: Callable):
    """実行時間ログデコレータ"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            logging_system = get_unified_logging_system()
            logging_system.log_performance(
                operation=func.__name__,
                duration_ms=duration_ms,
                component=func.__module__
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logging_system = get_unified_logging_system()
            logging_system.log_performance(
                operation=func.__name__,
                duration_ms=duration_ms,
                component=func.__module__,
                error=str(e)
            )
            raise
    return wrapper


# グローバルログシステム
_global_logging_system: Optional[UnifiedLoggingSystem] = None


def get_unified_logging_system() -> UnifiedLoggingSystem:
    """統一ログシステム取得"""
    global _global_logging_system
    if _global_logging_system is None:
        _global_logging_system = UnifiedLoggingSystem()
    return _global_logging_system


async def initialize_unified_logging(config: LoggingConfig = None):
    """統一ログシステム初期化"""
    logging_system = get_unified_logging_system()
    
    if config:
        logging_system.configure(config)
    
    await logging_system.start()
    return logging_system