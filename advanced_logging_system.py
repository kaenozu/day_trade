#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Logging System - 高度ログ・デバッグシステム
Issue #947対応: 統一ログ + 構造化ログ + デバッグ機能
"""

import logging
import logging.handlers
import json
import sys
import os
import time
import threading
import traceback
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import contextlib
import functools

# 高度なログ機能（オプショナル）
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

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


class LogCategory(Enum):
    """ログカテゴリ"""
    SYSTEM = "SYSTEM"
    AI_ANALYSIS = "AI_ANALYSIS"
    TRADING = "TRADING"
    DATABASE = "DATABASE"
    API = "API"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    USER_ACTION = "USER_ACTION"


@dataclass
class LogContext:
    """ログコンテキスト"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    function: Optional[str] = None
    trace_id: Optional[str] = None
    additional_data: Dict[str, Any] = None


@dataclass
class LogEvent:
    """ログイベント"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext
    stack_trace: Optional[str]
    execution_time: Optional[float]
    memory_usage: Optional[float]
    metadata: Dict[str, Any]


class StructuredFormatter(logging.Formatter):
    """構造化ログフォーマッター"""

    def format(self, record):
        """ログレコードフォーマット"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # 追加属性
        if hasattr(record, 'category'):
            log_data['category'] = record.category
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time

        # 例外情報
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """カラーログフォーマッター"""

    def __init__(self):
        super().__init__()

        if HAS_COLORLOG:
            self.formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                reset=True,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            self.formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def format(self, record):
        return self.formatter.format(record)


class LogAnalyzer:
    """ログ分析システム"""

    def __init__(self):
        self.log_buffer: deque = deque(maxlen=10000)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.performance_metrics: List[float] = []

    def analyze_logs(self, log_events: List[LogEvent]) -> Dict[str, Any]:
        """ログ分析"""
        if not log_events:
            return {}

        # 基本統計
        total_logs = len(log_events)
        level_distribution = defaultdict(int)
        category_distribution = defaultdict(int)
        component_distribution = defaultdict(int)

        error_count = 0
        warning_count = 0

        for event in log_events:
            level_distribution[event.level.name] += 1
            category_distribution[event.category.name] += 1

            if event.context.component:
                component_distribution[event.context.component] += 1

            if event.level == LogLevel.ERROR:
                error_count += 1
            elif event.level == LogLevel.WARNING:
                warning_count += 1

        # パフォーマンス分析
        execution_times = [e.execution_time for e in log_events if e.execution_time]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        return {
            'total_logs': total_logs,
            'error_count': error_count,
            'warning_count': warning_count,
            'error_rate': error_count / total_logs if total_logs > 0 else 0,
            'level_distribution': dict(level_distribution),
            'category_distribution': dict(category_distribution),
            'component_distribution': dict(component_distribution),
            'avg_execution_time': avg_execution_time,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def detect_anomalies(self, log_events: List[LogEvent]) -> List[Dict[str, Any]]:
        """異常検出"""
        anomalies = []

        # 高頻度エラー
        error_messages = defaultdict(int)
        for event in log_events:
            if event.level == LogLevel.ERROR:
                error_messages[event.message] += 1

        for message, count in error_messages.items():
            if count > 5:  # 5回以上の同じエラー
                anomalies.append({
                    'type': 'HIGH_FREQUENCY_ERROR',
                    'message': message,
                    'count': count,
                    'severity': 'HIGH'
                })

        # 長時間実行
        long_executions = [e for e in log_events if e.execution_time and e.execution_time > 5.0]
        if long_executions:
            anomalies.append({
                'type': 'LONG_EXECUTION_TIME',
                'count': len(long_executions),
                'max_time': max(e.execution_time for e in long_executions),
                'severity': 'MEDIUM'
            })

        return anomalies


class DebugProfiler:
    """デバッグプロファイラ"""

    def __init__(self):
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.call_stack: List[Dict[str, Any]] = []

    def start_profiling(self, function_name: str, context: Dict[str, Any] = None):
        """プロファイリング開始"""
        profile_id = f"{function_name}_{int(time.time() * 1000)}"

        self.active_profiles[profile_id] = {
            'function_name': function_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'context': context or {},
            'call_count': 0,
            'nested_calls': []
        }

        return profile_id

    def end_profiling(self, profile_id: str) -> Dict[str, Any]:
        """プロファイリング終了"""
        if profile_id not in self.active_profiles:
            return {}

        profile = self.active_profiles[profile_id]
        end_time = time.time()
        end_memory = self._get_memory_usage()

        result = {
            'profile_id': profile_id,
            'function_name': profile['function_name'],
            'execution_time': end_time - profile['start_time'],
            'memory_delta': end_memory - profile['start_memory'],
            'call_count': profile['call_count'],
            'nested_calls': profile['nested_calls'],
            'context': profile['context']
        }

        del self.active_profiles[profile_id]
        return result

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def profile_function(self, func: Callable) -> Callable:
        """関数プロファイリングデコレータ"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile_id = self.start_profiling(func.__name__)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profile_result = self.end_profiling(profile_id)
                if profile_result['execution_time'] > 0.1:  # 100ms以上の場合ログ
                    advanced_logger.log(
                        LogLevel.DEBUG,
                        f"Function {func.__name__} took {profile_result['execution_time']:.3f}s",
                        LogCategory.PERFORMANCE,
                        execution_time=profile_result['execution_time']
                    )

        return wrapper


class AdvancedLogger:
    """高度ログシステム"""

    def __init__(self, name: str = "DayTradePersonal"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # ログコンテキスト（スレッドローカル）
        self.context_local = threading.local()

        # 分析・デバッグ
        self.log_analyzer = LogAnalyzer()
        self.debug_profiler = DebugProfiler()

        # ログバッファ
        self.log_events: deque = deque(maxlen=5000)

        # 設定
        self.structured_logging = True
        self.color_logging = True
        self.file_logging = True
        self.console_logging = True

        # ハンドラー設定
        self._setup_handlers()

        # 統計
        self.log_counts = defaultdict(int)
        self.start_time = datetime.now()

    def _setup_handlers(self):
        """ログハンドラー設定"""
        # 既存ハンドラーを削除
        self.logger.handlers.clear()

        # コンソールハンドラー
        if self.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.color_logging:
                console_handler.setFormatter(ColoredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

        # ファイルハンドラー
        if self.file_logging:
            # 回転ファイルハンドラー
            file_handler = logging.handlers.RotatingFileHandler(
                'logs/advanced_system.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )

            if self.structured_logging:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )

            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        # エラー専用ファイル
        error_handler = logging.handlers.RotatingFileHandler(
            'logs/errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)

    def set_context(self, **kwargs):
        """ログコンテキスト設定"""
        if not hasattr(self.context_local, 'context'):
            self.context_local.context = LogContext()

        for key, value in kwargs.items():
            if hasattr(self.context_local.context, key):
                setattr(self.context_local.context, key, value)

    def get_context(self) -> LogContext:
        """現在のログコンテキスト取得"""
        if not hasattr(self.context_local, 'context'):
            self.context_local.context = LogContext()
        return self.context_local.context

    def log(self, level: LogLevel, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """統一ログ出力"""
        # ログイベント作成
        log_event = LogEvent(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            context=self.get_context(),
            stack_trace=kwargs.get('stack_trace'),
            execution_time=kwargs.get('execution_time'),
            memory_usage=kwargs.get('memory_usage'),
            metadata=kwargs
        )

        # ログバッファに追加
        self.log_events.append(log_event)

        # 統計更新
        self.log_counts[level.name] += 1

        # 標準ログ出力
        log_record = self.logger.makeRecord(
            name=self.name,
            level=level.value,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # 追加属性
        log_record.category = category.name
        if log_event.context.request_id:
            log_record.request_id = log_event.context.request_id
        if log_event.context.user_id:
            log_record.user_id = log_event.context.user_id
        if log_event.execution_time:
            log_record.execution_time = log_event.execution_time

        self.logger.handle(log_record)

    def trace(self, message: str, **kwargs):
        """トレースレベルログ"""
        self.log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """デバッグレベルログ"""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """情報レベルログ"""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告レベルログ"""
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exception: Exception = None, **kwargs):
        """エラーレベルログ"""
        if exception:
            kwargs['stack_trace'] = traceback.format_exc()
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """重要レベルログ"""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    @contextlib.contextmanager
    def log_context(self, **context_vars):
        """ログコンテキスト管理"""
        # 現在のコンテキストを保存
        old_context = self.get_context()

        # 新しいコンテキスト設定
        self.set_context(**context_vars)

        try:
            yield
        finally:
            # コンテキスト復元
            self.context_local.context = old_context

    def measure_time(self, operation: str, category: LogCategory = LogCategory.PERFORMANCE):
        """実行時間測定デコレータ"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    self.log(
                        LogLevel.INFO,
                        f"{operation} completed",
                        category,
                        execution_time=execution_time
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.error(
                        f"{operation} failed: {str(e)}",
                        exception=e,
                        execution_time=execution_time
                    )
                    raise

            return wrapper
        return decorator

    def get_log_statistics(self) -> Dict[str, Any]:
        """ログ統計取得"""
        uptime = datetime.now() - self.start_time

        # 最近のログ分析
        recent_events = [e for e in self.log_events if
                        (datetime.now() - e.timestamp).seconds < 3600]  # 1時間以内

        analysis = self.log_analyzer.analyze_logs(recent_events)
        anomalies = self.log_analyzer.detect_anomalies(recent_events)

        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'total_logs': sum(self.log_counts.values()),
            'log_counts_by_level': dict(self.log_counts),
            'recent_analysis': analysis,
            'detected_anomalies': anomalies,
            'buffer_size': len(self.log_events),
            'handlers_count': len(self.logger.handlers)
        }

    def export_logs(self, hours: int = 24, format: str = 'json') -> str:
        """ログエクスポート"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_events = [e for e in self.log_events if e.timestamp > cutoff_time]

        if format == 'json':
            return json.dumps([asdict(event) for event in filtered_events],
                            default=str, ensure_ascii=False, indent=2)
        elif format == 'csv':
            # CSV形式のエクスポート（簡略化）
            lines = ['timestamp,level,category,message,component']
            for event in filtered_events:
                line = f"{event.timestamp},{event.level.name},{event.category.name}," \
                      f"\"{event.message}\",{event.context.component or ''}"
                lines.append(line)
            return '\n'.join(lines)
        else:
            return str(filtered_events)


# グローバルインスタンス
advanced_logger = AdvancedLogger()


def get_logger(name: str = None) -> AdvancedLogger:
    """ログインスタンス取得"""
    return advanced_logger


def log_execution_time(operation: str):
    """実行時間ログデコレータ"""
    return advanced_logger.measure_time(operation)


async def test_advanced_logging():
    """高度ログシステムテスト"""
    print("=== Advanced Logging System Test ===")

    # ログコンテキスト設定
    advanced_logger.set_context(
        request_id="REQ123",
        user_id="USER456",
        component="test_module"
    )

    # 各レベルのログテスト
    advanced_logger.info("System started successfully", category=LogCategory.SYSTEM)
    advanced_logger.debug("Debug information", category=LogCategory.AI_ANALYSIS)
    advanced_logger.warning("This is a warning message", category=LogCategory.TRADING)

    # 実行時間測定テスト
    @log_execution_time("test_operation")
    def slow_operation():
        import time
        time.sleep(0.1)
        return "Operation completed"

    result = slow_operation()
    print(f"Operation result: {result}")

    # エラーログテスト
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        advanced_logger.error("Test error occurred", exception=e)

    # プロファイリングテスト
    @advanced_logger.debug_profiler.profile_function
    def cpu_intensive_task():
        total = 0
        for i in range(100000):
            total += i
        return total

    cpu_result = cpu_intensive_task()
    print(f"CPU task result: {cpu_result}")

    # 統計取得
    stats = advanced_logger.get_log_statistics()
    print(f"Total logs: {stats['total_logs']}")
    print(f"Log level distribution: {stats['log_counts_by_level']}")

    # 異常検出
    anomalies = stats.get('detected_anomalies', [])
    if anomalies:
        print(f"Detected {len(anomalies)} anomalies")

    print("Advanced logging test completed!")


if __name__ == "__main__":
    # ログディレクトリ作成
    os.makedirs('logs', exist_ok=True)

    import asyncio
    asyncio.run(test_advanced_logging())