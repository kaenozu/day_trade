#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張ログシステム

パフォーマンス最適化されたログシステムと構造化ログ出力
"""

import logging
import logging.handlers
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import queue
import atexit


@dataclass
class LogMetrics:
    """ログメトリクス"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    thread_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class AsyncLogHandler(logging.Handler):
    """
    非同期ログハンドラー

    メインスレッドの処理をブロックしないログ出力
    """

    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.start_worker()

    def start_worker(self):
        """ワーカースレッド開始"""
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="AsyncLogWorker"
        )
        self.worker_thread.start()

    def _worker(self):
        """ログ処理ワーカー"""
        while not self.shutdown_event.is_set():
            try:
                # タイムアウト付きでキューから取得
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # シャットダウンシグナル
                    break

                self.target_handler.emit(record)
                self.log_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                # ログハンドラー内のエラーは標準エラーに出力
                print(f"AsyncLogHandler error: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """ログレコードをキューに追加"""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # キューが満杯の場合は古いレコードを破棄
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(record)
            except queue.Empty:
                pass

    def shutdown(self):
        """ハンドラーをシャットダウン"""
        self.shutdown_event.set()

        # キューの残りを処理
        try:
            while not self.log_queue.empty():
                record = self.log_queue.get_nowait()
                self.target_handler.emit(record)
        except queue.Empty:
            pass

        # ワーカースレッド終了を待機
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_queue.put(None)  # シャットダウンシグナル
            self.worker_thread.join(timeout=5.0)


class StructuredFormatter(logging.Formatter):
    """
    構造化ログフォーマッター

    JSON形式で構造化されたログを出力
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        # 例外情報があれば追加
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # カスタム属性があれば追加
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class PerformanceLogFilter(logging.Filter):
    """
    パフォーマンスログフィルター

    ログレベルや頻度に基づいてフィルタリング
    """

    def __init__(self,
                 min_level: int = logging.INFO,
                 rate_limit: Optional[float] = None):
        super().__init__()
        self.min_level = min_level
        self.rate_limit = rate_limit
        self.last_log_time = {}

    def filter(self, record: logging.LogRecord) -> bool:
        # レベルフィルタ
        if record.levelno < self.min_level:
            return False

        # レート制限
        if self.rate_limit:
            key = f"{record.name}:{record.funcName}:{record.lineno}"
            current_time = time.time()

            if key in self.last_log_time:
                if current_time - self.last_log_time[key] < self.rate_limit:
                    return False

            self.last_log_time[key] = current_time

        return True


class EnhancedLogger:
    """
    拡張ログシステム

    高性能な非同期ログ処理と構造化ログ出力
    """

    def __init__(self,
                 name: str = "day_trade",
                 log_dir: Union[str, Path] = "logs",
                 async_logging: bool = True,
                 structured_logging: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):

        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.async_logging = async_logging
        self.structured_logging = structured_logging

        # ロガー作成
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # 既存のハンドラーをクリア（重複防止）
        self.logger.handlers.clear()

        # ハンドラー設定
        self._setup_handlers(max_file_size, backup_count)

        # メトリクス収集
        self.metrics: list[LogMetrics] = []
        self.metrics_lock = threading.RLock()

        # シャットダウン時のクリーンアップ登録
        atexit.register(self.shutdown)

    def _setup_handlers(self, max_file_size: int, backup_count: int):
        """ログハンドラーを設定"""
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(message)s [%(name)s] '
            'filename=%(filename)s func_name=%(funcName)s lineno=%(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # パフォーマンスフィルター適用
        console_filter = PerformanceLogFilter(
            min_level=logging.INFO,
            rate_limit=1.0  # 同じ場所からは1秒に1回まで
        )
        console_handler.addFilter(console_filter)

        # 非同期ラッピング
        if self.async_logging:
            console_handler = AsyncLogHandler(console_handler)

        self.logger.addHandler(console_handler)

        # ファイルハンドラー（一般ログ）
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )

        if self.structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)8s] %(message)s [%(name)s] '
                'module=%(module)s func=%(funcName)s line=%(lineno)d '
                'thread=%(threadName)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        if self.async_logging:
            file_handler = AsyncLogHandler(file_handler)

        self.logger.addHandler(file_handler)

        # エラーログファイル（ERROR以上）
        error_file = self.log_dir / f"{self.name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        if self.async_logging:
            error_handler = AsyncLogHandler(error_handler)

        self.logger.addHandler(error_handler)

    def log_with_metrics(self,
                        level: int,
                        message: str,
                        context: Optional[Dict[str, Any]] = None,
                        duration_ms: Optional[float] = None,
                        memory_mb: Optional[float] = None):
        """メトリクス付きログ出力"""
        # 標準ログ出力
        extra = context or {}
        if duration_ms is not None:
            extra['duration_ms'] = duration_ms
        if memory_mb is not None:
            extra['memory_mb'] = memory_mb

        self.logger.log(level, message, extra=extra)

        # メトリクス記録
        metrics = LogMetrics(
            timestamp=time.time(),
            level=logging.getLevelName(level),
            logger_name=self.name,
            message=message,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            thread_id=threading.get_ident(),
            context=context
        )

        with self.metrics_lock:
            self.metrics.append(metrics)

            # メトリクス数の制限（メモリ使用量制御）
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-500:]  # 半分に削減

    def get_metrics(self,
                   hours: int = 1) -> list[LogMetrics]:
        """指定時間内のメトリクスを取得"""
        cutoff_time = time.time() - (hours * 3600)

        with self.metrics_lock:
            return [m for m in self.metrics if m.timestamp >= cutoff_time]

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        metrics = self.get_metrics(1)  # 過去1時間

        if not metrics:
            return {"message": "No metrics available"}

        # 統計計算
        total_logs = len(metrics)
        avg_duration = None
        max_duration = None

        durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)

        # レベル別カウント
        level_counts = {}
        for m in metrics:
            level_counts[m.level] = level_counts.get(m.level, 0) + 1

        return {
            "total_logs_1h": total_logs,
            "average_duration_ms": avg_duration,
            "max_duration_ms": max_duration,
            "level_counts": level_counts,
            "metrics_memory_usage": len(self.metrics)
        }

    def shutdown(self):
        """ログシステムをシャットダウン"""
        for handler in self.logger.handlers:
            if isinstance(handler, AsyncLogHandler):
                handler.shutdown()
            handler.close()

        self.logger.info("ログシステムシャットダウン完了")


# グローバルロガーインスタンス
_enhanced_logger: Optional[EnhancedLogger] = None


def get_enhanced_logger() -> EnhancedLogger:
    """グローバル拡張ロガーを取得"""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger()
    return _enhanced_logger


def log_performance(func_name: str, duration_ms: float, memory_mb: Optional[float] = None):
    """パフォーマンスログを記録"""
    logger = get_enhanced_logger()
    logger.log_with_metrics(
        logging.INFO,
        f"パフォーマンス: {func_name}",
        context={"performance_log": True},
        duration_ms=duration_ms,
        memory_mb=memory_mb
    )


def performance_timer(func):
    """パフォーマンス測定デコレータ"""
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            log_performance(func.__name__, duration_ms)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger = get_enhanced_logger()
            logger.log_with_metrics(
                logging.ERROR,
                f"エラー: {func.__name__} - {e}",
                context={"error_log": True, "function": func.__name__},
                duration_ms=duration_ms
            )
            raise

    return wrapper


# 既存ロガーとの互換性
def setup_enhanced_logging(debug: bool = False):
    """拡張ログシステムを設定"""
    logger = get_enhanced_logger()

    if debug:
        logger.logger.setLevel(logging.DEBUG)
        for handler in logger.logger.handlers:
            if hasattr(handler, 'target_handler'):
                handler.target_handler.setLevel(logging.DEBUG)
            else:
                handler.setLevel(logging.DEBUG)

    logger.logger.info("拡張ログシステム初期化完了")
    return logger.logger