#!/usr/bin/env python3
"""
構造化ログシステム

Issue #312: 詳細ログ・トレーシング・エラー分析システム
問題発生時の迅速な原因特定を支援
"""

import json
import sys
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional

from .logging_config import get_context_logger

# ベースロガー
base_logger = get_context_logger(__name__)


class LogLevel(Enum):
    """ログレベル"""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """イベントタイプ"""

    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    DATA_FETCH = "data_fetch"
    ML_ANALYSIS = "ml_analysis"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_ALERT = "performance_alert"
    RECOVERY_ACTION = "recovery_action"
    USER_ACTION = "user_action"


@dataclass
class LogContext:
    """ログコンテキスト情報"""

    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    operation_name: Optional[str] = None
    parent_operation: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class StructuredLogEntry:
    """構造化ログエントリ"""

    timestamp: str
    level: LogLevel
    event_type: EventType
    message: str
    context: LogContext
    data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    tags: Optional[List[str]] = None


class ContextManager:
    """ログコンテキスト管理"""

    def __init__(self):
        self._local = local()

    def set_context(self, context: LogContext):
        """コンテキスト設定"""
        self._local.context = context

    def get_context(self) -> Optional[LogContext]:
        """コンテキスト取得"""
        return getattr(self._local, "context", None)

    def clear_context(self):
        """コンテキストクリア"""
        if hasattr(self._local, "context"):
            delattr(self._local, "context")

    def update_context(self, **kwargs):
        """コンテキスト更新"""
        current = self.get_context()
        if current:
            for key, value in kwargs.items():
                if hasattr(current, key):
                    setattr(current, key, value)


class StructuredLogger:
    """構造化ロガー"""

    def __init__(self, output_dir: str = "logs", max_file_size_mb: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        self.context_manager = ContextManager()
        self.correlation_counter = 0

        # ログファイルハンドラー設定
        self._setup_file_handlers()

        base_logger.info("構造化ログシステム初期化完了")

    def _setup_file_handlers(self):
        """ファイルハンドラー設定"""
        # メインログファイル
        self.main_log_file = self.output_dir / "application.jsonl"

        # エラー専用ログファイル
        self.error_log_file = self.output_dir / "errors.jsonl"

        # パフォーマンス専用ログファイル
        self.performance_log_file = self.output_dir / "performance.jsonl"

    def _generate_correlation_id(self) -> str:
        """相関ID生成"""
        self.correlation_counter += 1
        return (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.correlation_counter:06d}"
        )

    def create_context(
        self, operation_name: str, user_id: str = None, parent_operation: str = None
    ) -> LogContext:
        """新しいログコンテキスト作成"""
        return LogContext(
            correlation_id=self._generate_correlation_id(),
            session_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            operation_name=operation_name,
            parent_operation=parent_operation,
            request_id=str(uuid.uuid4()),
        )

    @contextmanager
    def operation_context(
        self,
        operation_name: str,
        user_id: str = None,
        data: Dict = None,
        tags: List[str] = None,
    ):
        """操作コンテキストマネージャー"""
        # 現在のコンテキストを取得（親操作として使用）
        current_context = self.context_manager.get_context()
        parent_op = current_context.operation_name if current_context else None

        # 新しいコンテキスト作成
        context = self.create_context(operation_name, user_id, parent_op)

        # 開始ログ
        start_time = datetime.now()
        self.log(
            LogLevel.INFO,
            EventType.SYSTEM_START,
            f"Operation started: {operation_name}",
            context=context,
            data=data,
            tags=tags,
        )

        # コンテキスト設定
        old_context = current_context
        self.context_manager.set_context(context)

        try:
            yield context
        except Exception as e:
            # エラーログ
            self.log_error(
                f"Operation failed: {operation_name}",
                e,
                context=context,
                tags=(tags or []) + ["operation_failure"],
            )
            raise
        finally:
            # 終了ログ
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000

            self.log(
                LogLevel.INFO,
                EventType.SYSTEM_STOP,
                f"Operation completed: {operation_name}",
                context=context,
                execution_time_ms=execution_time,
                tags=tags,
            )

            # コンテキスト復元
            if old_context:
                self.context_manager.set_context(old_context)
            else:
                self.context_manager.clear_context()

    def log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        context: LogContext = None,
        data: Dict = None,
        execution_time_ms: float = None,
        tags: List[str] = None,
    ):
        """構造化ログ出力"""

        # コンテキスト取得
        if context is None:
            context = self.context_manager.get_context()

        if context is None:
            # デフォルトコンテキスト
            context = LogContext(
                correlation_id=self._generate_correlation_id(), session_id="default"
            )

        # ログエントリ作成
        entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            event_type=event_type,
            message=message,
            context=context,
            data=data,
            execution_time_ms=execution_time_ms,
            tags=tags,
        )

        # ファイル出力
        self._write_log_entry(entry)

        # 標準出力（開発時）
        self._print_log_entry(entry)

    def log_error(
        self,
        message: str,
        exception: Exception = None,
        context: LogContext = None,
        tags: List[str] = None,
    ):
        """エラーログ"""
        error_details = None

        if exception:
            error_details = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
                "exception_args": list(exception.args) if exception.args else None,
            }

        entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.ERROR,
            event_type=EventType.ERROR_OCCURRED,
            message=message,
            context=context or self.context_manager.get_context(),
            error_details=error_details,
            tags=tags,
        )

        # エラーログファイルに出力
        self._write_log_entry(entry, self.error_log_file)
        self._write_log_entry(entry, self.main_log_file)

        # 標準エラー出力
        self._print_log_entry(entry, use_stderr=True)

    def log_performance(
        self,
        operation: str,
        execution_time_ms: float,
        data: Dict = None,
        context: LogContext = None,
    ):
        """パフォーマンスログ"""
        entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            event_type=EventType.PERFORMANCE_ALERT,
            message=f"Performance metric: {operation}",
            context=context or self.context_manager.get_context(),
            data=data,
            execution_time_ms=execution_time_ms,
            tags=["performance"],
        )

        # パフォーマンスログファイルに出力
        self._write_log_entry(entry, self.performance_log_file)
        self._write_log_entry(entry, self.main_log_file)

    def _write_log_entry(self, entry: StructuredLogEntry, file_path: Path = None):
        """ログエントリをファイルに書き込み"""
        if file_path is None:
            file_path = self.main_log_file

        try:
            # ファイルサイズチェック・ローテーション
            if (
                file_path.exists()
                and file_path.stat().st_size > self.max_file_size_bytes
            ):
                self._rotate_log_file(file_path)

            # JSON Lines形式で出力
            log_dict = asdict(entry)

            # datetimeとenumの処理
            log_dict["level"] = entry.level.value
            log_dict["event_type"] = entry.event_type.value

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

        except Exception as e:
            # ログ出力エラーは標準エラーに出力
            print(f"Log write error: {e}", file=sys.stderr)

    def _rotate_log_file(self, file_path: Path):
        """ログファイルローテーション"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            rotated_path = file_path.parent / rotated_name

            file_path.rename(rotated_path)
            base_logger.info(
                f"ログファイルローテーション: {file_path} -> {rotated_path}"
            )

        except Exception as e:
            print(f"Log rotation error: {e}", file=sys.stderr)

    def _print_log_entry(self, entry: StructuredLogEntry, use_stderr: bool = False):
        """ログエントリを標準出力に表示（開発用）"""
        try:
            output_stream = sys.stderr if use_stderr else sys.stdout

            # コンパクトな表示形式
            timestamp = entry.timestamp[:19]  # 秒まで
            level = entry.level.value[:4]  # 短縮
            correlation = (
                entry.context.correlation_id[:8] if entry.context else "unknown"
            )

            log_line = f"[{timestamp}] {level} [{correlation}] {entry.message}"

            if entry.execution_time_ms:
                log_line += f" ({entry.execution_time_ms:.1f}ms)"

            if entry.tags:
                log_line += f" #{','.join(entry.tags)}"

            print(log_line, file=output_stream)

        except Exception:
            # 表示エラーは無視（ログファイルには記録済み）
            pass

    def search_logs(self, query: Dict, hours: int = 24) -> List[Dict]:
        """ログ検索"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            results = []

            for log_file in [
                self.main_log_file,
                self.error_log_file,
                self.performance_log_file,
            ]:
                if not log_file.exists():
                    continue

                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(
                                entry["timestamp"]
                            ).timestamp()

                            if entry_time < cutoff_time:
                                continue

                            # クエリマッチング
                            if self._matches_query(entry, query):
                                results.append(entry)

                        except (json.JSONDecodeError, KeyError):
                            continue

            # 時間順ソート
            results.sort(key=lambda x: x["timestamp"])
            return results

        except Exception as e:
            base_logger.error(f"ログ検索エラー: {e}")
            return []

    def _matches_query(self, entry: Dict, query: Dict) -> bool:
        """クエリマッチング判定"""
        for key, value in query.items():
            if (
                key == "level"
                and entry.get("level") != value
                or key == "event_type"
                and entry.get("event_type") != value
                or key == "message_contains"
                and value.lower() not in entry.get("message", "").lower()
                or key == "correlation_id"
                and entry.get("context", {}).get("correlation_id") != value
                or key == "tags"
                and not any(tag in entry.get("tags", []) for tag in value)
            ):
                return False

        return True

    def generate_error_report(self, hours: int = 24) -> Dict:
        """エラーレポート生成"""
        error_logs = self.search_logs({"level": "ERROR"}, hours)

        if not error_logs:
            return {"error_count": 0, "report_period_hours": hours}

        # エラー分析
        error_types = {}
        operation_failures = {}
        timeline = []

        for log_entry in error_logs:
            # エラータイプ集計
            error_details = log_entry.get("error_details", {})
            error_type = error_details.get("exception_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

            # 操作別失敗集計
            context = log_entry.get("context", {})
            operation = context.get("operation_name", "unknown")
            operation_failures[operation] = operation_failures.get(operation, 0) + 1

            # タイムライン
            timeline.append(
                {
                    "timestamp": log_entry["timestamp"],
                    "operation": operation,
                    "error_type": error_type,
                    "message": log_entry["message"],
                }
            )

        return {
            "report_period_hours": hours,
            "error_count": len(error_logs),
            "error_types": dict(
                sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            ),
            "operation_failures": dict(
                sorted(operation_failures.items(), key=lambda x: x[1], reverse=True)
            ),
            "error_timeline": timeline[-20:],  # 最新20件
            "generated_at": datetime.now().isoformat(),
        }


# グローバルインスタンス
_global_structured_logger = None


def get_structured_logger() -> StructuredLogger:
    """グローバル構造化ロガー取得"""
    global _global_structured_logger
    if _global_structured_logger is None:
        _global_structured_logger = StructuredLogger()
    return _global_structured_logger


# 便利関数
def log_info(message: str, data: Dict = None, tags: List[str] = None):
    """情報ログ"""
    logger = get_structured_logger()
    logger.log(LogLevel.INFO, EventType.USER_ACTION, message, data=data, tags=tags)


def log_error(message: str, exception: Exception = None, tags: List[str] = None):
    """エラーログ"""
    logger = get_structured_logger()
    logger.log_error(message, exception, tags=tags)


def log_performance(operation: str, execution_time_ms: float, data: Dict = None):
    """パフォーマンスログ"""
    logger = get_structured_logger()
    logger.log_performance(operation, execution_time_ms, data)


@contextmanager
def operation_logging(
    operation_name: str, user_id: str = None, data: Dict = None, tags: List[str] = None
):
    """操作ログコンテキスト"""
    logger = get_structured_logger()
    with logger.operation_context(operation_name, user_id, data, tags):
        yield


if __name__ == "__main__":
    # テスト実行
    print("Structured Logging System Test")
    print("=" * 50)

    try:
        logger = get_structured_logger()

        # 基本ログテスト
        with logger.operation_context(
            "test_main_operation", "test_user", {"param": "value"}, ["test"]
        ):
            # 情報ログ
            logger.log(
                LogLevel.INFO,
                EventType.DATA_FETCH,
                "Data fetch started",
                {"symbols": ["TEST1", "TEST2"]},
            )

            # サブ操作
            with logger.operation_context("test_sub_operation"):
                logger.log(
                    LogLevel.DEBUG,
                    EventType.ML_ANALYSIS,
                    "ML analysis in progress",
                    {"model": "test_model"},
                )

                # パフォーマンスログ
                logger.log_performance("ml_analysis", 1500.0, {"accuracy": 0.95})

                # エラーログテスト
                try:
                    raise ValueError("Test error for logging")
                except Exception as e:
                    logger.log_error("Test error occurred", e, ["test_error"])

        # ログ検索テスト
        import time

        time.sleep(0.1)  # ログ書き込み完了待ち

        error_logs = logger.search_logs({"level": "ERROR"}, hours=1)
        print(f"Found {len(error_logs)} error logs")

        # エラーレポート生成
        error_report = logger.generate_error_report(hours=1)
        print(f"Error report: {error_report['error_count']} errors found")

        print("Structured logging test completed successfully")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
