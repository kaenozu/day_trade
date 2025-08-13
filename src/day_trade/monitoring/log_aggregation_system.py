#!/usr/bin/env python3
"""
統合ログ集約・分析システム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

ELK Stack風のログ集約・検索・分析機能を提供:
- ログ収集・パース・インデックス化
- リアルタイムログストリーム処理
- 高速ログ検索・フィルタリング
- ログ分析・パターン検出
- アラート生成・通知
- ダッシュボード統合
- ログローテーション・アーカイブ
"""

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"log_agg_key_{hash(str(args))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class LogLevel(Enum):
    """ログレベル"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    """ログソース種別"""

    APPLICATION = "application"
    SYSTEM = "system"
    DATABASE = "database"
    API = "api"
    TRADING = "trading"
    ML = "ml"
    SECURITY = "security"
    PERFORMANCE = "performance"


class AlertSeverity(Enum):
    """アラート重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """ログエントリ"""

    id: str
    timestamp: datetime
    level: LogLevel
    source: LogSource
    component: str
    message: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    raw_log: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogPattern:
    """ログパターン定義"""

    pattern_id: str
    name: str
    description: str
    regex_pattern: str
    source_filter: Optional[LogSource] = None
    level_filter: Optional[LogLevel] = None
    alert_threshold: int = 10  # この回数検出されたらアラート
    time_window_minutes: int = 5
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogAlert:
    """ログアラート"""

    alert_id: str
    pattern_id: str
    pattern_name: str
    severity: AlertSeverity
    message: str
    occurrence_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    related_logs: List[str]  # ログIDリスト
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class LogSearchQuery:
    """ログ検索クエリ"""

    query_text: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: List[LogLevel] = field(default_factory=list)
    sources: List[LogSource] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    limit: int = 1000
    order_by: str = "timestamp"
    order_desc: bool = True


class LogParser(ABC):
    """抽象ログパーサー"""

    @abstractmethod
    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """ログをパース"""
        pass

    @abstractmethod
    def can_parse(self, raw_log: str) -> bool:
        """このパーサーが対象ログをパース可能かチェック"""
        pass


class StructuredLogParser(LogParser):
    """構造化ログ（JSON）パーサー"""

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """JSON形式ログをパース"""
        try:
            data = json.loads(raw_log.strip())

            # 必須フィールドチェック
            if not all(key in data for key in ["timestamp", "level", "message"]):
                return None

            # タイムスタンプ解析
            timestamp = self._parse_timestamp(data["timestamp"])
            if not timestamp:
                return None

            # ログレベル解析
            try:
                level = LogLevel(data["level"].upper())
            except ValueError:
                level = LogLevel.INFO

            # ソース推定
            source = self._infer_source(data)

            # エントリ作成
            return LogEntry(
                id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                timestamp=timestamp,
                level=level,
                source=source,
                component=data.get("logger_name", data.get("component", "unknown")),
                message=data["message"],
                structured_data={
                    k: v
                    for k, v in data.items()
                    if k not in ["timestamp", "level", "message", "logger_name"]
                },
                tags=data.get("tags", []),
                trace_id=data.get("trace_id"),
                user_id=data.get("user_id"),
                session_id=data.get("session_id"),
                raw_log=raw_log,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"JSON ログパースエラー: {e}")
            return None

    def can_parse(self, raw_log: str) -> bool:
        """JSON形式かチェック"""
        stripped = raw_log.strip()
        return stripped.startswith("{") and stripped.endswith("}")

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """タイムスタンプ文字列を解析"""
        try:
            # ISO形式
            if "T" in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # その他の形式を試す
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            return None

        except Exception as e:
            logger.debug(f"タイムスタンプ解析エラー: {e}")
            return None

    def _infer_source(self, data: Dict[str, Any]) -> LogSource:
        """ログデータからソースを推定"""
        logger_name = data.get("logger_name", "").lower()
        component = data.get("component", "").lower()
        message = data.get("message", "").lower()

        # パターンマッチングでソース推定
        if any(
            keyword in logger_name + component + message
            for keyword in ["database", "db", "sql", "query"]
        ):
            return LogSource.DATABASE

        elif any(
            keyword in logger_name + component + message
            for keyword in ["api", "http", "rest", "endpoint"]
        ):
            return LogSource.API

        elif any(
            keyword in logger_name + component + message
            for keyword in ["trading", "trade", "order", "portfolio"]
        ):
            return LogSource.TRADING

        elif any(
            keyword in logger_name + component + message
            for keyword in ["ml", "model", "prediction", "training"]
        ):
            return LogSource.ML

        elif any(
            keyword in logger_name + component + message
            for keyword in ["security", "auth", "login", "permission"]
        ):
            return LogSource.SECURITY

        elif any(
            keyword in logger_name + component + message
            for keyword in ["performance", "metric", "cpu", "memory"]
        ):
            return LogSource.PERFORMANCE

        elif any(
            keyword in logger_name + component + message
            for keyword in ["system", "os", "disk", "network"]
        ):
            return LogSource.SYSTEM

        else:
            return LogSource.APPLICATION


class StandardLogParser(LogParser):
    """標準形式ログパーサー"""

    def __init__(self):
        # 標準ログ形式のパターン
        self.patterns = [
            # 2024-01-01 12:00:00 - component - LEVEL - message
            re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*-\s*(?P<component>[^-]+)\s*-\s*(?P<level>\w+)\s*-\s*(?P<message>.*)"
            ),
            # [2024-01-01 12:00:00] LEVEL: message
            re.compile(
                r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(?P<level>\w+):\s*(?P<message>.*)"
            ),
            # 2024-01-01 12:00:00 LEVEL message
            re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)"
            ),
        ]

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """標準形式ログをパース"""
        try:
            for pattern in self.patterns:
                match = pattern.match(raw_log.strip())
                if match:
                    groups = match.groupdict()

                    # タイムスタンプ解析
                    try:
                        timestamp = datetime.strptime(
                            groups["timestamp"], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        continue

                    # ログレベル解析
                    try:
                        level = LogLevel(groups["level"].upper())
                    except ValueError:
                        level = LogLevel.INFO

                    return LogEntry(
                        id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                        timestamp=timestamp,
                        level=level,
                        source=LogSource.APPLICATION,
                        component=groups.get("component", "unknown"),
                        message=groups["message"],
                        raw_log=raw_log,
                    )

            return None

        except Exception as e:
            logger.debug(f"標準ログパースエラー: {e}")
            return None

    def can_parse(self, raw_log: str) -> bool:
        """標準形式かチェック"""
        return any(pattern.match(raw_log.strip()) for pattern in self.patterns)


class LogAggregationSystem:
    """統合ログ集約・分析システム"""

    def __init__(
        self,
        storage_path: str = "data/logs",
        enable_cache: bool = True,
        max_memory_logs: int = 10000,
        retention_days: int = 30,
        enable_compression: bool = True,
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.max_memory_logs = max_memory_logs
        self.retention_days = retention_days
        self.enable_compression = enable_compression

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "raw").mkdir(exist_ok=True)
        (self.storage_path / "processed").mkdir(exist_ok=True)
        (self.storage_path / "archives").mkdir(exist_ok=True)
        (self.storage_path / "indexes").mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.storage_path / "logs.db"
        self._initialize_database()

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=128, l2_memory_mb=512, l3_disk_mb=2048
                )
                logger.info("ログ集約キャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # ログパーサー初期化
        self.parsers: List[LogParser] = [StructuredLogParser(), StandardLogParser()]

        # ログストリーム処理
        self.log_stream = deque(maxlen=max_memory_logs)
        self.log_stream_lock = threading.Lock()

        # パターン検出・アラート
        self.log_patterns: Dict[str, LogPattern] = {}
        self.active_alerts: Dict[str, LogAlert] = {}
        self.pattern_counters: Dict[str, Dict[datetime, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # ストリーミング処理スレッド
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()

        # デフォルト設定
        self._setup_default_patterns()

        logger.info("統合ログ集約システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - メモリログ上限: {max_memory_logs}")
        logger.info(f"  - 保持期間: {retention_days}日")
        logger.info(f"  - 圧縮: {'有効' if enable_compression else '無効'}")

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # ログエントリテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    structured_data TEXT,
                    tags TEXT,
                    trace_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    raw_log TEXT,
                    parsed_at TEXT NOT NULL
                )
            """
            )

            # ログパターンテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    regex_pattern TEXT NOT NULL,
                    source_filter TEXT,
                    level_filter TEXT,
                    alert_threshold INTEGER DEFAULT 10,
                    time_window_minutes INTEGER DEFAULT 5,
                    severity TEXT DEFAULT 'medium',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """
            )

            # アラートテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log_alerts (
                    alert_id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    first_occurrence TEXT NOT NULL,
                    last_occurrence TEXT NOT NULL,
                    related_logs TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON log_entries(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_level ON log_entries(level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_source ON log_entries(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_component ON log_entries(component)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_trace ON log_entries(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_pattern ON log_alerts(pattern_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON log_alerts(severity)"
            )

    def _setup_default_patterns(self):
        """デフォルトログパターン設定"""
        # エラーパターン
        error_pattern = LogPattern(
            pattern_id="error_pattern",
            name="エラーログ検出",
            description="ERROR/CRITICALレベルのログを検出",
            regex_pattern=r"(ERROR|CRITICAL|Exception|Error|Failed|Failure)",
            level_filter=LogLevel.ERROR,
            alert_threshold=5,
            time_window_minutes=5,
            severity=AlertSeverity.HIGH,
        )
        self.log_patterns[error_pattern.pattern_id] = error_pattern

        # 認証失敗パターン
        auth_pattern = LogPattern(
            pattern_id="auth_failure_pattern",
            name="認証失敗検出",
            description="認証失敗を検出",
            regex_pattern=r"(authentication failed|login failed|invalid credentials|unauthorized)",
            source_filter=LogSource.SECURITY,
            alert_threshold=3,
            time_window_minutes=5,
            severity=AlertSeverity.CRITICAL,
        )
        self.log_patterns[auth_pattern.pattern_id] = auth_pattern

        # パフォーマンス問題パターン
        performance_pattern = LogPattern(
            pattern_id="performance_issue_pattern",
            name="パフォーマンス問題検出",
            description="遅いクエリや高レスポンス時間を検出",
            regex_pattern=r"(slow query|timeout|high latency|response time)",
            source_filter=LogSource.PERFORMANCE,
            alert_threshold=10,
            time_window_minutes=10,
            severity=AlertSeverity.MEDIUM,
        )
        self.log_patterns[performance_pattern.pattern_id] = performance_pattern

        # データベースエラーパターン
        db_pattern = LogPattern(
            pattern_id="database_error_pattern",
            name="データベースエラー検出",
            description="データベース関連のエラーを検出",
            regex_pattern=r"(database error|connection failed|deadlock|constraint violation)",
            source_filter=LogSource.DATABASE,
            alert_threshold=3,
            time_window_minutes=5,
            severity=AlertSeverity.HIGH,
        )
        self.log_patterns[db_pattern.pattern_id] = db_pattern

    async def start_processing(self):
        """ログストリーム処理開始"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("ログ処理は既に開始されています")
            return

        logger.info("ログストリーム処理開始")

        self.stop_processing.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop, name="LogProcessingThread", daemon=True
        )
        self.processing_thread.start()

    async def stop_processing(self):
        """ログストリーム処理停止"""
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.warning("ログ処理は既に停止されています")
            return

        logger.info("ログストリーム処理停止中...")

        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=10)

        logger.info("ログストリーム処理停止完了")

    def _processing_loop(self):
        """ログ処理メインループ"""
        logger.info("ログ処理ループ開始")

        while not self.stop_processing.is_set():
            try:
                # メモリ内ログのバッチ処理
                logs_to_process = []
                with self.log_stream_lock:
                    if self.log_stream:
                        logs_to_process = list(self.log_stream)
                        self.log_stream.clear()

                if logs_to_process:
                    # パターン検出とアラート処理
                    for log_entry in logs_to_process:
                        self._process_log_patterns(log_entry)

                    # データベースバッチ保存
                    asyncio.run_coroutine_threadsafe(
                        self._batch_save_logs(logs_to_process), asyncio.new_event_loop()
                    )

                # アラートチェック
                self._check_pattern_alerts()

                # 定期クリーンアップ
                current_time = time.time()
                if (
                    not hasattr(self, "_last_cleanup")
                    or current_time - self._last_cleanup > 3600
                ):  # 1時間毎
                    self._cleanup_old_data()
                    self._last_cleanup = current_time

                # 短い間隔で再チェック
                self.stop_processing.wait(5)  # 5秒間隔

            except Exception as e:
                logger.error(f"ログ処理ループエラー: {e}")
                self.stop_processing.wait(30)  # エラー時は30秒待機

        logger.info("ログ処理ループ終了")

    async def ingest_log(
        self, raw_log: str, source_hint: Optional[LogSource] = None
    ) -> bool:
        """ログを取り込み"""
        try:
            # ログパース
            parsed_log = None
            for parser in self.parsers:
                if parser.can_parse(raw_log):
                    parsed_log = parser.parse(raw_log)
                    if parsed_log:
                        break

            if not parsed_log:
                # パースできない場合は基本的なエントリを作成
                parsed_log = LogEntry(
                    id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                    timestamp=datetime.utcnow(),
                    level=LogLevel.INFO,
                    source=source_hint or LogSource.APPLICATION,
                    component="unknown",
                    message=raw_log,
                    raw_log=raw_log,
                )

            # メモリストリームに追加
            with self.log_stream_lock:
                self.log_stream.append(parsed_log)

            # キャッシュに保存
            if self.cache_manager:
                cache_key = generate_unified_cache_key("log_entry", parsed_log.id)
                self.cache_manager.put(cache_key, parsed_log, priority=3.0)

            return True

        except Exception as e:
            logger.error(f"ログ取り込みエラー: {e}")
            return False

    def _process_log_patterns(self, log_entry: LogEntry):
        """ログパターン処理"""
        try:
            current_minute = datetime.utcnow().replace(second=0, microsecond=0)

            for pattern_id, pattern in self.log_patterns.items():
                if not pattern.enabled:
                    continue

                # フィルタチェック
                if pattern.source_filter and log_entry.source != pattern.source_filter:
                    continue

                if pattern.level_filter and log_entry.level != pattern.level_filter:
                    continue

                # パターンマッチング
                if re.search(pattern.regex_pattern, log_entry.message, re.IGNORECASE):
                    # カウンター更新
                    self.pattern_counters[pattern_id][current_minute] += 1

                    logger.debug(
                        f"ログパターン検出: {pattern.name} - {log_entry.message[:100]}"
                    )

        except Exception as e:
            logger.error(f"ログパターン処理エラー: {e}")

    def _check_pattern_alerts(self):
        """パターンアラートチェック"""
        try:
            current_time = datetime.utcnow()

            for pattern_id, pattern in self.log_patterns.items():
                if not pattern.enabled:
                    continue

                # 時間窓内のカウントを集計
                time_threshold = current_time - timedelta(
                    minutes=pattern.time_window_minutes
                )
                total_count = 0

                for timestamp, count in self.pattern_counters[pattern_id].items():
                    if timestamp >= time_threshold:
                        total_count += count

                # アラート閾値チェック
                if total_count >= pattern.alert_threshold:
                    # 既存のアクティブアラートをチェック
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if (
                            alert.pattern_id == pattern_id
                            and not alert.resolved
                            and (current_time - alert.last_occurrence).total_seconds()
                            < pattern.time_window_minutes * 60
                        ):
                            existing_alert = alert
                            break

                    if existing_alert:
                        # 既存アラート更新
                        existing_alert.occurrence_count += total_count
                        existing_alert.last_occurrence = current_time
                    else:
                        # 新規アラート作成
                        alert = LogAlert(
                            alert_id=f"alert_{pattern_id}_{int(time.time())}",
                            pattern_id=pattern_id,
                            pattern_name=pattern.name,
                            severity=pattern.severity,
                            message=f"パターン '{pattern.name}' が {total_count}回検出されました",
                            occurrence_count=total_count,
                            first_occurrence=current_time,
                            last_occurrence=current_time,
                            related_logs=[],
                        )

                        self.active_alerts[alert.alert_id] = alert

                        # アラート通知（実装は環境に依存）
                        asyncio.run_coroutine_threadsafe(
                            self._send_alert_notification(alert),
                            asyncio.new_event_loop(),
                        )

                        logger.warning(f"ログアラート生成: {alert.message}")

            # 古いカウンターをクリーンアップ
            cleanup_threshold = current_time - timedelta(hours=1)
            for pattern_id in self.pattern_counters:
                timestamps_to_remove = []
                for timestamp in self.pattern_counters[pattern_id]:
                    if timestamp < cleanup_threshold:
                        timestamps_to_remove.append(timestamp)

                for timestamp in timestamps_to_remove:
                    del self.pattern_counters[pattern_id][timestamp]

        except Exception as e:
            logger.error(f"パターンアラートチェックエラー: {e}")

    async def _batch_save_logs(self, log_entries: List[LogEntry]):
        """ログエントリバッチ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for log_entry in log_entries:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO log_entries
                        (id, timestamp, level, source, component, message, structured_data,
                         tags, trace_id, user_id, session_id, raw_log, parsed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            log_entry.id,
                            log_entry.timestamp.isoformat(),
                            log_entry.level.value,
                            log_entry.source.value,
                            log_entry.component,
                            log_entry.message,
                            json.dumps(log_entry.structured_data),
                            json.dumps(log_entry.tags),
                            log_entry.trace_id,
                            log_entry.user_id,
                            log_entry.session_id,
                            log_entry.raw_log,
                            log_entry.parsed_at.isoformat(),
                        ),
                    )

                conn.commit()
                logger.debug(f"ログエントリ保存完了: {len(log_entries)}件")

        except Exception as e:
            logger.error(f"ログバッチ保存エラー: {e}")

    async def _send_alert_notification(self, alert: LogAlert):
        """アラート通知送信（実装は環境依存）"""
        try:
            # ここでは通知の模擬実装
            notification_message = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "pattern": alert.pattern_name,
                "message": alert.message,
                "occurrence_count": alert.occurrence_count,
                "timestamp": alert.first_occurrence.isoformat(),
            }

            # 実際の実装では、メール、Slack、Teams等への通知
            logger.info(
                f"アラート通知: {json.dumps(notification_message, ensure_ascii=False)}"
            )

        except Exception as e:
            logger.error(f"アラート通知送信エラー: {e}")

    async def search_logs(self, query: LogSearchQuery) -> List[LogEntry]:
        """ログ検索"""
        try:
            # クエリ構築
            where_conditions = []
            params = []

            if query.query_text:
                where_conditions.append("(message LIKE ? OR raw_log LIKE ?)")
                params.extend([f"%{query.query_text}%", f"%{query.query_text}%"])

            if query.start_time:
                where_conditions.append("timestamp >= ?")
                params.append(query.start_time.isoformat())

            if query.end_time:
                where_conditions.append("timestamp <= ?")
                params.append(query.end_time.isoformat())

            if query.levels:
                level_placeholders = ",".join(["?" for _ in query.levels])
                where_conditions.append(f"level IN ({level_placeholders})")
                params.extend([level.value for level in query.levels])

            if query.sources:
                source_placeholders = ",".join(["?" for _ in query.sources])
                where_conditions.append(f"source IN ({source_placeholders})")
                params.extend([source.value for source in query.sources])

            if query.components:
                comp_placeholders = ",".join(["?" for _ in query.components])
                where_conditions.append(f"component IN ({comp_placeholders})")
                params.extend(query.components)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            order_clause = (
                f"ORDER BY {query.order_by} {'DESC' if query.order_desc else 'ASC'}"
            )
            limit_clause = f"LIMIT {query.limit}"

            sql_query = f"""
                SELECT id, timestamp, level, source, component, message, structured_data,
                       tags, trace_id, user_id, session_id, raw_log, parsed_at
                FROM log_entries
                {where_clause}
                {order_clause}
                {limit_clause}
            """

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql_query, params)

                results = []
                for row in cursor.fetchall():
                    log_entry = LogEntry(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        level=LogLevel(row[2]),
                        source=LogSource(row[3]),
                        component=row[4],
                        message=row[5],
                        structured_data=json.loads(row[6]) if row[6] else {},
                        tags=json.loads(row[7]) if row[7] else [],
                        trace_id=row[8],
                        user_id=row[9],
                        session_id=row[10],
                        raw_log=row[11],
                        parsed_at=datetime.fromisoformat(row[12]),
                    )
                    results.append(log_entry)

                return results

        except Exception as e:
            logger.error(f"ログ検索エラー: {e}")
            return []

    async def get_log_analytics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """ログ分析データ取得"""
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.utcnow()

            with sqlite3.connect(self.db_path) as conn:
                # 基本統計
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as total_logs,
                           COUNT(DISTINCT component) as unique_components,
                           COUNT(DISTINCT source) as unique_sources
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                basic_stats = cursor.fetchone()

                # レベル別統計
                cursor = conn.execute(
                    """
                    SELECT level, COUNT(*) as count
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY level
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                level_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # ソース別統計
                cursor = conn.execute(
                    """
                    SELECT source, COUNT(*) as count
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY source
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                source_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # 時間別統計（1時間毎）
                cursor = conn.execute(
                    """
                    SELECT
                        strftime('%Y-%m-%d %H', timestamp) as hour,
                        COUNT(*) as count
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY strftime('%Y-%m-%d %H', timestamp)
                    ORDER BY hour
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                hourly_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # トップエラー
                cursor = conn.execute(
                    """
                    SELECT message, COUNT(*) as count
                    FROM log_entries
                    WHERE level IN ('ERROR', 'CRITICAL')
                          AND timestamp BETWEEN ? AND ?
                    GROUP BY message
                    ORDER BY count DESC
                    LIMIT 10
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                top_errors = [
                    {"message": row[0], "count": row[1]} for row in cursor.fetchall()
                ]

                return {
                    "period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_hours": (end_time - start_time).total_seconds()
                        / 3600,
                    },
                    "basic_statistics": {
                        "total_logs": basic_stats[0],
                        "unique_components": basic_stats[1],
                        "unique_sources": basic_stats[2],
                    },
                    "level_distribution": level_stats,
                    "source_distribution": source_stats,
                    "hourly_distribution": hourly_stats,
                    "top_errors": top_errors,
                    "active_alerts": len(self.active_alerts),
                    "patterns_enabled": sum(
                        1 for p in self.log_patterns.values() if p.enabled
                    ),
                }

        except Exception as e:
            logger.error(f"ログ分析エラー: {e}")
            return {}

    def _cleanup_old_data(self):
        """古いデータクリーンアップ"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

            with sqlite3.connect(self.db_path) as conn:
                # 古いログエントリ削除
                cursor = conn.execute(
                    """
                    DELETE FROM log_entries
                    WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                deleted_logs = cursor.rowcount

                # 古いアラート削除
                cursor = conn.execute(
                    """
                    DELETE FROM log_alerts
                    WHERE first_occurrence < ? AND resolved = 1
                """,
                    (cutoff_date.isoformat(),),
                )

                deleted_alerts = cursor.rowcount

                conn.commit()

                if deleted_logs > 0 or deleted_alerts > 0:
                    logger.info(
                        f"古いデータクリーンアップ: ログ{deleted_logs}件、アラート{deleted_alerts}件削除"
                    )

            # 古いアーカイブファイルクリーンアップ
            archive_path = self.storage_path / "archives"
            for archive_file in archive_path.glob("*.gz"):
                if archive_file.stat().st_mtime < cutoff_date.timestamp():
                    archive_file.unlink()

        except Exception as e:
            logger.error(f"データクリーンアップエラー: {e}")

    async def export_logs(
        self,
        query: LogSearchQuery,
        export_format: str = "json",
        export_path: Optional[str] = None,
    ) -> str:
        """ログエクスポート"""
        logger.info(f"ログエクスポート開始: {export_format}")

        try:
            # ログ検索
            logs = await self.search_logs(query)

            # エクスポートファイルパス
            if not export_path:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                export_path = str(
                    self.storage_path / "exports" / f"logs_{timestamp}.{export_format}"
                )

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            if export_format == "json":
                # JSON形式でエクスポート
                log_data = []
                for log in logs:
                    log_data.append(
                        {
                            "id": log.id,
                            "timestamp": log.timestamp.isoformat(),
                            "level": log.level.value,
                            "source": log.source.value,
                            "component": log.component,
                            "message": log.message,
                            "structured_data": log.structured_data,
                            "tags": log.tags,
                            "trace_id": log.trace_id,
                            "user_id": log.user_id,
                            "session_id": log.session_id,
                        }
                    )

                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)

            elif export_format == "csv":
                # CSV形式でエクスポート
                df_data = []
                for log in logs:
                    df_data.append(
                        {
                            "timestamp": log.timestamp.isoformat(),
                            "level": log.level.value,
                            "source": log.source.value,
                            "component": log.component,
                            "message": log.message,
                            "trace_id": log.trace_id or "",
                            "user_id": log.user_id or "",
                        }
                    )

                df = pd.DataFrame(df_data)
                df.to_csv(export_file, index=False, encoding="utf-8")

            logger.info(f"ログエクスポート完了: {export_file} ({len(logs)}件)")
            return str(export_file)

        except Exception as e:
            logger.error(f"ログエクスポートエラー: {e}")
            raise

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("統合ログ集約システム クリーンアップ開始")

        # 処理停止
        await self.stop_processing()

        # メモリクリア
        with self.log_stream_lock:
            self.log_stream.clear()

        self.active_alerts.clear()
        self.pattern_counters.clear()

        logger.info("統合ログ集約システム クリーンアップ完了")


# Factory function
def create_log_aggregation_system(
    storage_path: str = "data/logs",
    enable_cache: bool = True,
    max_memory_logs: int = 10000,
    retention_days: int = 30,
) -> LogAggregationSystem:
    """統合ログ集約システム作成"""
    return LogAggregationSystem(
        storage_path=storage_path,
        enable_cache=enable_cache,
        max_memory_logs=max_memory_logs,
        retention_days=retention_days,
    )


# 後方互換性のためのエイリアス
def get_log_aggregation_system(*args, **kwargs) -> LogAggregationSystem:
    """後方互換性のためのエイリアス関数"""
    return create_log_aggregation_system(*args, **kwargs)


if __name__ == "__main__":
    # テスト実行
    async def test_log_aggregation_system():
        print("=== Issue #417 統合ログ集約・分析システムテスト ===")

        try:
            # ログ集約システム初期化
            log_system = create_log_aggregation_system(
                storage_path="test_logs",
                enable_cache=True,
                max_memory_logs=1000,
                retention_days=7,
            )

            print("\n1. 統合ログ集約システム初期化完了")
            print(f"   ストレージパス: {log_system.storage_path}")
            print(f"   ログパターン数: {len(log_system.log_patterns)}")
            print(f"   パーサー数: {len(log_system.parsers)}")

            # ログ処理開始
            print("\n2. ログ処理開始...")
            await log_system.start_processing()

            # テストログの取り込み
            print("\n3. テストログ取り込みテスト...")

            # JSON構造化ログ
            json_log = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "INFO",
                    "logger_name": "test.component",
                    "message": "Application started successfully",
                    "user_id": "test_user",
                    "trace_id": "trace_123",
                }
            )

            success = await log_system.ingest_log(json_log)
            print(f"   JSON構造化ログ取り込み: {'成功' if success else '失敗'}")

            # エラーログ（アラート発生テスト用）
            for i in range(7):  # 閾値5を超える
                error_log = json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": "ERROR",
                        "logger_name": "test.error",
                        "message": f"Database connection failed: attempt {i + 1}",
                        "error": "ConnectionError",
                    }
                )
                await log_system.ingest_log(error_log)

            print("   エラーログ取り込み: 7件")

            # 標準形式ログ
            standard_log = "2024-01-15 12:00:00 - api.handler - INFO - Request processed successfully"
            success = await log_system.ingest_log(standard_log)
            print(f"   標準形式ログ取り込み: {'成功' if success else '失敗'}")

            # パフォーマンスログ
            perf_log = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "WARNING",
                    "logger_name": "performance.monitor",
                    "message": "Slow query detected: SELECT * FROM large_table took 5.2s",
                    "duration": 5.2,
                    "query": "SELECT * FROM large_table",
                }
            )
            await log_system.ingest_log(perf_log)
            print("   パフォーマンスログ取り込み: 成功")

            # 処理時間を与える
            await asyncio.sleep(2)

            # ログ検索テスト
            print("\n4. ログ検索テスト...")

            search_query = LogSearchQuery(
                query_text="connection",
                start_time=datetime.utcnow() - timedelta(minutes=5),
                end_time=datetime.utcnow(),
                limit=50,
            )

            search_results = await log_system.search_logs(search_query)
            print(f"   検索結果: {len(search_results)}件")

            if search_results:
                print(
                    f"   最新ログ: {search_results[0].level.value} - {search_results[0].message[:50]}"
                )

            # ログ分析テスト
            print("\n5. ログ分析テスト...")
            analytics = await log_system.get_log_analytics(
                start_time=datetime.utcnow() - timedelta(hours=1),
                end_time=datetime.utcnow(),
            )

            if analytics:
                stats = analytics.get("basic_statistics", {})
                print(f"   総ログ数: {stats.get('total_logs', 0)}")
                print(
                    f"   ユニークコンポーネント数: {stats.get('unique_components', 0)}"
                )
                print(f"   ユニークソース数: {stats.get('unique_sources', 0)}")

                level_dist = analytics.get("level_distribution", {})
                print(f"   レベル分布: {dict(level_dist)}")

                if analytics.get("top_errors"):
                    print("   トップエラー:")
                    for error in analytics["top_errors"][:3]:
                        print(f"     - {error['message'][:40]}... ({error['count']}回)")

            # アクティブアラート確認
            print("\n6. アクティブアラート確認...")
            print(f"   アクティブアラート数: {len(log_system.active_alerts)}")

            for alert_id, alert in list(log_system.active_alerts.items())[:3]:
                print(f"   アラート: {alert.pattern_name} - {alert.message}")
                print(
                    f"     重要度: {alert.severity.value}, 発生回数: {alert.occurrence_count}"
                )

            # ログパターン確認
            print("\n7. ログパターン確認...")
            for pattern_id, pattern in log_system.log_patterns.items():
                enabled_status = "有効" if pattern.enabled else "無効"
                print(
                    f"   {pattern.name}: {enabled_status} (閾値: {pattern.alert_threshold})"
                )

            # ログエクスポートテスト
            print("\n8. ログエクスポートテスト...")
            try:
                export_query = LogSearchQuery(limit=100)
                export_file = await log_system.export_logs(
                    export_query, export_format="json"
                )
                print(f"   エクスポート成功: {export_file}")

                # エクスポートファイル確認
                with open(export_file, encoding="utf-8") as f:
                    exported_data = json.load(f)
                print(f"   エクスポートデータ: {len(exported_data)}件")

            except Exception as e:
                print(f"   エクスポートエラー: {e}")

            # パフォーマンステスト
            print("\n9. パフォーマンステスト...")
            start_time = time.time()

            # 大量ログ取り込み
            for i in range(100):
                test_log = json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": "INFO",
                        "logger_name": f"test.batch.{i % 5}",
                        "message": f"Batch test message {i}",
                        "batch_id": i,
                    }
                )
                await log_system.ingest_log(test_log)

            processing_time = time.time() - start_time
            print(f"   100件ログ取り込み時間: {processing_time:.2f}秒")
            print(f"   スループット: {100 / processing_time:.1f}件/秒")

            # しばらく処理継続
            await asyncio.sleep(3)

            # システム停止
            await log_system.stop_processing()

            # クリーンアップ
            await log_system.cleanup()

            print("\n✅ Issue #417 統合ログ集約・分析システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_log_aggregation_system())
