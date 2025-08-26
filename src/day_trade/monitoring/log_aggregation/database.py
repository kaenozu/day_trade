#!/usr/bin/env python3
"""
ログ集約システムのデータベース操作
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from .enums import LogLevel, LogSource
from .models import LogEntry, LogSearchQuery

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class LogDatabase:
    """ログデータベース管理クラス"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialize_database()

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

            self._create_indexes(conn)

    def _create_indexes(self, conn: sqlite3.Connection):
        """インデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON log_entries(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON log_entries(level)",
            "CREATE INDEX IF NOT EXISTS idx_logs_source ON log_entries(source)",
            "CREATE INDEX IF NOT EXISTS idx_logs_component ON log_entries(component)",
            "CREATE INDEX IF NOT EXISTS idx_logs_trace ON log_entries(trace_id)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_pattern ON log_alerts(pattern_id)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON log_alerts(severity)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    async def batch_save_logs(self, log_entries: List[LogEntry]):
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

    def cleanup_old_data(self, cutoff_date: datetime) -> tuple[int, int]:
        """古いデータクリーンアップ"""
        try:
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

                return deleted_logs, deleted_alerts

        except Exception as e:
            logger.error(f"データクリーンアップエラー: {e}")
            return 0, 0