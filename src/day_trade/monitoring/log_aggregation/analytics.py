#!/usr/bin/env python3
"""
ログ分析とレポート機能
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class LogAnalytics:
    """ログ分析クラス"""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def get_log_analytics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
        active_alerts_count: int = 0, patterns_enabled_count: int = 0
    ) -> Dict[str, Any]:
        """ログ分析データ取得"""
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.utcnow()

            with sqlite3.connect(self.db_path) as conn:
                # 基本統計
                basic_stats = self._get_basic_statistics(
                    conn, start_time, end_time
                )

                # レベル別統計
                level_stats = self._get_level_statistics(
                    conn, start_time, end_time
                )

                # ソース別統計
                source_stats = self._get_source_statistics(
                    conn, start_time, end_time
                )

                # 時間別統計（1時間毎）
                hourly_stats = self._get_hourly_statistics(
                    conn, start_time, end_time
                )

                # トップエラー
                top_errors = self._get_top_errors(
                    conn, start_time, end_time
                )

                return {
                    "period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_hours": (end_time - start_time).total_seconds()
                        / 3600,
                    },
                    "basic_statistics": basic_stats,
                    "level_distribution": level_stats,
                    "source_distribution": source_stats,
                    "hourly_distribution": hourly_stats,
                    "top_errors": top_errors,
                    "active_alerts": active_alerts_count,
                    "patterns_enabled": patterns_enabled_count,
                }

        except Exception as e:
            logger.error(f"ログ分析エラー: {e}")
            return {}

    def _get_basic_statistics(
        self, conn: sqlite3.Connection, start_time: datetime, end_time: datetime
    ) -> Dict[str, int]:
        """基本統計を取得"""
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

        result = cursor.fetchone()
        return {
            "total_logs": result[0],
            "unique_components": result[1],
            "unique_sources": result[2],
        }

    def _get_level_statistics(
        self, conn: sqlite3.Connection, start_time: datetime, end_time: datetime
    ) -> Dict[str, int]:
        """レベル別統計を取得"""
        cursor = conn.execute(
            """
            SELECT level, COUNT(*) as count
            FROM log_entries
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY level
        """,
            (start_time.isoformat(), end_time.isoformat()),
        )

        return {row[0]: row[1] for row in cursor.fetchall()}

    def _get_source_statistics(
        self, conn: sqlite3.Connection, start_time: datetime, end_time: datetime
    ) -> Dict[str, int]:
        """ソース別統計を取得"""
        cursor = conn.execute(
            """
            SELECT source, COUNT(*) as count
            FROM log_entries
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY source
        """,
            (start_time.isoformat(), end_time.isoformat()),
        )

        return {row[0]: row[1] for row in cursor.fetchall()}

    def _get_hourly_statistics(
        self, conn: sqlite3.Connection, start_time: datetime, end_time: datetime
    ) -> Dict[str, int]:
        """時間別統計を取得"""
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

        return {row[0]: row[1] for row in cursor.fetchall()}

    def _get_top_errors(
        self, conn: sqlite3.Connection, start_time: datetime, end_time: datetime
    ) -> list[Dict[str, Any]]:
        """トップエラーを取得"""
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

        return [
            {"message": row[0], "count": row[1]} for row in cursor.fetchall()
        ]

    def get_component_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """コンポーネント別統計を取得"""
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.utcnow()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT component, COUNT(*) as count, 
                           MIN(timestamp) as first_log,
                           MAX(timestamp) as last_log
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY component
                    ORDER BY count DESC
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                components = []
                for row in cursor.fetchall():
                    components.append({
                        "component": row[0],
                        "count": row[1],
                        "first_log": row[2],
                        "last_log": row[3]
                    })

                return {
                    "period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat()
                    },
                    "components": components
                }

        except Exception as e:
            logger.error(f"コンポーネント統計エラー: {e}")
            return {}

    def get_error_trends(
        self, hours: int = 24
    ) -> Dict[str, Any]:
        """エラートレンド分析"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT strftime('%Y-%m-%d %H', timestamp) as hour,
                           level,
                           COUNT(*) as count
                    FROM log_entries
                    WHERE timestamp BETWEEN ? AND ?
                          AND level IN ('ERROR', 'CRITICAL', 'WARNING')
                    GROUP BY strftime('%Y-%m-%d %H', timestamp), level
                    ORDER BY hour, level
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                trends = {}
                for row in cursor.fetchall():
                    hour, level, count = row
                    if hour not in trends:
                        trends[hour] = {}
                    trends[hour][level] = count

                return {
                    "period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "hours": hours
                    },
                    "trends": trends
                }

        except Exception as e:
            logger.error(f"エラートレンド分析エラー: {e}")
            return {}