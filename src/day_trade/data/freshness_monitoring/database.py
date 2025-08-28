#!/usr/bin/env python3
"""
データベース操作モジュール
高度データ鮮度・整合性監視システム用のデータベース初期化と基本操作
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .models import (
    DataAlert,
    DataSourceConfig,
    FreshnessCheck,
    IntegrityCheck,
    SLAMetrics,
)


class DatabaseManager:
    """データベース操作管理クラス"""

    def __init__(self, db_path: str = "data_freshness_monitor.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 鮮度チェックテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS freshness_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_update DATETIME NOT NULL,
                    age_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    quality_score REAL,
                    record_count INTEGER,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 整合性チェックテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    passed BOOLEAN NOT NULL,
                    issues_found TEXT,
                    metrics TEXT,
                    baseline_comparison TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # SLAメトリクステーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sla_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    availability_percent REAL NOT NULL,
                    average_response_time REAL NOT NULL,
                    error_count INTEGER NOT NULL,
                    total_requests INTEGER NOT NULL,
                    uptime_seconds REAL NOT NULL,
                    downtime_seconds REAL NOT NULL,
                    sla_violations INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # アラートテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_alerts (
                    alert_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    resolution_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # データソース状態テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_source_states (
                    source_id TEXT PRIMARY KEY,
                    last_check DATETIME,
                    current_status TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    last_success DATETIME,
                    recovery_attempts INTEGER DEFAULT 0,
                    metadata TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # インデックス作成
            self._create_indexes(conn)
            conn.commit()

    def _create_indexes(self, conn):
        """インデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_freshness_source_time ON freshness_checks(source_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_integrity_source_time ON integrity_checks(source_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sla_source_period ON sla_metrics(source_id, period_start, period_end)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_source_severity ON data_alerts(source_id, severity, timestamp)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    async def save_freshness_check(self, check: FreshnessCheck):
        """鮮度チェック結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO freshness_checks
                    (source_id, timestamp, last_update, age_seconds, status,
                     quality_score, record_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        check.source_id,
                        check.timestamp.isoformat(),
                        check.last_update.isoformat(),
                        check.age_seconds,
                        check.status.value,
                        check.quality_score,
                        check.record_count,
                        json.dumps(check.metadata, ensure_ascii=False),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"鮮度チェック保存エラー: {e}")

    async def save_integrity_check(self, check: IntegrityCheck):
        """整合性チェック結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO integrity_checks
                    (source_id, check_type, timestamp, passed, issues_found,
                     metrics, baseline_comparison)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        check.source_id,
                        check.check_type,
                        check.timestamp.isoformat(),
                        check.passed,
                        json.dumps(check.issues_found, ensure_ascii=False),
                        json.dumps(check.metrics, ensure_ascii=False),
                        (
                            json.dumps(check.baseline_comparison, ensure_ascii=False)
                            if check.baseline_comparison
                            else None
                        ),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"整合性チェック保存エラー: {e}")

    async def save_sla_metrics(self, metrics: SLAMetrics):
        """SLAメトリクス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sla_metrics
                    (source_id, period_start, period_end, availability_percent,
                     average_response_time, error_count, total_requests,
                     uptime_seconds, downtime_seconds, sla_violations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics.source_id,
                        metrics.period_start.isoformat(),
                        metrics.period_end.isoformat(),
                        metrics.availability_percent,
                        metrics.average_response_time,
                        metrics.error_count,
                        metrics.total_requests,
                        metrics.uptime_seconds,
                        metrics.downtime_seconds,
                        metrics.sla_violations,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"SLAメトリクス保存エラー: {e}")

    async def save_alert(self, alert: DataAlert):
        """アラート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_alerts
                    (alert_id, source_id, severity, alert_type, message, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.source_id,
                        alert.severity.value,
                        alert.alert_type,
                        alert.message,
                        alert.timestamp.isoformat(),
                        json.dumps(alert.metadata, ensure_ascii=False),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")

    def initialize_data_source_state(self, config: DataSourceConfig):
        """データソース状態初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_source_states
                (source_id, current_status, metadata, updated_at)
                VALUES (?, ?, ?, ?)
            """,
                (
                    config.source_id,
                    "unknown",
                    json.dumps(
                        {
                            "source_type": config.source_type.value,
                            "monitoring_level": config.monitoring_level.value,
                            "sla_target": config.sla_target,
                        },
                        ensure_ascii=False,
                    ),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_source_state(self, source_id: str) -> Optional[tuple]:
        """データソース状態取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT consecutive_failures, recovery_attempts, current_status
                FROM data_source_states WHERE source_id = ?
            """,
                (source_id,),
            )
            return cursor.fetchone()

    def update_source_state(
        self,
        source_id: str,
        current_status: str,
        consecutive_failures: int,
        last_success: Optional[datetime] = None,
    ):
        """データソース状態更新"""
        try:
            current_time = datetime.now(timezone.utc)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE data_source_states
                    SET last_check = ?, current_status = ?, consecutive_failures = ?,
                        last_success = COALESCE(?, last_success), updated_at = ?
                    WHERE source_id = ?
                """,
                    (
                        current_time.isoformat(),
                        current_status,
                        consecutive_failures,
                        last_success.isoformat() if last_success else None,
                        current_time.isoformat(),
                        source_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"データソース状態更新エラー ({source_id}): {e}")

    def update_recovery_attempts(self, source_id: str):
        """回復試行回数更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE data_source_states
                    SET recovery_attempts = recovery_attempts + 1, updated_at = ?
                    WHERE source_id = ?
                """,
                    (datetime.now(timezone.utc).isoformat(), source_id),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"回復試行回数更新エラー ({source_id}): {e}")

    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE data_alerts
                    SET resolved = TRUE, resolved_at = ?, resolution_notes = ?
                    WHERE alert_id = ?
                """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        resolution_notes,
                        alert_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"アラート解決エラー: {e}")