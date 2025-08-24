#!/usr/bin/env python3
"""
データ鮮度監視システム用データベース操作
SQLiteデータベースの初期化、CRUD操作を提供します
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from .models import (
    DataAlert,
    DataSourceConfig,
    DataSourceState,
    FreshnessCheck,
    IntegrityCheck,
    SLAMetrics
)


class DatabaseOperations:
    """データベース操作クラス
    
    SQLiteデータベースの初期化、データの保存・取得操作を提供します。
    鮮度チェック、整合性チェック、アラート、SLAメトリクスのテーブルを管理します。
    """
    
    def __init__(self, db_path: str = "data_freshness_monitor.db"):
        """データベース操作を初期化
        
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """データベース初期化
        
        必要なテーブルとインデックスを作成します。
        """
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_freshness_source_time "
                "ON freshness_checks(source_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_integrity_source_time "
                "ON integrity_checks(source_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sla_source_period "
                "ON sla_metrics(source_id, period_start, period_end)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_source_severity "
                "ON data_alerts(source_id, severity, timestamp)"
            )
            
            conn.commit()
    
    async def save_freshness_check(self, check: FreshnessCheck):
        """鮮度チェック結果保存
        
        Args:
            check: 鮮度チェック結果
        """
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
        """整合性チェック結果保存
        
        Args:
            check: 整合性チェック結果
        """
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
        """SLAメトリクス保存
        
        Args:
            metrics: SLAメトリクス
        """
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
        """アラート保存
        
        Args:
            alert: データアラート
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO data_alerts
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
    
    async def update_source_state(
        self, 
        source_id: str, 
        state: DataSourceState
    ):
        """データソース状態更新
        
        Args:
            source_id: データソースID
            state: データソース状態
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO data_source_states
                    (source_id, last_check, current_status, consecutive_failures,
                     last_success, recovery_attempts, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        source_id,
                        state.last_check.isoformat() if state.last_check else None,
                        state.current_status,
                        state.consecutive_failures,
                        state.last_success.isoformat() if state.last_success else None,
                        state.recovery_attempts,
                        json.dumps(state.metadata, ensure_ascii=False),
                        current_time.isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"データソース状態更新エラー: {e}")
    
    def get_source_state(self, source_id: str) -> Optional[DataSourceState]:
        """データソース状態取得
        
        Args:
            source_id: データソースID
            
        Returns:
            データソース状態、存在しない場合はNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT last_check, current_status, consecutive_failures,
                           last_success, recovery_attempts, metadata, updated_at
                    FROM data_source_states WHERE source_id = ?
                """,
                    (source_id,),
                )
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                return DataSourceState(
                    source_id=source_id,
                    last_check=datetime.fromisoformat(result[0]) if result[0] else None,
                    current_status=result[1] or "unknown",
                    consecutive_failures=result[2] or 0,
                    last_success=datetime.fromisoformat(result[3]) if result[3] else None,
                    recovery_attempts=result[4] or 0,
                    metadata=json.loads(result[5]) if result[5] else {},
                    updated_at=datetime.fromisoformat(result[6]) if result[6] else None,
                )
                
        except Exception as e:
            self.logger.error(f"データソース状態取得エラー: {e}")
            return None
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決
        
        Args:
            alert_id: アラートID
            resolution_notes: 解決ノート
        """
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
    
    def get_historical_record_counts(
        self, 
        source_id: str, 
        hours: int = 24, 
        limit: int = 10
    ) -> List[int]:
        """履歴レコード数取得
        
        Args:
            source_id: データソースID
            hours: 過去何時間分を取得するか
            limit: 最大取得件数
            
        Returns:
            履歴レコード数のリスト
        """
        try:
            from datetime import timedelta
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT record_count FROM freshness_checks
                    WHERE source_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (source_id, start_time.isoformat(), limit),
                )
                
                return [
                    row[0] for row in cursor.fetchall() 
                    if row[0] is not None
                ]
                
        except Exception as e:
            self.logger.error(f"履歴レコード数取得エラー: {e}")
            return []