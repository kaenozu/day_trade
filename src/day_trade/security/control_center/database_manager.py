#!/usr/bin/env python3
"""
セキュリティ管制センター - データベース管理
"""

import json
import logging
import sqlite3
from typing import Any, Dict, List

from .enums import SecurityLevel, ThreatCategory, IncidentStatus
from .models import SecurityThreat, SecurityIncident


class DatabaseManager:
    """データベース管理クラス"""

    def __init__(self, db_path: str = "security_control_center.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_threats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    affected_assets TEXT,
                    indicators TEXT,
                    metadata TEXT,
                    detected_at DATETIME NOT NULL,
                    expires_at DATETIME,
                    status TEXT DEFAULT 'active'
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_incidents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assigned_to TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    response_actions TEXT,
                    impact_assessment TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    requirements_met INTEGER NOT NULL,
                    total_requirements INTEGER NOT NULL,
                    violations TEXT,
                    recommendations TEXT,
                    checked_at DATETIME NOT NULL
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_threats_severity ON security_threats(severity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_threats_category ON security_threats(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_status ON security_incidents(status)"
            )

            conn.commit()

    async def save_threat(self, threat: SecurityThreat):
        """脅威をデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_threats
                    (id, title, description, category, severity, source, affected_assets,
                     indicators, metadata, detected_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        threat.id,
                        threat.title,
                        threat.description,
                        threat.category.value,
                        threat.severity.value,
                        threat.source,
                        json.dumps(threat.affected_assets),
                        json.dumps(threat.indicators),
                        json.dumps(threat.metadata),
                        threat.detected_at.isoformat(),
                        threat.expires_at.isoformat() if threat.expires_at else None,
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"脅威保存エラー: {e}")
            raise

    async def save_incident(self, incident: SecurityIncident):
        """インシデントをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_incidents
                    (id, title, description, severity, status, assigned_to,
                     created_at, updated_at, resolved_at, response_actions, impact_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        incident.id,
                        incident.title,
                        incident.description,
                        incident.severity.value,
                        incident.status.value,
                        incident.assigned_to,
                        incident.created_at.isoformat(),
                        incident.updated_at.isoformat(),
                        (
                            incident.resolved_at.isoformat()
                            if incident.resolved_at
                            else None
                        ),
                        json.dumps(incident.response_actions),
                        json.dumps(incident.impact_assessment),
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"インシデント保存エラー: {e}")
            raise

    async def cleanup_expired_threats(self, current_time):
        """期限切れ脅威のクリーンアップ"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM security_threats WHERE expires_at < ? AND status = 'active'",
                    (current_time.isoformat(),),
                )
                expired_threat_ids = [row[0] for row in cursor.fetchall()]

                if expired_threat_ids:
                    conn.execute(
                        "UPDATE security_threats SET status = 'expired' WHERE expires_at < ?",
                        (current_time.isoformat(),),
                    )
                    conn.commit()

                return expired_threat_ids

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return []