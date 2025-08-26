#!/usr/bin/env python3
"""
マスターデータ データベース管理
データベーステーブル作成・管理
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import (
    DataDomain,
    DataElement,
    DataGovernancePolicy,
    DataLineage,
    DataSteward,
    MasterDataEntity,
    MasterDataStatus,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DatabaseManager:
    """データベース管理クラス"""

    def __init__(self, db_path: Path, enable_audit: bool = True):
        """初期化"""
        self.db_path = db_path
        self.enable_audit = enable_audit
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # データ要素テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_elements (
                    element_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    data_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    classification TEXT,
                    is_pii INTEGER DEFAULT 0,
                    business_rules TEXT,
                    validation_rules TEXT,
                    source_systems TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT,
                    metadata TEXT
                )
            """
            )

            # マスターエンティティテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS master_entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    primary_key TEXT NOT NULL,
                    attributes TEXT,
                    domain TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    version INTEGER DEFAULT 1,
                    source_system TEXT,
                    data_quality_score REAL DEFAULT 1.0,
                    last_validated TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    steward TEXT,
                    lineage TEXT,
                    relationships TEXT,
                    metadata TEXT
                )
            """
            )

            # データスチュワードテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_stewards (
                    steward_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    domains TEXT,
                    responsibilities TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )

            # ガバナンスポリシーテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS governance_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    domain TEXT NOT NULL,
                    policy_type TEXT NOT NULL,
                    rules TEXT,
                    enforcement_level TEXT,
                    effective_date TEXT NOT NULL,
                    expiration_date TEXT,
                    owner TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    metadata TEXT
                )
            """
            )

            # データ系譜テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    source_entity TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    transformation TEXT,
                    transformation_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )

            # 監査ログテーブル（監査有効時）
            if self.enable_audit:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        audit_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        old_values TEXT,
                        new_values TEXT,
                        user_id TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                """
                )

            self._create_indexes(conn)

    def _create_indexes(self, conn):
        """インデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON master_entities(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_domain ON master_entities(domain)",
            "CREATE INDEX IF NOT EXISTS idx_entities_status ON master_entities(status)",
            "CREATE INDEX IF NOT EXISTS idx_lineage_source ON data_lineage(source_entity)",
            "CREATE INDEX IF NOT EXISTS idx_lineage_target ON data_lineage(target_entity)",
        ]

        if self.enable_audit:
            indexes.extend([
                "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
            ])

        for index_sql in indexes:
            conn.execute(index_sql)

    async def save_entity(self, entity: MasterDataEntity):
        """エンティティデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO master_entities
                (entity_id, entity_type, primary_key, attributes, domain, status,
                 version, source_system, data_quality_score, last_validated,
                 created_at, updated_at, steward, lineage, relationships, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity.entity_id,
                    entity.entity_type,
                    entity.primary_key,
                    json.dumps(entity.attributes, default=str),
                    entity.domain.value,
                    entity.status.value,
                    entity.version,
                    entity.source_system,
                    entity.data_quality_score,
                    entity.last_validated.isoformat(),
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.steward,
                    json.dumps(entity.lineage),
                    json.dumps(entity.relationships),
                    json.dumps(entity.metadata, default=str),
                ),
            )

    async def update_entity(self, entity: MasterDataEntity):
        """エンティティデータベース更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE master_entities SET
                    attributes = ?, status = ?, version = ?, data_quality_score = ?,
                    last_validated = ?, updated_at = ?, steward = ?, lineage = ?,
                    relationships = ?, metadata = ?
                WHERE entity_id = ?
            """,
                (
                    json.dumps(entity.attributes, default=str),
                    entity.status.value,
                    entity.version,
                    entity.data_quality_score,
                    entity.last_validated.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.steward,
                    json.dumps(entity.lineage),
                    json.dumps(entity.relationships),
                    json.dumps(entity.metadata, default=str),
                    entity.entity_id,
                ),
            )

    async def get_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """エンティティ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_id, entity_type, primary_key, attributes, domain, status,
                           version, source_system, data_quality_score, last_validated,
                           created_at, updated_at, steward, lineage, relationships, metadata
                    FROM master_entities WHERE entity_id = ?
                """,
                    (entity_id,),
                )

                row = cursor.fetchone()
                if row:
                    return MasterDataEntity(
                        entity_id=row[0],
                        entity_type=row[1],
                        primary_key=row[2],
                        attributes=json.loads(row[3]) if row[3] else {},
                        domain=DataDomain(row[4]),
                        status=MasterDataStatus(row[5]),
                        version=row[6],
                        source_system=row[7],
                        data_quality_score=row[8],
                        last_validated=datetime.fromisoformat(row[9]),
                        created_at=datetime.fromisoformat(row[10]),
                        updated_at=datetime.fromisoformat(row[11]),
                        steward=row[12],
                        lineage=json.loads(row[13]) if row[13] else [],
                        relationships=json.loads(row[14]) if row[14] else {},
                        metadata=json.loads(row[15]) if row[15] else {},
                    )
                return None

        except Exception as e:
            logger.error(f"エンティティ取得エラー: {e}")
            return None

    async def find_existing_entity(
        self, entity_type: str, primary_key: str
    ) -> Optional[MasterDataEntity]:
        """既存エンティティ検索"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_id FROM master_entities
                    WHERE entity_type = ? AND primary_key = ? AND status = 'active'
                    ORDER BY version DESC LIMIT 1
                """,
                    (entity_type, primary_key),
                )

                row = cursor.fetchone()
                if row:
                    return await self.get_entity(row[0])

                return None

        except Exception as e:
            logger.error(f"既存エンティティ検索エラー: {e}")
            return None

    async def search_entities(
        self,
        entity_type: Optional[str] = None,
        domain: Optional[DataDomain] = None,
        status: Optional[MasterDataStatus] = None,
        primary_key_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[MasterDataEntity]:
        """エンティティ検索"""
        try:
            query_conditions = []
            query_params = []

            if entity_type:
                query_conditions.append("entity_type = ?")
                query_params.append(entity_type)

            if domain:
                query_conditions.append("domain = ?")
                query_params.append(domain.value)

            if status:
                query_conditions.append("status = ?")
                query_params.append(status.value)

            if primary_key_pattern:
                query_conditions.append("primary_key LIKE ?")
                query_params.append(f"%{primary_key_pattern}%")

            where_clause = ""
            if query_conditions:
                where_clause = "WHERE " + " AND ".join(query_conditions)

            query = f"""
                SELECT entity_id, entity_type, primary_key, attributes, domain, status,
                       version, source_system, data_quality_score, last_validated,
                       created_at, updated_at, steward, lineage, relationships, metadata
                FROM master_entities
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
            """
            query_params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, query_params)

                entities = []
                for row in cursor.fetchall():
                    entity = MasterDataEntity(
                        entity_id=row[0],
                        entity_type=row[1],
                        primary_key=row[2],
                        attributes=json.loads(row[3]) if row[3] else {},
                        domain=DataDomain(row[4]),
                        status=MasterDataStatus(row[5]),
                        version=row[6],
                        source_system=row[7],
                        data_quality_score=row[8],
                        last_validated=datetime.fromisoformat(row[9]),
                        created_at=datetime.fromisoformat(row[10]),
                        updated_at=datetime.fromisoformat(row[11]),
                        steward=row[12],
                        lineage=json.loads(row[13]) if row[13] else [],
                        relationships=json.loads(row[14]) if row[14] else {},
                        metadata=json.loads(row[15]) if row[15] else {},
                    )
                    entities.append(entity)

                return entities

        except Exception as e:
            logger.error(f"エンティティ検索エラー: {e}")
            return []

    async def save_lineage(self, lineage: DataLineage):
        """データ系譜保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_lineage
                (lineage_id, source_entity, target_entity, transformation,
                 transformation_type, confidence, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    lineage.lineage_id,
                    lineage.source_entity,
                    lineage.target_entity,
                    lineage.transformation,
                    lineage.transformation_type,
                    lineage.confidence,
                    lineage.created_at.isoformat(),
                    json.dumps(lineage.metadata),
                ),
            )

    async def record_audit_log(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        user_id: str = "system",
    ):
        """監査ログ記録"""
        if not self.enable_audit:
            return

        audit_id = f"audit_{int(datetime.now().timestamp())}_{hash(f'{entity_type}_{entity_id}_{operation}')}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log
                (audit_id, entity_type, entity_id, operation, old_values,
                 new_values, user_id, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    audit_id,
                    entity_type,
                    entity_id,
                    operation,
                    json.dumps(old_values, default=str) if old_values else None,
                    json.dumps(new_values, default=str) if new_values else None,
                    user_id,
                    datetime.utcnow().isoformat(),
                    json.dumps({}),
                ),
            )

    async def get_quality_metrics(self) -> Dict[str, Any]:
        """品質メトリクス取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 全体品質統計
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entities,
                        AVG(data_quality_score) as avg_quality_score,
                        MIN(data_quality_score) as min_quality_score,
                        MAX(data_quality_score) as max_quality_score
                    FROM master_entities WHERE status = 'active'
                """
                )

                stats = cursor.fetchone()

                # ドメイン別品質統計
                cursor = conn.execute(
                    """
                    SELECT
                        domain,
                        COUNT(*) as count,
                        AVG(data_quality_score) as avg_score
                    FROM master_entities WHERE status = 'active'
                    GROUP BY domain
                """
                )

                domain_stats = {}
                for row in cursor.fetchall():
                    domain_stats[row[0]] = {
                        "entity_count": row[1],
                        "average_quality_score": row[2],
                    }

                return {
                    "total_entities": stats[0] or 0,
                    "average_quality_score": stats[1] or 0,
                    "min_quality_score": stats[2] or 0,
                    "max_quality_score": stats[3] or 0,
                    "domain_statistics": domain_stats,
                    "calculated_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"品質メトリクス取得エラー: {e}")
            return {}

    async def get_lineage_summary(self) -> Dict[str, Any]:
        """データ系譜サマリー取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 系譜統計
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_lineages,
                        COUNT(DISTINCT source_entity) as unique_sources,
                        COUNT(DISTINCT target_entity) as unique_targets,
                        AVG(confidence) as avg_confidence
                    FROM data_lineage
                """
                )

                stats = cursor.fetchone()

                # 変換タイプ別統計
                cursor = conn.execute(
                    """
                    SELECT transformation_type, COUNT(*) as count
                    FROM data_lineage
                    GROUP BY transformation_type
                """
                )

                transformation_stats = {}
                for row in cursor.fetchall():
                    transformation_stats[row[0]] = row[1]

                return {
                    "total_lineages": stats[0] or 0,
                    "unique_sources": stats[1] or 0,
                    "unique_targets": stats[2] or 0,
                    "average_confidence": stats[3] or 0,
                    "transformation_types": transformation_stats,
                }

        except Exception as e:
            logger.error(f"データ系譜サマリー取得エラー: {e}")
            return {}

    async def get_recent_activities(self) -> List[Dict[str, Any]]:
        """最近の活動取得"""
        activities = []

        try:
            if self.enable_audit:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT entity_type, entity_id, operation, user_id, timestamp
                        FROM audit_log
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """
                    )

                    for row in cursor.fetchall():
                        activities.append(
                            {
                                "entity_type": row[0],
                                "entity_id": row[1],
                                "operation": row[2],
                                "user_id": row[3],
                                "timestamp": row[4],
                            }
                        )

            return activities

        except Exception as e:
            logger.error(f"最近の活動取得エラー: {e}")
            return []

    async def get_domain_distribution(self) -> Dict[str, int]:
        """ドメイン分布取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT domain, COUNT(*) as count
                    FROM master_entities
                    WHERE status = 'active'
                    GROUP BY domain
                """
                )

                distribution = {}
                for row in cursor.fetchall():
                    distribution[row[0]] = row[1]

                return distribution

        except Exception as e:
            logger.error(f"ドメイン分布取得エラー: {e}")
            return {}

    async def cleanup_old_audit_logs(self, retention_days: int):
        """古い監査ログクリーンアップ"""
        if not self.enable_audit:
            return 0

        try:
            retention_date = datetime.utcnow().timestamp() - (retention_days * 24 * 3600)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM audit_log
                    WHERE timestamp < ?
                """,
                    (datetime.fromtimestamp(retention_date).isoformat(),),
                )
                return cursor.rowcount

        except Exception as e:
            logger.error(f"監査ログクリーンアップエラー: {e}")
            return 0