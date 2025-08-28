#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - データベース操作

このモジュールは、MDMシステムのデータベース操作を担当します。
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import ApprovalStatus, ChangeType, DataGovernanceLevel, MasterDataType
from .models import (
    DataChangeRequest,
    DataGovernancePolicy,
    MasterDataEntity,
    MasterDataHierarchy,
)


class DatabaseOperations:
    """データベース操作クラス
    
    SQLiteデータベースへのCRUD操作を担当します。
    """
    
    def __init__(self, db_path: str = "enterprise_mdm.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # マスターデータエンティティテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS master_data_entities (
                        entity_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        primary_key TEXT NOT NULL,
                        attributes TEXT NOT NULL,
                        metadata TEXT,
                        quality_score REAL DEFAULT 0.0,
                        governance_level TEXT NOT NULL,
                        version INTEGER DEFAULT 1,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        created_by TEXT NOT NULL,
                        updated_by TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        is_golden_record BOOLEAN DEFAULT FALSE,
                        source_systems TEXT,
                        related_entities TEXT
                    )
                """)

                # 変更リクエストテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_change_requests (
                        request_id TEXT PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        proposed_changes TEXT NOT NULL,
                        business_justification TEXT NOT NULL,
                        requested_by TEXT NOT NULL,
                        requested_at DATETIME NOT NULL,
                        approval_status TEXT NOT NULL,
                        approved_by TEXT,
                        approved_at DATETIME,
                        rejection_reason TEXT,
                        impact_assessment TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (entity_id) REFERENCES master_data_entities(entity_id)
                    )
                """)

                # ガバナンスポリシーテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS governance_policies (
                        policy_id TEXT PRIMARY KEY,
                        policy_name TEXT NOT NULL,
                        entity_types TEXT NOT NULL,
                        rules TEXT NOT NULL,
                        governance_level TEXT NOT NULL,
                        requires_approval BOOLEAN DEFAULT TRUE,
                        approval_roles TEXT,
                        retention_period INTEGER,
                        audit_required BOOLEAN DEFAULT TRUE,
                        quality_threshold REAL DEFAULT 80.0,
                        created_by TEXT NOT NULL,
                        created_at DATETIME NOT NULL
                    )
                """)

                # データ階層テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_hierarchies (
                        hierarchy_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        root_entity_id TEXT NOT NULL,
                        parent_child_mapping TEXT NOT NULL,
                        level_definitions TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 変更履歴テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS change_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        old_version INTEGER,
                        new_version INTEGER,
                        changed_fields TEXT,
                        changed_by TEXT NOT NULL,
                        changed_at DATETIME NOT NULL,
                        business_context TEXT,
                        approval_reference TEXT,
                        metadata TEXT
                    )
                """)

                # データ品質履歴テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id TEXT NOT NULL,
                        quality_score REAL NOT NULL,
                        quality_issues TEXT,
                        quality_metrics TEXT,
                        checked_at DATETIME NOT NULL,
                        checked_by TEXT,
                        remediation_actions TEXT,
                        metadata TEXT
                    )
                """)

                # インデックス作成
                self._create_indexes(conn)
                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise

    def _create_indexes(self, conn):
        """インデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON master_data_entities(entity_type, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_entities_golden ON master_data_entities(is_golden_record, entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_requests_status ON data_change_requests(approval_status, requested_at)",
            "CREATE INDEX IF NOT EXISTS idx_history_entity ON change_history(entity_id, changed_at)",
            "CREATE INDEX IF NOT EXISTS idx_quality_entity ON quality_history(entity_id, checked_at)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    # エンティティ操作
    async def save_entity(self, entity: MasterDataEntity):
        """エンティティ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO master_data_entities
                    (entity_id, entity_type, primary_key, attributes, metadata, quality_score,
                     governance_level, version, created_at, updated_at, created_by, updated_by,
                     is_active, is_golden_record, source_systems, related_entities)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.entity_id,
                    entity.entity_type.value,
                    entity.primary_key,
                    json.dumps(entity.attributes, ensure_ascii=False),
                    json.dumps(entity.metadata, ensure_ascii=False),
                    entity.quality_score,
                    entity.governance_level.value,
                    entity.version,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.created_by,
                    entity.updated_by,
                    entity.is_active,
                    entity.is_golden_record,
                    json.dumps(entity.source_systems),
                    json.dumps(entity.related_entities),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"エンティティ保存エラー: {e}")
            raise

    async def get_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """エンティティ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM master_data_entities WHERE entity_id = ?
                """, (entity_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None

                return self._row_to_entity(row)
        except Exception as e:
            self.logger.error(f"エンティティ取得エラー: {e}")
            return None

    async def list_entities(
        self,
        entity_type: Optional[MasterDataType] = None,
        is_active: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[MasterDataEntity]:
        """エンティティ一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = "SELECT * FROM master_data_entities WHERE 1=1"
                params = []

                if entity_type:
                    sql += " AND entity_type = ?"
                    params.append(entity_type.value)

                if is_active is not None:
                    sql += " AND is_active = ?"
                    params.append(is_active)

                sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                return [self._row_to_entity(row) for row in rows]
        except Exception as e:
            self.logger.error(f"エンティティ一覧取得エラー: {e}")
            return []

    def _row_to_entity(self, row) -> MasterDataEntity:
        """データベース行をエンティティに変換"""
        return MasterDataEntity(
            entity_id=row[0],
            entity_type=MasterDataType(row[1]),
            primary_key=row[2],
            attributes=json.loads(row[3]),
            metadata=json.loads(row[4]) if row[4] else {},
            quality_score=row[5],
            governance_level=DataGovernanceLevel(row[6]),
            version=row[7],
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
            created_by=row[10],
            updated_by=row[11],
            is_active=bool(row[12]),
            is_golden_record=bool(row[13]),
            source_systems=json.loads(row[14]) if row[14] else [],
            related_entities=json.loads(row[15]) if row[15] else [],
        )

    # 変更リクエスト操作
    async def save_change_request(self, request: DataChangeRequest):
        """変更リクエスト保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO data_change_requests
                    (request_id, entity_id, change_type, proposed_changes, business_justification,
                     requested_by, requested_at, approval_status, approved_by, approved_at,
                     rejection_reason, impact_assessment, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request.request_id,
                    request.entity_id,
                    request.change_type.value,
                    json.dumps(request.proposed_changes, ensure_ascii=False),
                    request.business_justification,
                    request.requested_by,
                    request.requested_at.isoformat(),
                    request.approval_status.value,
                    request.approved_by,
                    request.approved_at.isoformat() if request.approved_at else None,
                    request.rejection_reason,
                    json.dumps(request.impact_assessment, ensure_ascii=False),
                    json.dumps(request.metadata, ensure_ascii=False),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"変更リクエスト保存エラー: {e}")
            raise

    async def get_change_request(self, request_id: str) -> Optional[DataChangeRequest]:
        """変更リクエスト取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM data_change_requests WHERE request_id = ?
                """, (request_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None

                return DataChangeRequest(
                    request_id=row[0],
                    entity_id=row[1],
                    change_type=ChangeType(row[2]),
                    proposed_changes=json.loads(row[3]),
                    business_justification=row[4],
                    requested_by=row[5],
                    requested_at=datetime.fromisoformat(row[6]),
                    approval_status=ApprovalStatus(row[7]),
                    approved_by=row[8],
                    approved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    rejection_reason=row[10],
                    impact_assessment=json.loads(row[11]) if row[11] else {},
                    metadata=json.loads(row[12]) if row[12] else {},
                )
        except Exception as e:
            self.logger.error(f"変更リクエスト取得エラー: {e}")
            return None

    # ガバナンスポリシー操作
    async def save_governance_policy(self, policy: DataGovernancePolicy):
        """ガバナンスポリシー保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO governance_policies
                    (policy_id, policy_name, entity_types, rules, governance_level,
                     requires_approval, approval_roles, retention_period, audit_required,
                     quality_threshold, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    policy.policy_id,
                    policy.policy_name,
                    json.dumps([et.value for et in policy.entity_types]),
                    json.dumps(policy.rules, ensure_ascii=False),
                    policy.governance_level.value,
                    policy.requires_approval,
                    json.dumps(policy.approval_roles),
                    policy.retention_period,
                    policy.audit_required,
                    policy.quality_threshold,
                    policy.created_by,
                    policy.created_at.isoformat(),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"ガバナンスポリシー保存エラー: {e}")
            raise

    # 階層操作
    async def save_hierarchy(self, hierarchy: MasterDataHierarchy):
        """階層保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO data_hierarchies
                    (hierarchy_id, name, entity_type, root_entity_id, parent_child_mapping,
                     level_definitions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    hierarchy.hierarchy_id,
                    hierarchy.name,
                    hierarchy.entity_type.value,
                    hierarchy.root_entity_id,
                    json.dumps(hierarchy.parent_child_mapping, ensure_ascii=False),
                    json.dumps(hierarchy.level_definitions, ensure_ascii=False),
                    json.dumps(hierarchy.metadata, ensure_ascii=False),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"階層保存エラー: {e}")
            raise

    # 履歴操作
    async def record_change_history(
        self,
        entity_id: str,
        change_type: ChangeType,
        old_version: Optional[int],
        new_version: int,
        changed_fields: List[str],
        changed_by: str,
        changed_at: datetime,
        business_context: str,
        metadata: Dict[str, Any],
        approval_reference: Optional[str] = None
    ):
        """変更履歴記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO change_history
                    (entity_id, change_type, old_version, new_version, changed_fields,
                     changed_by, changed_at, business_context, approval_reference, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    change_type.value,
                    old_version,
                    new_version,
                    json.dumps(changed_fields),
                    changed_by,
                    changed_at.isoformat(),
                    business_context,
                    approval_reference,
                    json.dumps(metadata, ensure_ascii=False),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"変更履歴記録エラー: {e}")
            raise

    async def record_quality_history(
        self,
        entity_id: str,
        quality_score: float,
        quality_issues: List[str],
        quality_metrics: Dict[str, Any],
        checked_at: datetime,
        checked_by: Optional[str] = None,
        remediation_actions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """品質履歴記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO quality_history
                    (entity_id, quality_score, quality_issues, quality_metrics,
                     checked_at, checked_by, remediation_actions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    quality_score,
                    json.dumps(quality_issues, ensure_ascii=False),
                    json.dumps(quality_metrics, ensure_ascii=False),
                    checked_at.isoformat(),
                    checked_by,
                    json.dumps(remediation_actions or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"品質履歴記録エラー: {e}")
            raise

    # カタログ・統計操作
    async def get_catalog_data(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """データカタログ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if entity_type:
                    cursor = conn.execute("""
                        SELECT entity_type, COUNT(*) as total_count,
                               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_count,
                               SUM(CASE WHEN is_golden_record THEN 1 ELSE 0 END) as golden_count,
                               AVG(quality_score) as avg_quality,
                               MIN(created_at) as earliest_created,
                               MAX(updated_at) as latest_updated
                        FROM master_data_entities
                        WHERE entity_type = ?
                        GROUP BY entity_type
                    """, (entity_type.value,))
                else:
                    cursor = conn.execute("""
                        SELECT entity_type, COUNT(*) as total_count,
                               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_count,
                               SUM(CASE WHEN is_golden_record THEN 1 ELSE 0 END) as golden_count,
                               AVG(quality_score) as avg_quality,
                               MIN(created_at) as earliest_created,
                               MAX(updated_at) as latest_updated
                        FROM master_data_entities
                        GROUP BY entity_type
                    """)

                catalog_data = []
                for row in cursor.fetchall():
                    catalog_data.append({
                        "entity_type": row[0],
                        "total_count": row[1],
                        "active_count": row[2],
                        "golden_records": row[3],
                        "average_quality": round(row[4], 2) if row[4] else 0,
                        "earliest_created": row[5],
                        "latest_updated": row[6],
                    })

                return {
                    "catalog": catalog_data,
                    "total_entity_types": len(catalog_data),
                    "generated_at": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(f"データカタログ取得エラー: {e}")
            return {"error": str(e)}

    async def get_entity_lineage(self, entity_id: str) -> Dict[str, Any]:
        """エンティティデータ系譜取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 変更履歴取得
                cursor = conn.execute("""
                    SELECT change_type, old_version, new_version, changed_fields,
                           changed_by, changed_at, business_context, metadata
                    FROM change_history
                    WHERE entity_id = ?
                    ORDER BY changed_at DESC
                """, (entity_id,))

                change_history = []
                for row in cursor.fetchall():
                    change_history.append({
                        "change_type": row[0],
                        "old_version": row[1],
                        "new_version": row[2],
                        "changed_fields": json.loads(row[3]) if row[3] else [],
                        "changed_by": row[4],
                        "changed_at": row[5],
                        "business_context": row[6],
                        "metadata": json.loads(row[7]) if row[7] else {},
                    })

                # 品質履歴取得
                cursor = conn.execute("""
                    SELECT quality_score, quality_issues, checked_at, checked_by
                    FROM quality_history
                    WHERE entity_id = ?
                    ORDER BY checked_at DESC LIMIT 20
                """, (entity_id,))

                quality_history = []
                for row in cursor.fetchall():
                    quality_history.append({
                        "quality_score": row[0],
                        "quality_issues": json.loads(row[1]) if row[1] else [],
                        "checked_at": row[2],
                        "checked_by": row[3],
                    })

                return {
                    "entity_id": entity_id,
                    "change_history": change_history,
                    "quality_history": quality_history,
                    "retrieved_at": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(f"データ系譜取得エラー: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    async def cleanup_database(self):
        """データベース最適化・クリーンアップ"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # VACUUMでデータベース最適化
                conn.execute("VACUUM")
                
                # 統計情報更新
                conn.execute("ANALYZE")
                
                self.logger.info("データベースクリーンアップ完了")
        except Exception as e:
            self.logger.error(f"データベースクリーンアップエラー: {e}")
            raise