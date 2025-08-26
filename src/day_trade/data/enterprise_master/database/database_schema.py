#!/usr/bin/env python3
"""
データベーススキーマ管理モジュール

データベースの初期化、テーブル作成、インデックス管理を担当します。
"""

import logging
import sqlite3
from typing import Any, Dict


class DatabaseSchema:
    """データベーススキーマ管理クラス
    
    データベースの構造定義と初期化を担当します。
    """
    
    def __init__(self, db_path: str = "enterprise_mdm.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._create_master_data_entities_table(conn)
                self._create_change_requests_table(conn)
                self._create_governance_policies_table(conn)
                self._create_hierarchies_table(conn)
                self._create_history_tables(conn)
                self._create_indexes(conn)
                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise

    def _create_master_data_entities_table(self, conn):
        """マスターデータエンティティテーブル作成"""
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

    def _create_change_requests_table(self, conn):
        """変更リクエストテーブル作成"""
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

    def _create_governance_policies_table(self, conn):
        """ガバナンスポリシーテーブル作成"""
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

    def _create_hierarchies_table(self, conn):
        """データ階層テーブル作成"""
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

    def _create_history_tables(self, conn):
        """履歴テーブル群作成"""
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

    def _create_indexes(self, conn):
        """インデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON master_data_entities(entity_type, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_entities_golden ON master_data_entities(is_golden_record, entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_requests_status ON data_change_requests(approval_status, requested_at)",
            "CREATE INDEX IF NOT EXISTS idx_history_entity ON change_history(entity_id, changed_at)",
            "CREATE INDEX IF NOT EXISTS idx_quality_entity ON quality_history(entity_id, checked_at)",
            "CREATE INDEX IF NOT EXISTS idx_entities_primary ON master_data_entities(primary_key, entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_requests_entity ON data_change_requests(entity_id, change_type)",
            "CREATE INDEX IF NOT EXISTS idx_hierarchies_type ON data_hierarchies(entity_type, name)",
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"インデックス作成警告: {e}")

    async def cleanup_database(self):
        """データベース清理"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 古い履歴データのクリーンアップ（例：1年以上前のデータ）
                cleanup_queries = [
                    "DELETE FROM change_history WHERE changed_at < datetime('now', '-1 year')",
                    "DELETE FROM quality_history WHERE checked_at < datetime('now', '-1 year')",
                    "DELETE FROM data_change_requests WHERE created_at < datetime('now', '-2 years') AND approval_status IN ('approved', 'rejected')",
                ]
                
                for query in cleanup_queries:
                    conn.execute(query)
                
                # VACUUM操作でデータベースファイル最適化
                conn.execute("VACUUM")
                conn.commit()
                
                self.logger.info("データベース清理完了")

        except Exception as e:
            self.logger.error(f"データベース清理エラー: {e}")
            raise

    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # テーブル情報
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # データベースサイズ情報
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = page_count * page_size
                
                # 各テーブルの行数
                table_counts = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]
                
                return {
                    "database_path": self.db_path,
                    "database_size_bytes": db_size,
                    "total_tables": len(tables),
                    "tables": tables,
                    "table_row_counts": table_counts,
                }

        except Exception as e:
            self.logger.error(f"データベース情報取得エラー: {e}")
            return {
                "error": str(e),
                "database_path": self.db_path,
            }

    def validate_schema(self) -> Dict[str, Any]:
        """データベーススキーマ検証"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "missing_tables": [],
                "missing_indexes": [],
            }
            
            expected_tables = [
                "master_data_entities",
                "data_change_requests", 
                "governance_policies",
                "data_hierarchies",
                "change_history",
                "quality_history"
            ]
            
            expected_indexes = [
                "idx_entities_type",
                "idx_entities_golden",
                "idx_requests_status",
                "idx_history_entity",
                "idx_quality_entity"
            ]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # テーブル存在確認
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                for table in expected_tables:
                    if table not in existing_tables:
                        validation_result["is_valid"] = False
                        validation_result["missing_tables"].append(table)
                        validation_result["errors"].append(f"必須テーブル '{table}' が見つかりません")
                
                # インデックス存在確認
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                existing_indexes = {row[0] for row in cursor.fetchall() if row[0] is not None}
                
                for index in expected_indexes:
                    if index not in existing_indexes:
                        validation_result["missing_indexes"].append(index)
                        validation_result["warnings"].append(f"推奨インデックス '{index}' が見つかりません")
            
            return validation_result

        except Exception as e:
            self.logger.error(f"スキーマ検証エラー: {e}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
            }