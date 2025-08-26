#!/usr/bin/env python3
"""
データベース管理モジュール

データバージョン管理システムのSQLiteデータベース初期化、
テーブル管理、および基本的なデータベース操作を提供します。

Classes:
    DatabaseManager: データベース管理メインクラス
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import (
    ConflictResolution,
    DataBranch,
    DataStatus,
    DataTag,
    DataVersion,
    VersionConflict,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DatabaseManager:
    """データベース管理クラス
    
    SQLiteデータベースの初期化、テーブル作成、基本的なCRUD操作を提供します。
    バージョン、ブランチ、タグ、競合情報の管理を行います。
    """

    def __init__(self, db_path: Path):
        """DatabaseManagerの初期化
        
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = db_path
        self._initialize_database()
        logger.info(f"データベースマネージャー初期化完了: {db_path}")

    def _initialize_database(self) -> None:
        """データベースとテーブルを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            self._create_versions_table(conn)
            self._create_branches_table(conn)
            self._create_tags_table(conn)
            self._create_conflicts_table(conn)
            self._create_indexes(conn)
            self._ensure_default_branch(conn)

    def _create_versions_table(self, conn: sqlite3.Connection) -> None:
        """バージョン管理テーブルの作成"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                parent_version TEXT,
                branch TEXT NOT NULL,
                tag TEXT,
                author TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT,
                data_hash TEXT NOT NULL,
                metadata TEXT,
                status TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                file_count INTEGER DEFAULT 0,
                checksum TEXT,
                FOREIGN KEY (branch) REFERENCES branches (branch_name)
            )
            """
        )

    def _create_branches_table(self, conn: sqlite3.Connection) -> None:
        """ブランチ管理テーブルの作成"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                branch_name TEXT PRIMARY KEY,
                created_from TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                description TEXT,
                is_protected INTEGER DEFAULT 0,
                latest_version TEXT,
                merge_policy TEXT
            )
            """
        )

    def _create_tags_table(self, conn: sqlite3.Connection) -> None:
        """タグ管理テーブルの作成"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                tag_name TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                description TEXT,
                is_release INTEGER DEFAULT 0,
                FOREIGN KEY (version_id) REFERENCES versions (version_id)
            )
            """
        )

    def _create_conflicts_table(self, conn: sqlite3.Connection) -> None:
        """競合管理テーブルの作成"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conflicts (
                conflict_id TEXT PRIMARY KEY,
                source_version TEXT NOT NULL,
                target_version TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                file_path TEXT,
                description TEXT,
                resolution_strategy TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_by TEXT,
                resolved_at TEXT,
                FOREIGN KEY (source_version) REFERENCES versions (version_id),
                FOREIGN KEY (target_version) REFERENCES versions (version_id)
            )
            """
        )

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """パフォーマンス向上のためのインデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_versions_branch ON versions(branch)",
            "CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON versions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_versions_status ON versions(status)",
            "CREATE INDEX IF NOT EXISTS idx_versions_data_hash ON versions(data_hash)",
            "CREATE INDEX IF NOT EXISTS idx_tags_version_id ON tags(version_id)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_source ON conflicts(source_version)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_target ON conflicts(target_version)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)

    def _ensure_default_branch(self, conn: sqlite3.Connection) -> None:
        """デフォルトのmainブランチを確保"""
        cursor = conn.execute(
            "SELECT COUNT(*) FROM branches WHERE branch_name = ?", ("main",)
        )
        if cursor.fetchone()[0] == 0:
            main_branch = DataBranch(
                branch_name="main",
                created_from="root",
                created_by="system",
                description="メインブランチ",
                is_protected=True,
            )
            self.save_branch(main_branch)
            logger.info("デフォルトmainブランチ作成完了")

    def save_version(self, version: DataVersion) -> None:
        """バージョンをデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO versions
                (version_id, parent_version, branch, tag, author, timestamp, message,
                 data_hash, metadata, status, size_bytes, file_count, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.parent_version,
                    version.branch,
                    version.tag,
                    version.author,
                    version.timestamp.isoformat(),
                    version.message,
                    version.data_hash,
                    json.dumps(version.metadata, default=str),
                    version.status.value,
                    version.size_bytes,
                    version.file_count,
                    version.checksum,
                ),
            )

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """バージョンIDでバージョン情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, parent_version, branch, tag, author, timestamp,
                       message, data_hash, metadata, status, size_bytes, file_count, checksum
                FROM versions WHERE version_id = ?
                """,
                (version_id,),
            )

            row = cursor.fetchone()
            if row:
                return DataVersion(
                    version_id=row[0],
                    parent_version=row[1],
                    branch=row[2],
                    tag=row[3],
                    author=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    message=row[6],
                    data_hash=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                    status=DataStatus(row[9]),
                    size_bytes=row[10],
                    file_count=row[11],
                    checksum=row[12],
                )
            return None

    def update_version(self, version: DataVersion) -> None:
        """バージョン情報を更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE versions
                SET tag = ?, status = ?, metadata = ?
                WHERE version_id = ?
                """,
                (
                    version.tag,
                    version.status.value,
                    json.dumps(version.metadata, default=str),
                    version.version_id,
                ),
            )

    def get_versions_by_branch(
        self, branch: str, limit: int = 50
    ) -> List[DataVersion]:
        """ブランチのバージョン履歴を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, parent_version, branch, tag, author, timestamp,
                       message, data_hash, metadata, status, size_bytes, file_count, checksum
                FROM versions
                WHERE branch = ? AND status != ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (branch, DataStatus.DELETED.value, limit),
            )

            return [
                DataVersion(
                    version_id=row[0],
                    parent_version=row[1],
                    branch=row[2],
                    tag=row[3],
                    author=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    message=row[6],
                    data_hash=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                    status=DataStatus(row[9]),
                    size_bytes=row[10],
                    file_count=row[11],
                    checksum=row[12],
                )
                for row in cursor.fetchall()
            ]

    def find_version_by_hash(
        self, data_hash: str, branch: str
    ) -> Optional[DataVersion]:
        """データハッシュでバージョンを検索"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, parent_version, branch, tag, author, timestamp,
                       message, data_hash, metadata, status, size_bytes, file_count, checksum
                FROM versions
                WHERE data_hash = ? AND branch = ? AND status != ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (data_hash, branch, DataStatus.DELETED.value),
            )

            row = cursor.fetchone()
            if row:
                return DataVersion(
                    version_id=row[0],
                    parent_version=row[1],
                    branch=row[2],
                    tag=row[3],
                    author=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    message=row[6],
                    data_hash=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                    status=DataStatus(row[9]),
                    size_bytes=row[10],
                    file_count=row[11],
                    checksum=row[12],
                )
            return None

    def save_branch(self, branch: DataBranch) -> None:
        """ブランチをデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO branches
                (branch_name, created_from, created_by, created_at, description,
                 is_protected, latest_version, merge_policy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    branch.branch_name,
                    branch.created_from,
                    branch.created_by,
                    branch.created_at.isoformat(),
                    branch.description,
                    1 if branch.is_protected else 0,
                    branch.latest_version,
                    branch.merge_policy.value,
                ),
            )

    def get_branch(self, branch_name: str) -> Optional[DataBranch]:
        """ブランチ名でブランチ情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT branch_name, created_from, created_by, created_at, description,
                       is_protected, latest_version, merge_policy
                FROM branches WHERE branch_name = ?
                """,
                (branch_name,),
            )

            row = cursor.fetchone()
            if row:
                return DataBranch(
                    branch_name=row[0],
                    created_from=row[1],
                    created_by=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    description=row[4],
                    is_protected=bool(row[5]),
                    latest_version=row[6],
                    merge_policy=ConflictResolution(row[7]),
                )
            return None

    def update_branch_latest_version(
        self, branch_name: str, version_id: str
    ) -> None:
        """ブランチの最新バージョンを更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE branches
                SET latest_version = ?
                WHERE branch_name = ?
                """,
                (version_id, branch_name),
            )

    def save_tag(self, tag: DataTag) -> None:
        """タグをデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tags
                (tag_name, version_id, created_by, created_at, description, is_release)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tag.tag_name,
                    tag.version_id,
                    tag.created_by,
                    tag.created_at.isoformat(),
                    tag.description,
                    1 if tag.is_release else 0,
                ),
            )

    def get_tag(self, tag_name: str) -> Optional[DataTag]:
        """タグ名でタグ情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT tag_name, version_id, created_by, created_at, description, is_release
                FROM tags WHERE tag_name = ?
                """,
                (tag_name,),
            )

            row = cursor.fetchone()
            if row:
                return DataTag(
                    tag_name=row[0],
                    version_id=row[1],
                    created_by=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    description=row[4],
                    is_release=bool(row[5]),
                )
            return None

    def save_conflict(self, conflict: VersionConflict) -> None:
        """競合情報をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conflicts
                (conflict_id, source_version, target_version, conflict_type,
                 file_path, description, resolution_strategy, resolved,
                 resolved_by, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict.conflict_id,
                    conflict.source_version,
                    conflict.target_version,
                    conflict.conflict_type,
                    conflict.file_path,
                    conflict.description,
                    conflict.resolution_strategy.value,
                    1 if conflict.resolved else 0,
                    conflict.resolved_by,
                    (
                        conflict.resolved_at.isoformat()
                        if conflict.resolved_at
                        else None
                    ),
                ),
            )

    def get_system_statistics(self) -> Dict[str, int]:
        """システム統計情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            version_count = conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
            branch_count = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
            tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
            
            return {
                "total_versions": version_count,
                "total_branches": branch_count,
                "total_tags": tag_count,
            }

    def get_branch_statistics(self) -> Dict[str, Dict[str, int]]:
        """ブランチ別統計情報を取得"""
        branch_stats = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT branch, COUNT(*) as count, SUM(size_bytes) as total_size
                FROM versions WHERE status != ?
                GROUP BY branch
                """,
                (DataStatus.DELETED.value,),
            )

            for branch, count, total_size in cursor.fetchall():
                branch_stats[branch] = {
                    "version_count": count,
                    "total_size_bytes": total_size or 0,
                }

        return branch_stats

    def cleanup_old_versions(
        self, branch: str, max_versions: int
    ) -> List[DataVersion]:
        """古いバージョンを特定してアーカイブ対象を返す"""
        if max_versions <= 0:
            return []

        versions = self.get_versions_by_branch(branch, limit=1000)
        if len(versions) <= max_versions:
            return []

        # タグ付きとAPPROVEDは保護
        versions_to_archive = []
        keep_count = 0

        for version in versions:  # 新しい順で並んでいる
            if version.tag or version.status == DataStatus.APPROVED:
                continue  # 保護対象

            if keep_count < max_versions:
                keep_count += 1
            else:
                versions_to_archive.append(version)

        return versions_to_archive

    def archive_version(self, version_id: str) -> None:
        """バージョンをアーカイブ状態に更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE versions
                SET status = ?
                WHERE version_id = ?
                """,
                (DataStatus.ARCHIVED.value, version_id),
            )