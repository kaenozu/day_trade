#!/usr/bin/env python3
"""
データバージョン管理システム - データベース操作

Issue #420: データ管理とデータ品質保証メカニズムの強化

データベース初期化とテーブル作成、基本操作を提供
"""

import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .models import DataBranch, ConflictResolution, DataStatus

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DatabaseManager:
    """データベース管理クラス"""

    def __init__(self, repository_path: Path):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
        """
        self.repository_path = repository_path
        self.db_path = repository_path / "versions.db"

    def initialize_database(self) -> None:
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # バージョン管理テーブル
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
                    checksum TEXT
                )
            """
            )

            # ブランチ管理テーブル
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

            # タグ管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    tag_name TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    description TEXT,
                    is_release INTEGER DEFAULT 0
                )
            """
            )

            # 競合管理テーブル
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
                    resolved_at TEXT
                )
            """
            )

            # インデックス作成
            self._create_indexes(conn)

            # デフォルトブランチ作成
            self._ensure_default_branch(conn)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """インデックス作成"""
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_branch ON versions(branch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON versions(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_status ON versions(status)"
        )

    def _ensure_default_branch(self, conn: sqlite3.Connection) -> None:
        """デフォルトブランチ確保"""
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

            conn.execute(
                """
                INSERT INTO branches
                (branch_name, created_from, created_by, created_at, description, is_protected, merge_policy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    main_branch.branch_name,
                    main_branch.created_from,
                    main_branch.created_by,
                    main_branch.created_at.isoformat(),
                    main_branch.description,
                    1 if main_branch.is_protected else 0,
                    main_branch.merge_policy.value,
                ),
            )

            logger.info("デフォルトmainブランチ作成完了")

    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 統計情報取得
                version_count = conn.execute(
                    "SELECT COUNT(*) FROM versions"
                ).fetchone()[0]
                branch_count = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[
                    0
                ]
                tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]

                # ブランチ別バージョン数
                branch_stats = {}
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

                return {
                    "total_versions": version_count,
                    "total_branches": branch_count,
                    "total_tags": tag_count,
                    "branch_statistics": branch_stats,
                }

        except Exception as e:
            logger.error(f"システム統計情報取得エラー: {e}")
            return {"error": str(e)}