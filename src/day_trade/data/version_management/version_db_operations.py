#!/usr/bin/env python3
"""
データバージョン管理システム - バージョンDB操作

Issue #420: データ管理とデータ品質保証メカニズムの強化

バージョン関連のデータベース操作を提供
"""

import json
import shutil
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import DataVersion, DataStatus

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"version_key_{hash(str(args))}"


logger = get_context_logger(__name__)


class VersionDbOperations:
    """バージョンデータベース操作クラス"""

    def __init__(
        self,
        repository_path: Path,
        cache_manager: Optional[UnifiedCacheManager] = None
    ):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
            cache_manager: キャッシュマネージャー
        """
        self.repository_path = repository_path
        self.db_path = repository_path / "versions.db"
        self.cache_manager = cache_manager
        self.version_cache: Dict[str, DataVersion] = {}

    async def save_version_to_db(self, version: DataVersion) -> None:
        """バージョンをデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO versions
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

    async def get_version(self, version_id: str) -> Optional[DataVersion]:
        """バージョン取得"""
        # キャッシュチェック
        if version_id in self.version_cache:
            return self.version_cache[version_id]

        if self.cache_manager:
            cache_key = generate_unified_cache_key("version", version_id)
            cached_version = self.cache_manager.get(cache_key)
            if cached_version:
                self.version_cache[version_id] = cached_version
                return cached_version

        # データベース検索
        try:
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
                    version = DataVersion(
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

                    # キャッシュ更新
                    self.version_cache[version_id] = version
                    if self.cache_manager:
                        cache_key = generate_unified_cache_key("version", version_id)
                        self.cache_manager.put(cache_key, version, priority=4.0)

                    return version

                return None

        except Exception as e:
            logger.error(f"バージョン取得エラー: {e}")
            return None

    async def get_latest_version(self, branch: str) -> Optional[DataVersion]:
        """最新バージョン取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE branch = ? AND status != ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """,
                    (branch, DataStatus.DELETED.value),
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

        except Exception as e:
            logger.error(f"最新バージョン取得エラー: {e}")
            return None

    async def list_versions(self, branch: str) -> List[DataVersion]:
        """ブランチのバージョンリスト取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE branch = ? AND status != ?
                    ORDER BY timestamp ASC
                """,
                    (branch, DataStatus.DELETED.value),
                )

                versions = []
                for row in cursor.fetchall():
                    version = DataVersion(
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
                    versions.append(version)

                return versions

        except Exception as e:
            logger.error(f"バージョンリスト取得エラー: {e}")
            return []

    async def find_version_by_hash(
        self, data_hash: str, branch: str
    ) -> Optional[DataVersion]:
        """データハッシュによるバージョン検索"""
        try:
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

        except Exception as e:
            logger.error(f"ハッシュによるバージョン検索エラー: {e}")
            return None

    async def update_branch_latest_version(self, branch: str, version_id: str) -> None:
        """ブランチ最新バージョン更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE branches
                    SET latest_version = ?
                    WHERE branch_name = ?
                """,
                    (version_id, branch),
                )

        except Exception as e:
            logger.error(f"ブランチ最新バージョン更新エラー: {e}")

    async def archive_version(self, version_id: str) -> None:
        """バージョンアーカイブ"""
        try:
            # ステータス更新
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE versions
                    SET status = ?
                    WHERE version_id = ?
                """,
                    (DataStatus.ARCHIVED.value, version_id),
                )

            # データファイル移動
            source_path = self.repository_path / "data" / f"{version_id}.json"
            archive_path = (
                self.repository_path / "backups" / f"archived_{version_id}.json"
            )

            if source_path.exists():
                shutil.move(str(source_path), str(archive_path))

            # キャッシュクリア
            self.version_cache.pop(version_id, None)

        except Exception as e:
            logger.error(f"バージョンアーカイブエラー: {e}")

    async def get_version_history(
        self, branch: str, limit: int = 50
    ) -> List[DataVersion]:
        """バージョン履歴取得"""
        try:
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

                versions = []
                for row in cursor.fetchall():
                    version = DataVersion(
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
                    versions.append(version)

                return versions

        except Exception as e:
            logger.error(f"バージョン履歴取得エラー: {e}")
            return []