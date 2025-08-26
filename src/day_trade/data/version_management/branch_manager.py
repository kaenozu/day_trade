#!/usr/bin/env python3
"""
データバージョン管理システム - ブランチ・タグ管理

Issue #420: データ管理とデータ品質保証メカニズムの強化

ブランチとタグの作成・管理機能を提供
"""

import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .models import DataBranch, DataTag, ConflictResolution

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


class BranchManager:
    """ブランチ・タグ管理クラス"""

    def __init__(self, repository_path: Path):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
        """
        self.repository_path = repository_path
        self.db_path = repository_path / "versions.db"
        self.branch_cache: Dict[str, DataBranch] = {}

    async def create_branch(
        self,
        branch_name: str,
        from_version: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
    ) -> str:
        """
        ブランチ作成
        
        Args:
            branch_name: 作成するブランチ名
            from_version: 分岐元のバージョンID
            description: ブランチの説明
            created_by: 作成者
            
        Returns:
            作成されたブランチ名
            
        Raises:
            ValueError: ブランチが既に存在する場合
            Exception: 作成に失敗した場合
        """
        logger.info(f"ブランチ作成: {branch_name}")

        try:
            # ブランチ名重複チェック
            existing_branch = await self._get_branch(branch_name)
            if existing_branch:
                raise ValueError(f"ブランチが既に存在します: {branch_name}")

            # 元バージョン決定
            if from_version is None:
                from .version_db_operations import VersionDbOperations
                db_ops = VersionDbOperations(self.repository_path)
                latest_version = await db_ops.get_latest_version("main")
                from_version = latest_version.version_id if latest_version else "root"

            # ブランチ作成
            branch = DataBranch(
                branch_name=branch_name,
                created_from=from_version,
                created_by=created_by,
                description=description,
            )

            # データベース保存
            await self._save_branch_to_db(branch)

            # キャッシュ更新
            self.branch_cache[branch_name] = branch

            logger.info(f"ブランチ作成完了: {branch_name}")
            return branch_name

        except Exception as e:
            logger.error(f"ブランチ作成エラー: {e}")
            raise

    async def create_tag(
        self,
        tag_name: str,
        version_id: str,
        description: str = "",
        is_release: bool = False,
        created_by: str = "system",
    ) -> str:
        """
        タグ作成
        
        Args:
            tag_name: 作成するタグ名
            version_id: 対象のバージョンID
            description: タグの説明
            is_release: リリース用タグかどうか
            created_by: 作成者
            
        Returns:
            作成されたタグ名
            
        Raises:
            ValueError: タグが既に存在する、またはバージョンが見つからない場合
            Exception: 作成に失敗した場合
        """
        logger.info(f"タグ作成: {tag_name} -> {version_id}")

        try:
            # タグ名重複チェック
            existing_tag = await self._get_tag(tag_name)
            if existing_tag:
                raise ValueError(f"タグが既に存在します: {tag_name}")

            # バージョン存在確認
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path)
            version = await db_ops.get_version(version_id)
            if not version:
                raise ValueError(f"バージョンが見つかりません: {version_id}")

            # タグ作成
            tag = DataTag(
                tag_name=tag_name,
                version_id=version_id,
                created_by=created_by,
                description=description,
                is_release=is_release,
            )

            # データベース保存
            await self._save_tag_to_db(tag)

            # バージョンにタグ情報更新
            version.tag = tag_name
            await self._update_version_tag(version_id, tag_name)

            logger.info(f"タグ作成完了: {tag_name}")
            return tag_name

        except Exception as e:
            logger.error(f"タグ作成エラー: {e}")
            raise

    async def _get_branch(self, branch_name: str) -> Optional[DataBranch]:
        """
        ブランチ取得
        
        Args:
            branch_name: 取得するブランチ名
            
        Returns:
            ブランチ情報（存在しない場合はNone）
        """
        if branch_name in self.branch_cache:
            return self.branch_cache[branch_name]

        try:
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
                    branch = DataBranch(
                        branch_name=row[0],
                        created_from=row[1],
                        created_by=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        description=row[4],
                        is_protected=bool(row[5]),
                        latest_version=row[6],
                        merge_policy=ConflictResolution(row[7]),
                    )

                    self.branch_cache[branch_name] = branch
                    return branch

                return None

        except Exception as e:
            logger.error(f"ブランチ取得エラー: {e}")
            return None

    async def _get_tag(self, tag_name: str) -> Optional[DataTag]:
        """
        タグ取得
        
        Args:
            tag_name: 取得するタグ名
            
        Returns:
            タグ情報（存在しない場合はNone）
        """
        try:
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

        except Exception as e:
            logger.error(f"タグ取得エラー: {e}")
            return None

    async def _save_branch_to_db(self, branch: DataBranch) -> None:
        """
        ブランチをデータベース保存
        
        Args:
            branch: 保存するブランチ情報
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO branches
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

    async def _save_tag_to_db(self, tag: DataTag) -> None:
        """
        タグをデータベース保存
        
        Args:
            tag: 保存するタグ情報
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tags
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

    async def _update_version_tag(self, version_id: str, tag_name: str) -> None:
        """
        バージョン情報にタグを更新
        
        Args:
            version_id: 更新するバージョンID
            tag_name: 設定するタグ名
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE versions
                    SET tag = ?
                    WHERE version_id = ?
                """,
                    (tag_name, version_id),
                )

        except Exception as e:
            logger.error(f"バージョンタグ更新エラー: {e}")

    async def update_branch_latest_version(self, branch: str, version_id: str) -> None:
        """
        ブランチ最新バージョン更新
        
        Args:
            branch: 更新するブランチ名
            version_id: 最新バージョンID
        """
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

            # キャッシュ更新
            if branch in self.branch_cache:
                self.branch_cache[branch].latest_version = version_id

        except Exception as e:
            logger.error(f"ブランチ最新バージョン更新エラー: {e}")

    def clear_branch_cache(self) -> None:
        """ブランチキャッシュクリア"""
        self.branch_cache.clear()
        logger.debug("ブランチキャッシュクリア完了")