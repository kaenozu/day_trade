#!/usr/bin/env python3
"""
データバージョン管理システム - メインマネージャー

Issue #420: データ管理とデータ品質保証メカニズムの強化

統合されたデータバージョン管理システムのメインクラス
"""

import asyncio
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .models import ConflictResolution, VersionConflict, DataVersion
from .database import DatabaseManager
from .version_operations import VersionOperations
from .branch_manager import BranchManager
from .merge_manager import MergeManager
from .snapshot_manager import SnapshotManager
from .utils import (
    initialize_repository_structure,
    create_config_file,
    cleanup_temp_files,
    calculate_repository_size,
)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import UnifiedCacheManager
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True


logger = get_context_logger(__name__)


class DataVersionManager:
    """データバージョン管理システムメインクラス"""

    def __init__(
        self,
        repository_path: str = "data/versions",
        enable_cache: bool = True,
        max_versions: int = 100,
        auto_backup: bool = True,
    ):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
            enable_cache: キャッシュ有効化フラグ
            max_versions: 最大バージョン数
            auto_backup: 自動バックアップフラグ
        """
        self.repository_path = Path(repository_path)
        self.enable_cache = enable_cache
        self.max_versions = max_versions
        self.auto_backup = auto_backup

        # ディレクトリ構造初期化
        initialize_repository_structure(self.repository_path)
        
        # 設定ファイル作成
        create_config_file(self.repository_path, {
            "auto_backup": auto_backup,
            "max_versions": max_versions,
        })

        # 各マネージャー初期化
        self.database_manager = DatabaseManager(self.repository_path)
        self.database_manager.initialize_database()
        
        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=1024
                )
                logger.info("バージョン管理キャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        self.version_operations = VersionOperations(
            self.repository_path, self.cache_manager, max_versions, auto_backup
        )
        self.branch_manager = BranchManager(self.repository_path)
        self.merge_manager = MergeManager(self.repository_path)
        self.snapshot_manager = SnapshotManager(self.repository_path)

        # 内部管理状態
        self.current_branch = "main"

        logger.info("データバージョン管理システム初期化完了")
        logger.info(f"  - リポジトリパス: {self.repository_path}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - 最大バージョン数: {max_versions}")
        logger.info(f"  - 自動バックアップ: {'有効' if auto_backup else '無効'}")

    async def commit_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        message: str,
        author: str = "system",
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        データコミット
        
        Args:
            data: コミットするデータ
            message: コミットメッセージ
            author: 作成者
            branch: ブランチ名
            metadata: メタデータ
            
        Returns:
            作成されたバージョンID
        """
        return await self.version_operations.commit_data(
            data, message, author, branch or self.current_branch, metadata
        )

    async def checkout_data(self, version_id: str) -> Tuple[Any, DataVersion]:
        """
        データチェックアウト
        
        Args:
            version_id: チェックアウトするバージョンID
            
        Returns:
            データとバージョン情報のタプル
        """
        return await self.version_operations.checkout_data(version_id)

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
        """
        return await self.branch_manager.create_branch(
            branch_name, from_version, description, created_by
        )

    async def merge_branches(
        self,
        source_branch: str,
        target_branch: str,
        strategy: ConflictResolution = ConflictResolution.MANUAL,
        author: str = "system",
    ) -> Tuple[str, List[VersionConflict]]:
        """
        ブランチマージ
        
        Args:
            source_branch: マージ元ブランチ名
            target_branch: マージ先ブランチ名
            strategy: 競合解決戦略
            author: マージ実行者
            
        Returns:
            マージバージョンIDと競合リストのタプル
        """
        return await self.merge_manager.merge_branches(
            source_branch, target_branch, strategy, author
        )

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
        """
        return await self.branch_manager.create_tag(
            tag_name, version_id, description, is_release, created_by
        )

    async def create_snapshot(self, branch: Optional[str] = None) -> str:
        """
        スナップショット作成
        
        Args:
            branch: スナップショット対象のブランチ名
            
        Returns:
            作成されたスナップショットID
        """
        return await self.snapshot_manager.create_snapshot(branch)

    async def get_version_history(
        self, branch: Optional[str] = None, limit: int = 50
    ) -> List[DataVersion]:
        """
        バージョン履歴取得
        
        Args:
            branch: ブランチ名
            limit: 取得件数の上限
            
        Returns:
            バージョン履歴のリスト
        """
        return await self.version_operations.get_version_history(
            branch or self.current_branch, limit
        )

    async def get_version_diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        バージョン間差分取得
        
        Args:
            version1: 比較元バージョンID
            version2: 比較先バージョンID
            
        Returns:
            差分情報
        """
        return await self.merge_manager.get_version_diff(version1, version2)

    async def get_system_status(self) -> Dict[str, Any]:
        """
        システム状態取得
        
        Returns:
            システム状態情報
        """
        try:
            # 統計情報取得
            statistics = self.database_manager.get_system_statistics()
            
            # リポジトリサイズ計算
            size_info = calculate_repository_size(self.repository_path)

            return {
                "repository_path": str(self.repository_path),
                "current_branch": self.current_branch,
                "cache_enabled": self.enable_cache,
                "auto_backup": self.auto_backup,
                "max_versions": self.max_versions,
                "statistics": statistics,
                "repository_size": size_info,
                "cache_status": {
                    "version_cache_size": len(self.version_operations.version_cache),
                    "branch_cache_size": len(self.branch_manager.branch_cache),
                },
            }

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("データバージョン管理システム クリーンアップ開始")

        # キャッシュクリア
        self.version_operations.version_cache.clear()
        self.branch_manager.clear_branch_cache()

        # 一時ファイルクリーンアップ
        cleanup_temp_files(self.repository_path)

        # 古いスナップショット・バックアップクリーンアップ
        self.snapshot_manager.cleanup_old_snapshots()
        self.snapshot_manager.cleanup_old_backups()

        logger.info("データバージョン管理システム クリーンアップ完了")


# Factory function
def create_data_version_manager(
    repository_path: str = "data/versions",
    enable_cache: bool = True,
    max_versions: int = 100,
    auto_backup: bool = True,
) -> DataVersionManager:
    """
    データバージョン管理システム作成
    
    Args:
        repository_path: リポジトリのパス
        enable_cache: キャッシュ有効化フラグ
        max_versions: 最大バージョン数
        auto_backup: 自動バックアップフラグ
        
    Returns:
        DataVersionManagerインスタンス
    """
    return DataVersionManager(
        repository_path=repository_path,
        enable_cache=enable_cache,
        max_versions=max_versions,
        auto_backup=auto_backup,
    )