#!/usr/bin/env python3
"""
データバージョン管理システム - メインマネージャー

全ての機能を統合したメインマネージャークラス。
元のDataVersionManagerの機能を複数のサブマネージャーに分割し、
統合されたAPIを提供します。

Classes:
    DataVersionManager: メインマネージャークラス
"""

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .branch_manager import BranchManager
from .data_operations import DataOperations
from .database import DatabaseManager
from .diff_calculator import DiffCalculator
from .merge_manager import MergeManager
from .snapshot_manager import SnapshotManager
from .tag_manager import TagManager
from .types import ConflictResolution, DataStatus, DataVersion, VersionConflict

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
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

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataVersionManager:
    """データバージョン管理システム - メインマネージャー
    
    複数のサブマネージャーを統合し、元のDataVersionManagerと
    互換性のあるAPIを提供します。
    """

    def __init__(
        self,
        repository_path: str = "data/versions",
        enable_cache: bool = True,
        max_versions: int = 100,
        auto_backup: bool = True,
    ):
        """DataVersionManagerの初期化
        
        Args:
            repository_path: データ保存先パス
            enable_cache: キャッシュ有効フラグ
            max_versions: 最大バージョン保持数
            auto_backup: 自動バックアップ有効フラグ
        """
        self.repository_path = Path(repository_path)
        self.enable_cache = enable_cache
        self.max_versions = max_versions
        self.auto_backup = auto_backup
        self.current_branch = "main"

        # リポジトリディレクトリ初期化
        self._initialize_repository()

        # サブマネージャー初期化
        self.db_manager = DatabaseManager(self.repository_path / "versions.db")
        self.data_ops = DataOperations(self.repository_path / "data")
        self.branch_manager = BranchManager(self.db_manager)
        self.tag_manager = TagManager(self.db_manager)
        self.merge_manager = MergeManager(self.db_manager, self.data_ops)
        self.snapshot_manager = SnapshotManager(
            self.db_manager, self.data_ops, self.repository_path
        )
        self.diff_calculator = DiffCalculator(self.data_ops)

        # キャッシュマネージャー初期化
        self.cache_manager = None
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=1024
                )
                logger.info("バージョン管理キャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")

        # 内部管理状態
        self.version_cache: Dict[str, DataVersion] = {}

        logger.info("データバージョン管理システム初期化完了")
        logger.info(f"  - リポジトリパス: {self.repository_path}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - 最大バージョン数: {max_versions}")
        logger.info(f"  - 自動バックアップ: {'有効' if auto_backup else '無効'}")

    def _initialize_repository(self) -> None:
        """リポジトリディレクトリ初期化"""
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        for subdir in ["data", "metadata", "snapshots", "backups", "temp"]:
            (self.repository_path / subdir).mkdir(exist_ok=True)

        # .dvconfig 設定ファイル作成
        config_file = self.repository_path / ".dvconfig"
        if not config_file.exists():
            config = {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "default_branch": "main",
                "auto_backup": self.auto_backup,
                "max_versions": self.max_versions,
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    async def commit_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        message: str,
        author: str = "system",
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """データコミット
        
        Args:
            data: コミットするデータ
            message: コミットメッセージ
            author: コミット作成者
            branch: コミット対象ブランチ
            metadata: 追加メタデータ
            
        Returns:
            作成されたバージョンID
        """
        start_time = time.time()
        branch = branch or self.current_branch
        metadata = metadata or {}

        logger.info(f"データコミット開始: {branch} - {message}")

        try:
            # データハッシュ計算
            data_hash = self.data_ops.calculate_data_hash(data)

            # 重複チェック
            existing_version = self.db_manager.find_version_by_hash(data_hash, branch)
            if existing_version:
                logger.info(f"同一データが既存: {existing_version.version_id}")
                return existing_version.version_id

            # バージョンID生成
            version_id = f"v_{int(time.time())}_{data_hash[:8]}"

            # 親バージョン取得
            parent_version = await self._get_latest_version(branch)
            parent_version_id = parent_version.version_id if parent_version else None

            # データ保存
            data_path = await self.data_ops.save_version_data(version_id, data)

            # メタデータ構築
            full_metadata = {
                **metadata,
                "commit_time": datetime.utcnow().isoformat(),
                "data_path": str(data_path),
                "data_type": type(data).__name__,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

            if isinstance(data, pd.DataFrame):
                full_metadata.update({
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "memory_usage": data.memory_usage(deep=True).sum(),
                })

            # バージョン情報作成
            version = DataVersion(
                version_id=version_id,
                parent_version=parent_version_id,
                branch=branch,
                author=author,
                timestamp=datetime.utcnow(),
                message=message,
                data_hash=data_hash,
                metadata=full_metadata,
                status=DataStatus.ACTIVE,
                size_bytes=self.data_ops.get_data_size(data),
                file_count=self.data_ops.get_file_count(data),
                checksum=self.data_ops.calculate_checksum(data),
            )

            # データベース保存
            self.db_manager.save_version(version)

            # ブランチ最新バージョン更新
            await self.branch_manager.update_branch_latest_version(branch, version_id)

            # キャッシュ更新
            if self.cache_manager:
                cache_key = generate_unified_cache_key("version", version_id)
                self.cache_manager.put(cache_key, version, priority=5.0)

            self.version_cache[version_id] = version

            # バージョン数制限チェック
            await self._cleanup_old_versions(branch)

            # 自動バックアップ
            if self.auto_backup:
                versions = self.db_manager.get_versions_by_branch(branch, limit=10)
                if len(versions) % 10 == 0:
                    await self.snapshot_manager.create_automatic_backup(version_id)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"データコミット完了: {version_id} ({duration_ms:.1f}ms)")

            return version_id

        except Exception as e:
            logger.error(f"データコミットエラー: {e}")
            raise

    async def checkout_data(self, version_id: str) -> Tuple[Any, DataVersion]:
        """データチェックアウト
        
        Args:
            version_id: チェックアウトするバージョンID
            
        Returns:
            (データ, バージョン情報)のタプル
        """
        logger.info(f"データチェックアウト: {version_id}")

        try:
            # バージョン情報取得
            version = await self._get_version(version_id)
            if not version:
                raise ValueError(f"バージョンが見つかりません: {version_id}")

            if version.status == DataStatus.DELETED:
                raise ValueError(f"削除されたバージョンです: {version_id}")

            # データロード
            data = await self.data_ops.load_version_data(version_id)

            logger.info(f"データチェックアウト完了: {version_id}")
            return data, version

        except Exception as e:
            logger.error(f"データチェックアウトエラー: {e}")
            raise

    async def create_branch(
        self,
        branch_name: str,
        from_version: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
    ) -> str:
        """ブランチ作成（BranchManagerに委譲）"""
        if from_version is None:
            latest_version = await self._get_latest_version(self.current_branch)
            from_version = latest_version.version_id if latest_version else None

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
        """ブランチマージ（MergeManagerに委譲）"""
        logger.info(f"ブランチマージ開始: {source_branch} -> {target_branch}")

        try:
            # ブランチ存在確認
            source_branch_info = await self.branch_manager.get_branch(source_branch)
            target_branch_info = await self.branch_manager.get_branch(target_branch)

            if not source_branch_info or not target_branch_info:
                raise ValueError("ブランチが見つかりません")

            # 最新バージョン取得
            source_version = await self._get_latest_version(source_branch)
            target_version = await self._get_latest_version(target_branch)

            if not source_version:
                raise ValueError(f"マージ元ブランチにバージョンがありません: {source_branch}")

            # マージ実行
            merge_data, conflicts = await self.merge_manager.merge_branches(
                source_version, target_version, strategy, author
            )

            if merge_data is not None:
                # マージコミット作成
                merge_message = f"Merge branch '{source_branch}' into '{target_branch}'"
                merge_version_id = await self.commit_data(
                    merge_data,
                    merge_message,
                    author,
                    target_branch,
                    {
                        "merge_source": source_branch,
                        "merge_target": target_branch,
                        "merge_strategy": strategy.value,
                        "conflicts_resolved": len([c for c in conflicts if c.resolved]),
                    },
                )

                logger.info(f"ブランチマージ完了: {merge_version_id}")
                return merge_version_id, conflicts
            else:
                # 手動解決が必要
                return "", conflicts

        except Exception as e:
            logger.error(f"ブランチマージエラー: {e}")
            raise

    async def create_tag(
        self,
        tag_name: str,
        version_id: str,
        description: str = "",
        is_release: bool = False,
        created_by: str = "system",
    ) -> str:
        """タグ作成（TagManagerに委譲）"""
        return await self.tag_manager.create_tag(
            tag_name, version_id, description, is_release, created_by
        )

    async def create_snapshot(self, branch: Optional[str] = None) -> str:
        """スナップショット作成（SnapshotManagerに委譲）"""
        branch = branch or self.current_branch
        return await self.snapshot_manager.create_snapshot(branch)

    async def get_version_history(
        self, branch: Optional[str] = None, limit: int = 50
    ) -> List[DataVersion]:
        """バージョン履歴取得"""
        branch = branch or self.current_branch
        return self.db_manager.get_versions_by_branch(branch, limit)

    async def get_version_diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """バージョン間差分取得（DiffCalculatorに委譲）"""
        logger.info(f"バージョン差分計算: {version1} <-> {version2}")

        try:
            v1 = await self._get_version(version1)
            v2 = await self._get_version(version2)

            if not v1 or not v2:
                raise ValueError("指定されたバージョンが見つかりません")

            return await self.diff_calculator.calculate_version_diff(v1, v2)

        except Exception as e:
            logger.error(f"バージョン差分計算エラー: {e}")
            raise

    async def _get_version(self, version_id: str) -> Optional[DataVersion]:
        """バージョン取得（キャッシュ対応）"""
        # キャッシュチェック
        if version_id in self.version_cache:
            return self.version_cache[version_id]

        if self.cache_manager:
            cache_key = generate_unified_cache_key("version", version_id)
            cached_version = self.cache_manager.get(cache_key)
            if cached_version:
                self.version_cache[version_id] = cached_version
                return cached_version

        # データベースから取得
        version = self.db_manager.get_version(version_id)
        
        if version:
            # キャッシュ更新
            self.version_cache[version_id] = version
            if self.cache_manager:
                cache_key = generate_unified_cache_key("version", version_id)
                self.cache_manager.put(cache_key, version, priority=4.0)

        return version

    async def _get_latest_version(self, branch: str) -> Optional[DataVersion]:
        """最新バージョン取得"""
        versions = self.db_manager.get_versions_by_branch(branch, limit=1)
        return versions[0] if versions else None

    async def _cleanup_old_versions(self, branch: str) -> None:
        """古いバージョンクリーンアップ"""
        if self.max_versions <= 0:
            return

        versions_to_archive = self.db_manager.cleanup_old_versions(branch, self.max_versions)
        
        for version in versions_to_archive:
            # データファイルをアーカイブディレクトリに移動
            self.data_ops.move_to_archive(
                version.version_id, 
                self.repository_path / "backups"
            )
            
            # ステータスをアーカイブに変更
            self.db_manager.archive_version(version.version_id)
            
            # キャッシュクリア
            self.version_cache.pop(version.version_id, None)

        if versions_to_archive:
            logger.info(f"古いバージョンをアーカイブ: {len(versions_to_archive)}件")

    async def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        try:
            # 基本統計
            basic_stats = self.db_manager.get_system_statistics()
            branch_stats = self.db_manager.get_branch_statistics()

            return {
                "repository_path": str(self.repository_path),
                "current_branch": self.current_branch,
                "cache_enabled": self.enable_cache,
                "auto_backup": self.auto_backup,
                "max_versions": self.max_versions,
                "statistics": {
                    **basic_stats,
                    "branch_statistics": branch_stats,
                },
                "cache_status": {
                    "version_cache_size": len(self.version_cache),
                    **self.branch_manager.get_cache_status(),
                    **self.tag_manager.get_tag_statistics(),
                },
                "storage_usage": self.snapshot_manager.get_storage_usage(),
            }

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        logger.info("データバージョン管理システム クリーンアップ開始")

        # キャッシュクリア
        self.version_cache.clear()
        self.branch_manager.clear_cache()
        self.tag_manager.clear_cache()

        # 一時ファイルクリーンアップ
        temp_path = self.repository_path / "temp"
        if temp_path.exists():
            import shutil
            for temp_file in temp_path.glob("*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                except Exception as e:
                    logger.warning(f"一時ファイル削除エラー: {e}")

        logger.info("データバージョン管理システム クリーンアップ完了")


# Factory function for backward compatibility
def create_data_version_manager(
    repository_path: str = "data/versions",
    enable_cache: bool = True,
    max_versions: int = 100,
    auto_backup: bool = True,
) -> DataVersionManager:
    """データバージョン管理システム作成
    
    Args:
        repository_path: データ保存先パス
        enable_cache: キャッシュ有効フラグ  
        max_versions: 最大バージョン保持数
        auto_backup: 自動バックアップ有効フラグ
        
    Returns:
        DataVersionManagerインスタンス
    """
    return DataVersionManager(
        repository_path=repository_path,
        enable_cache=enable_cache,
        max_versions=max_versions,
        auto_backup=auto_backup,
    )