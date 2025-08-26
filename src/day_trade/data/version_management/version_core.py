#!/usr/bin/env python3
"""
データバージョン管理システム - バージョンコア操作

Issue #420: データ管理とデータ品質保証メカニズムの強化

データのコミット・チェックアウトの核心機能を提供
"""

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .models import DataVersion, DataStatus
from .data_manager import DataManager

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


class VersionCore:
    """バージョンコア操作クラス"""

    def __init__(
        self,
        repository_path: Path,
        cache_manager: Optional[UnifiedCacheManager] = None,
        max_versions: int = 100,
        auto_backup: bool = True
    ):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
            cache_manager: キャッシュマネージャー
            max_versions: 最大バージョン数
            auto_backup: 自動バックアップフラグ
        """
        self.repository_path = repository_path
        self.cache_manager = cache_manager
        self.max_versions = max_versions
        self.auto_backup = auto_backup
        
        self.data_manager = DataManager(repository_path)
        self.version_cache: Dict[str, DataVersion] = {}

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
            
        Raises:
            Exception: コミットに失敗した場合
        """
        start_time = time.time()
        branch = branch or "main"
        metadata = metadata or {}

        logger.info(f"データコミット開始: {branch} - {message}")

        try:
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path, self.cache_manager)
            
            # データハッシュ計算
            data_hash = self.data_manager.calculate_data_hash(data)

            # 重複チェック
            existing_version = await db_ops.find_version_by_hash(data_hash, branch)
            if existing_version:
                logger.info(f"同一データが既存: {existing_version.version_id}")
                return existing_version.version_id

            # バージョンID生成
            version_id = f"v_{int(time.time())}_{data_hash[:8]}"

            # 親バージョン取得
            parent_version = await db_ops.get_latest_version(branch)
            parent_version_id = parent_version.version_id if parent_version else None

            # データ保存
            data_path = await self.data_manager.save_version_data(version_id, data)

            # メタデータ作成
            full_metadata = {
                **metadata,
                "commit_time": datetime.utcnow().isoformat(),
                "data_path": str(data_path),
                "data_type": type(data).__name__,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

            if isinstance(data, pd.DataFrame):
                full_metadata.update(
                    {
                        "shape": data.shape,
                        "columns": list(data.columns),
                        "dtypes": data.dtypes.to_dict(),
                        "memory_usage": data.memory_usage(deep=True).sum(),
                    }
                )

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
                size_bytes=self.data_manager.get_data_size(data),
                file_count=1,
                checksum=self.data_manager.calculate_checksum(data),
            )

            # データベース保存
            await db_ops.save_version_to_db(version)

            # ブランチの最新バージョン更新
            await db_ops.update_branch_latest_version(branch, version_id)

            # キャッシュ更新
            if self.cache_manager:
                cache_key = generate_unified_cache_key("version", version_id)
                self.cache_manager.put(cache_key, version, priority=5.0)

            self.version_cache[version_id] = version

            # バージョン数制限チェック
            await self._cleanup_old_versions(branch, db_ops)

            # 自動バックアップ
            if self.auto_backup and len(await db_ops.list_versions(branch)) % 10 == 0:
                await self._create_automatic_backup(version_id)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"データコミット完了: {version_id} ({duration_ms:.1f}ms)")

            return version_id

        except Exception as e:
            logger.error(f"データコミットエラー: {e}")
            raise

    async def checkout_data(self, version_id: str) -> Tuple[Any, DataVersion]:
        """
        データチェックアウト
        
        Args:
            version_id: チェックアウトするバージョンID
            
        Returns:
            データとバージョン情報のタプル
            
        Raises:
            ValueError: バージョンが見つからない、または削除されている場合
            Exception: チェックアウトに失敗した場合
        """
        logger.info(f"データチェックアウト: {version_id}")

        try:
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path, self.cache_manager)
            
            # バージョン情報取得
            version = await db_ops.get_version(version_id)
            if not version:
                raise ValueError(f"バージョンが見つかりません: {version_id}")

            if version.status == DataStatus.DELETED:
                raise ValueError(f"削除されたバージョンです: {version_id}")

            # データロード
            data = await self.data_manager.load_version_data(version_id)

            logger.info(f"データチェックアウト完了: {version_id}")
            return data, version

        except Exception as e:
            logger.error(f"データチェックアウトエラー: {e}")
            raise

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
        branch = branch or "main"

        try:
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path, self.cache_manager)
            return await db_ops.get_version_history(branch, limit)

        except Exception as e:
            logger.error(f"バージョン履歴取得エラー: {e}")
            return []

    async def _cleanup_old_versions(self, branch: str, db_ops) -> None:
        """古いバージョンのクリーンアップ"""
        if self.max_versions <= 0:
            return

        try:
            versions = await db_ops.list_versions(branch)
            if len(versions) <= self.max_versions:
                return

            # 古いバージョンを特定（タグ付きとリリースは保護）
            versions_to_delete = []
            keep_count = 0

            for version in reversed(versions):  # 新しい順
                if version.tag or version.status == DataStatus.APPROVED:
                    continue  # 保護対象

                if keep_count < self.max_versions:
                    keep_count += 1
                else:
                    versions_to_delete.append(version)

            # 削除実行
            for version in versions_to_delete:
                await db_ops.archive_version(version.version_id)

            if versions_to_delete:
                logger.info(f"古いバージョンをアーカイブ: {len(versions_to_delete)}件")

        except Exception as e:
            logger.error(f"バージョンクリーンアップエラー: {e}")

    async def _create_automatic_backup(self, version_id: str) -> None:
        """自動バックアップ作成"""
        try:
            from .snapshot_manager import SnapshotManager
            snapshot_manager = SnapshotManager(self.repository_path)
            backup_id = await snapshot_manager.create_backup("automatic", version_id)
            logger.info(f"自動バックアップ作成完了: {backup_id}")

        except Exception as e:
            logger.error(f"自動バックアップ作成エラー: {e}")