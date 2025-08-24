#!/usr/bin/env python3
"""
スナップショット管理モジュール

データバージョン管理システムのスナップショット作成、復元、
自動バックアップ機能を提供します。

Classes:
    SnapshotManager: スナップショット管理メインクラス
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .database import DatabaseManager
from .data_operations import DataOperations
from .types import DataVersion

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SnapshotManager:
    """スナップショット管理クラス
    
    ブランチ全体のスナップショット作成、復元、自動バックアップ機能を提供します。
    災害復旧やポイントインタイム復元に使用されます。
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        data_operations: DataOperations,
        repository_path: Path,
    ):
        """SnapshotManagerの初期化
        
        Args:
            database_manager: データベース管理インスタンス
            data_operations: データ操作インスタンス
            repository_path: リポジトリのルートパス
        """
        self.db_manager = database_manager
        self.data_ops = data_operations
        self.repository_path = repository_path
        
        # スナップショット用ディレクトリ
        self.snapshots_path = repository_path / "snapshots"
        self.backups_path = repository_path / "backups"
        
        self.snapshots_path.mkdir(parents=True, exist_ok=True)
        self.backups_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("スナップショットマネージャー初期化完了")

    async def create_snapshot(
        self, 
        branch: str = "main",
        description: str = "",
        include_data: bool = True,
        created_by: str = "system"
    ) -> str:
        """ブランチのスナップショットを作成
        
        Args:
            branch: スナップショット対象ブランチ
            description: スナップショットの説明
            include_data: データファイルも含めるか
            created_by: 作成者
            
        Returns:
            作成されたスナップショットID
        """
        snapshot_id = f"snapshot_{branch}_{int(time.time())}"
        logger.info(f"スナップショット作成開始: {snapshot_id}")

        try:
            # ブランチのバージョンリストを取得
            versions = self.db_manager.get_versions_by_branch(branch, limit=1000)
            
            if not versions:
                logger.warning(f"対象ブランチにバージョンがありません: {branch}")
                return snapshot_id

            # スナップショットディレクトリ作成
            snapshot_path = self.snapshots_path / snapshot_id
            snapshot_path.mkdir(parents=True, exist_ok=True)

            # メタデータ作成
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "branch": branch,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": created_by,
                "version_count": len(versions),
                "versions": [self._version_to_dict(v) for v in versions],
                "total_size_bytes": sum(v.size_bytes for v in versions),
                "include_data": include_data,
            }

            # メタデータファイル保存
            metadata_file = snapshot_path / "snapshot.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_metadata, f, indent=2, ensure_ascii=False)

            # データベースのバックアップ
            db_backup_path = snapshot_path / "versions.db"
            shutil.copy2(self.db_manager.db_path, db_backup_path)

            # 設定ファイルのバックアップ
            config_source = self.repository_path / ".dvconfig"
            if config_source.exists():
                shutil.copy2(config_source, snapshot_path / ".dvconfig")

            # データファイルのコピー（オプション）
            if include_data:
                data_count = await self._copy_version_data(versions, snapshot_path)
                snapshot_metadata["copied_data_files"] = data_count
                
                # メタデータを更新
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"スナップショット作成完了: {snapshot_id} ({len(versions)}版)")
            return snapshot_id

        except Exception as e:
            logger.error(f"スナップショット作成エラー ({snapshot_id}): {e}")
            # エラー時はディレクトリをクリーンアップ
            snapshot_path = self.snapshots_path / snapshot_id
            if snapshot_path.exists():
                shutil.rmtree(snapshot_path, ignore_errors=True)
            raise

    async def restore_snapshot(
        self, 
        snapshot_id: str, 
        target_branch: Optional[str] = None,
        restore_data: bool = True
    ) -> bool:
        """スナップショットから復元
        
        Args:
            snapshot_id: 復元するスナップショットID
            target_branch: 復元先ブランチ（Noneの場合はオリジナルブランチ）
            restore_data: データファイルも復元するか
            
        Returns:
            復元成功した場合True
        """
        logger.info(f"スナップショット復元開始: {snapshot_id}")

        try:
            snapshot_path = self.snapshots_path / snapshot_id
            
            if not snapshot_path.exists():
                raise FileNotFoundError(f"スナップショットが見つかりません: {snapshot_id}")

            # メタデータ読み込み
            metadata_file = snapshot_path / "snapshot.json"
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)

            original_branch = metadata["branch"]
            restore_branch = target_branch or original_branch
            versions_data = metadata["versions"]

            logger.info(f"復元先ブランチ: {restore_branch} ({len(versions_data)}版)")

            # データベース復元
            db_backup = snapshot_path / "versions.db"
            if db_backup.exists():
                # 現在のDBをバックアップ
                current_db_backup = self.backups_path / f"pre_restore_{int(time.time())}_versions.db"
                shutil.copy2(self.db_manager.db_path, current_db_backup)
                
                # スナップショットDBで復元
                shutil.copy2(db_backup, self.db_manager.db_path)

            # データファイル復元
            if restore_data and metadata.get("include_data", False):
                restored_count = await self._restore_version_data(snapshot_path, versions_data)
                logger.info(f"データファイル復元完了: {restored_count}件")

            # 設定ファイル復元
            config_backup = snapshot_path / ".dvconfig"
            if config_backup.exists():
                shutil.copy2(config_backup, self.repository_path / ".dvconfig")

            logger.info(f"スナップショット復元完了: {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"スナップショット復元エラー ({snapshot_id}): {e}")
            return False

    async def list_snapshots(self) -> List[Dict[str, any]]:
        """作成済みスナップショットのリストを取得
        
        Returns:
            スナップショット情報のリスト
        """
        snapshots = []

        try:
            for snapshot_dir in self.snapshots_path.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                    
                metadata_file = snapshot_dir / "snapshot.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    # 基本情報を抽出
                    snapshot_info = {
                        "snapshot_id": metadata.get("snapshot_id"),
                        "branch": metadata.get("branch"),
                        "description": metadata.get("description", ""),
                        "created_at": metadata.get("created_at"),
                        "created_by": metadata.get("created_by", "unknown"),
                        "version_count": metadata.get("version_count", 0),
                        "total_size_bytes": metadata.get("total_size_bytes", 0),
                        "include_data": metadata.get("include_data", False),
                    }
                    snapshots.append(snapshot_info)
                    
                except Exception as e:
                    logger.warning(f"スナップショットメタデータ読み込みエラー ({snapshot_dir.name}): {e}")

            # 作成日時でソート（新しい順）
            snapshots.sort(key=lambda s: s.get("created_at", ""), reverse=True)
            
            logger.debug(f"スナップショットリスト取得完了: {len(snapshots)}件")
            return snapshots

        except Exception as e:
            logger.error(f"スナップショットリスト取得エラー: {e}")
            return []

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """スナップショットを削除
        
        Args:
            snapshot_id: 削除するスナップショットID
            
        Returns:
            削除成功した場合True
        """
        logger.info(f"スナップショット削除開始: {snapshot_id}")

        try:
            snapshot_path = self.snapshots_path / snapshot_id
            
            if not snapshot_path.exists():
                logger.warning(f"削除対象スナップショットが見つかりません: {snapshot_id}")
                return False

            # ディレクトリごと削除
            shutil.rmtree(snapshot_path)

            logger.info(f"スナップショット削除完了: {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"スナップショット削除エラー ({snapshot_id}): {e}")
            return False

    async def create_automatic_backup(
        self, 
        trigger_version: str,
        backup_type: str = "automatic"
    ) -> str:
        """自動バックアップを作成
        
        Args:
            trigger_version: バックアップのトリガーとなったバージョンID
            backup_type: バックアップタイプ
            
        Returns:
            作成されたバックアップID
        """
        backup_id = f"auto_backup_{trigger_version}_{int(time.time())}"
        logger.info(f"自動バックアップ作成開始: {backup_id}")

        try:
            backup_path = self.backups_path / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)

            # データベースバックアップ
            backup_db = backup_path / "versions.db"
            shutil.copy2(self.db_manager.db_path, backup_db)

            # 設定ファイルバックアップ
            config_source = self.repository_path / ".dvconfig"
            if config_source.exists():
                shutil.copy2(config_source, backup_path / ".dvconfig")

            # バックアップメタデータ
            backup_metadata = {
                "backup_id": backup_id,
                "trigger_version": trigger_version,
                "created_at": datetime.utcnow().isoformat(),
                "backup_type": backup_type,
                "database_size_bytes": backup_db.stat().st_size if backup_db.exists() else 0,
            }

            metadata_file = backup_path / "backup.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"自動バックアップ作成完了: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"自動バックアップ作成エラー ({backup_id}): {e}")
            return backup_id

    async def cleanup_old_snapshots(self, max_snapshots: int = 10) -> int:
        """古いスナップショットをクリーンアップ
        
        Args:
            max_snapshots: 保持するスナップショット数
            
        Returns:
            削除されたスナップショット数
        """
        try:
            snapshots = await self.list_snapshots()
            
            if len(snapshots) <= max_snapshots:
                return 0

            # 削除対象を特定（古い順）
            snapshots_to_delete = snapshots[max_snapshots:]
            deleted_count = 0

            for snapshot in snapshots_to_delete:
                snapshot_id = snapshot.get("snapshot_id")
                if snapshot_id and await self.delete_snapshot(snapshot_id):
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"古いスナップショットクリーンアップ完了: {deleted_count}件削除")

            return deleted_count

        except Exception as e:
            logger.error(f"スナップショットクリーンアップエラー: {e}")
            return 0

    async def _copy_version_data(
        self, versions: List[DataVersion], target_path: Path
    ) -> int:
        """バージョンデータファイルをコピー
        
        Args:
            versions: コピー対象バージョンリスト
            target_path: コピー先パス
            
        Returns:
            コピーしたファイル数
        """
        copied_count = 0
        data_dir = target_path / "data"
        data_dir.mkdir(exist_ok=True)

        for version in versions:
            try:
                source_path = self.data_ops.data_storage_path / f"{version.version_id}.json"
                if source_path.exists():
                    target_file = data_dir / f"{version.version_id}.json"
                    shutil.copy2(source_path, target_file)
                    copied_count += 1
            except Exception as e:
                logger.warning(f"データファイルコピーエラー ({version.version_id}): {e}")

        return copied_count

    async def _restore_version_data(
        self, snapshot_path: Path, versions_data: List[Dict]
    ) -> int:
        """バージョンデータファイルを復元
        
        Args:
            snapshot_path: スナップショットディレクトリパス
            versions_data: バージョン情報リスト
            
        Returns:
            復元したファイル数
        """
        restored_count = 0
        source_data_dir = snapshot_path / "data"
        
        if not source_data_dir.exists():
            return 0

        for version_data in versions_data:
            try:
                version_id = version_data.get("version_id")
                if version_id:
                    source_file = source_data_dir / f"{version_id}.json"
                    if source_file.exists():
                        target_path = self.data_ops.data_storage_path / f"{version_id}.json"
                        shutil.copy2(source_file, target_path)
                        restored_count += 1
            except Exception as e:
                logger.warning(f"データファイル復元エラー ({version_data}): {e}")

        return restored_count

    def _version_to_dict(self, version: DataVersion) -> Dict:
        """DataVersionを辞書に変換（JSONシリアライズ用）
        
        Args:
            version: 変換するバージョンオブジェクト
            
        Returns:
            辞書形式のバージョン情報
        """
        return version.to_dict()

    def get_storage_usage(self) -> Dict[str, int]:
        """ストレージ使用量情報を取得
        
        Returns:
            ストレージ使用量情報
        """
        try:
            snapshots_size = sum(
                f.stat().st_size for f in self.snapshots_path.rglob("*") if f.is_file()
            )
            backups_size = sum(
                f.stat().st_size for f in self.backups_path.rglob("*") if f.is_file()
            )
            
            return {
                "snapshots_size_bytes": snapshots_size,
                "backups_size_bytes": backups_size,
                "total_size_bytes": snapshots_size + backups_size,
                "snapshots_count": len(list(self.snapshots_path.iterdir())),
                "backups_count": len(list(self.backups_path.iterdir())),
            }
            
        except Exception as e:
            logger.error(f"ストレージ使用量取得エラー: {e}")
            return {
                "snapshots_size_bytes": 0,
                "backups_size_bytes": 0,
                "total_size_bytes": 0,
                "snapshots_count": 0,
                "backups_count": 0,
            }