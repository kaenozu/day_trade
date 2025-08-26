#!/usr/bin/env python3
"""
データバージョン管理システム - スナップショット・バックアップ管理

Issue #420: データ管理とデータ品質保証メカニズムの強化

スナップショット作成とバックアップ機能を提供
"""

import json
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

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


class SnapshotManager:
    """スナップショット・バックアップ管理クラス"""

    def __init__(self, repository_path: Path):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
        """
        self.repository_path = repository_path
        self.snapshots_path = repository_path / "snapshots"
        self.backups_path = repository_path / "backups"
        self.db_path = repository_path / "versions.db"

    async def create_snapshot(self, branch: Optional[str] = None) -> str:
        """
        スナップショット作成
        
        Args:
            branch: スナップショット対象のブランチ名
            
        Returns:
            作成されたスナップショットID
            
        Raises:
            Exception: スナップショット作成に失敗した場合
        """
        branch = branch or "main"
        snapshot_id = f"snapshot_{branch}_{int(time.time())}"

        logger.info(f"スナップショット作成: {snapshot_id}")

        try:
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path)
            
            # ブランチの全バージョン取得
            versions = await db_ops.list_versions(branch)

            # スナップショットディレクトリ作成
            snapshot_path = self.snapshots_path / snapshot_id
            snapshot_path.mkdir(parents=True, exist_ok=True)

            # メタデータ作成
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "branch": branch,
                "created_at": datetime.utcnow().isoformat(),
                "version_count": len(versions),
                "versions": [v.version_id for v in versions],
                "total_size_bytes": sum(v.size_bytes for v in versions),
            }

            # メタデータ保存
            metadata_file = snapshot_path / "snapshot.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_metadata, f, indent=2, ensure_ascii=False)

            # データファイルコピー
            for version in versions:
                source_path = (
                    self.repository_path / "data" / f"{version.version_id}.json"
                )
                if source_path.exists():
                    target_path = snapshot_path / f"{version.version_id}.json"
                    shutil.copy2(source_path, target_path)

            logger.info(f"スナップショット作成完了: {snapshot_id}")
            return snapshot_id

        except Exception as e:
            logger.error(f"スナップショット作成エラー: {e}")
            raise

    async def create_backup(
        self,
        backup_type: str = "manual",
        trigger_version: Optional[str] = None
    ) -> str:
        """
        バックアップ作成
        
        Args:
            backup_type: バックアップ種別（manual, automatic）
            trigger_version: トリガーとなったバージョンID
            
        Returns:
            作成されたバックアップID
            
        Raises:
            Exception: バックアップ作成に失敗した場合
        """
        backup_id = f"{backup_type}_backup_{int(time.time())}"
        if trigger_version:
            backup_id = f"{backup_type}_backup_{trigger_version}_{int(time.time())}"

        logger.info(f"{backup_type}バックアップ作成: {backup_id}")

        try:
            backup_path = self.backups_path / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)

            # データベースバックアップ
            backup_db = backup_path / "versions.db"
            shutil.copy2(self.db_path, backup_db)

            # 設定ファイルバックアップ
            config_source = self.repository_path / ".dvconfig"
            if config_source.exists():
                shutil.copy2(config_source, backup_path / ".dvconfig")

            # バックアップメタデータ
            backup_metadata = {
                "backup_id": backup_id,
                "backup_type": backup_type,
                "created_at": datetime.utcnow().isoformat(),
                "trigger_version": trigger_version,
                "database_size": backup_db.stat().st_size if backup_db.exists() else 0,
            }

            with open(backup_path / "backup.json", "w", encoding="utf-8") as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"{backup_type}バックアップ作成完了: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"{backup_type}バックアップ作成エラー: {e}")
            raise

    async def restore_from_backup(self, backup_id: str) -> bool:
        """
        バックアップから復元
        
        Args:
            backup_id: 復元するバックアップID
            
        Returns:
            復元成功フラグ
            
        Raises:
            FileNotFoundError: バックアップが見つからない場合
            Exception: 復元に失敗した場合
        """
        logger.info(f"バックアップから復元開始: {backup_id}")

        try:
            backup_path = self.backups_path / backup_id
            if not backup_path.exists():
                raise FileNotFoundError(f"バックアップが見つかりません: {backup_id}")

            # バックアップメタデータ読み込み
            metadata_file = backup_path / "backup.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"バックアップメタデータが見つかりません: {backup_id}")

            with open(metadata_file, encoding="utf-8") as f:
                backup_metadata = json.load(f)

            # 現在のデータベースをバックアップ
            current_backup_id = f"pre_restore_{int(time.time())}"
            await self.create_backup("pre_restore", current_backup_id)

            # データベース復元
            backup_db = backup_path / "versions.db"
            if backup_db.exists():
                shutil.copy2(backup_db, self.db_path)

            # 設定ファイル復元
            config_backup = backup_path / ".dvconfig"
            if config_backup.exists():
                shutil.copy2(config_backup, self.repository_path / ".dvconfig")

            logger.info(f"バックアップから復元完了: {backup_id}")
            logger.info(f"復元前のデータは {current_backup_id} に保存されました")
            return True

        except Exception as e:
            logger.error(f"バックアップ復元エラー: {e}")
            raise

    def cleanup_old_snapshots(self, keep_count: int = 10) -> int:
        """
        古いスナップショットのクリーンアップ
        
        Args:
            keep_count: 保持するスナップショット数
            
        Returns:
            削除されたスナップショット数
        """
        try:
            if not self.snapshots_path.exists():
                return 0

            # スナップショットリスト取得（作成日時順）
            snapshots = []
            for snapshot_dir in self.snapshots_path.iterdir():
                if snapshot_dir.is_dir():
                    metadata_file = snapshot_dir / "snapshot.json"
                    if metadata_file.exists():
                        with open(metadata_file, encoding="utf-8") as f:
                            metadata = json.load(f)
                        snapshots.append((snapshot_dir, metadata["created_at"]))

            # 作成日時でソート（新しい順）
            snapshots.sort(key=lambda x: x[1], reverse=True)

            # 古いスナップショットを削除
            deleted_count = 0
            for snapshot_dir, created_at in snapshots[keep_count:]:
                try:
                    shutil.rmtree(snapshot_dir)
                    deleted_count += 1
                    logger.info(f"古いスナップショット削除: {snapshot_dir.name}")
                except Exception as e:
                    logger.warning(f"スナップショット削除エラー: {e}")

            return deleted_count

        except Exception as e:
            logger.error(f"スナップショットクリーンアップエラー: {e}")
            return 0

    def cleanup_old_backups(self, keep_count: int = 20) -> int:
        """
        古いバックアップのクリーンアップ
        
        Args:
            keep_count: 保持するバックアップ数
            
        Returns:
            削除されたバックアップ数
        """
        try:
            if not self.backups_path.exists():
                return 0

            # バックアップリスト取得（作成日時順）
            backups = []
            for backup_dir in self.backups_path.iterdir():
                if backup_dir.is_dir():
                    metadata_file = backup_dir / "backup.json"
                    if metadata_file.exists():
                        with open(metadata_file, encoding="utf-8") as f:
                            metadata = json.load(f)
                        # 手動バックアップは保護
                        if metadata.get("backup_type") != "manual":
                            backups.append((backup_dir, metadata["created_at"]))

            # 作成日時でソート（新しい順）
            backups.sort(key=lambda x: x[1], reverse=True)

            # 古いバックアップを削除
            deleted_count = 0
            for backup_dir, created_at in backups[keep_count:]:
                try:
                    shutil.rmtree(backup_dir)
                    deleted_count += 1
                    logger.info(f"古いバックアップ削除: {backup_dir.name}")
                except Exception as e:
                    logger.warning(f"バックアップ削除エラー: {e}")

            return deleted_count

        except Exception as e:
            logger.error(f"バックアップクリーンアップエラー: {e}")
            return 0