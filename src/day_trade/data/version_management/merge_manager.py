#!/usr/bin/env python3
"""
データバージョン管理システム - マージ・競合管理

Issue #420: データ管理とデータ品質保証メカニズムの強化

ブランチマージと競合検出・解決機能を提供
"""

import json
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import DataVersion, VersionConflict, ConflictResolution
from .branch_manager import BranchManager

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


class MergeManager:
    """マージ・競合管理クラス"""

    def __init__(self, repository_path: Path):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
        """
        self.repository_path = repository_path
        self.db_path = repository_path / "versions.db"
        self.branch_manager = BranchManager(repository_path)
        self.conflict_cache: List[VersionConflict] = []

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
            
        Raises:
            ValueError: ブランチが見つからない場合
            Exception: マージに失敗した場合
        """
        logger.info(f"ブランチマージ開始: {source_branch} -> {target_branch}")

        try:
            # ブランチ存在確認
            source_branch_info = await self.branch_manager._get_branch(source_branch)
            target_branch_info = await self.branch_manager._get_branch(target_branch)

            if not source_branch_info or not target_branch_info:
                raise ValueError("ブランチが見つかりません")

            # 最新バージョン取得
            from .version_db_operations import VersionDbOperations
            db_ops = VersionDbOperations(self.repository_path)
            
            source_version = await db_ops.get_latest_version(source_branch)
            target_version = await db_ops.get_latest_version(target_branch)

            if not source_version:
                raise ValueError(
                    f"マージ元ブランチにバージョンがありません: {source_branch}"
                )

            # 競合検出
            conflicts = await self._detect_merge_conflicts(
                source_version, target_version
            )

            if conflicts and strategy == ConflictResolution.MANUAL:
                logger.warning(f"マージ競合が検出されました: {len(conflicts)}件")
                return "", conflicts

            # 自動競合解決
            if conflicts:
                conflicts = await self._resolve_conflicts_automatically(
                    conflicts, strategy
                )

            # マージコミット作成
            merge_data = await self._create_merge_data(
                source_version, target_version, strategy
            )

            merge_message = f"Merge branch '{source_branch}' into '{target_branch}'"
            from .version_core import VersionCore
            version_core = VersionCore(self.repository_path)
            merge_version_id = await version_core.commit_data(
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

            # マージ操作記録
            await self._record_merge_operation(
                merge_version_id, source_branch, target_branch, conflicts
            )

            logger.info(f"ブランチマージ完了: {merge_version_id}")
            return merge_version_id, conflicts

        except Exception as e:
            logger.error(f"ブランチマージエラー: {e}")
            raise

    async def get_version_diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        バージョン間差分取得
        
        Args:
            version1: 比較元バージョンID
            version2: 比較先バージョンID
            
        Returns:
            差分情報
            
        Raises:
            ValueError: バージョンが見つからない場合
            Exception: 差分計算に失敗した場合
        """
        logger.info(f"バージョン差分計算: {version1} <-> {version2}")

        try:
            from .version_db_operations import VersionDbOperations
            from .version_core import VersionCore
            db_ops = VersionDbOperations(self.repository_path)
            version_core = VersionCore(self.repository_path)
            
            # バージョン情報取得
            v1 = await db_ops.get_version(version1)
            v2 = await db_ops.get_version(version2)

            if not v1 or not v2:
                raise ValueError("指定されたバージョンが見つかりません")

            # データロード
            data1, _ = await version_core.checkout_data(version1)
            data2, _ = await version_core.checkout_data(version2)

            # 差分計算
            diff_result = {
                "version1": version1,
                "version2": version2,
                "timestamp1": v1.timestamp.isoformat(),
                "timestamp2": v2.timestamp.isoformat(),
                "metadata_diff": self._calculate_metadata_diff(
                    v1.metadata, v2.metadata
                ),
                "data_diff": self._calculate_data_diff(data1, data2),
                "size_diff_bytes": v2.size_bytes - v1.size_bytes,
                "hash_changed": v1.data_hash != v2.data_hash,
            }

            return diff_result

        except Exception as e:
            logger.error(f"バージョン差分計算エラー: {e}")
            raise

    async def _detect_merge_conflicts(
        self, source_version: DataVersion, target_version: Optional[DataVersion]
    ) -> List[VersionConflict]:
        """
        マージ競合検出
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン
            
        Returns:
            検出された競合リスト
        """
        conflicts = []

        if not target_version:
            return conflicts  # ターゲットが空の場合は競合なし

        # データハッシュ競合
        if source_version.data_hash != target_version.data_hash:
            conflict = VersionConflict(
                conflict_id=f"conflict_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="data",
                description="データ内容が異なります",
            )
            conflicts.append(conflict)

        # メタデータ競合
        source_meta = source_version.metadata
        target_meta = target_version.metadata

        conflicting_keys = []
        for key in set(source_meta.keys()) & set(target_meta.keys()):
            if source_meta[key] != target_meta[key]:
                conflicting_keys.append(key)

        if conflicting_keys:
            conflict = VersionConflict(
                conflict_id=f"conflict_meta_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="metadata",
                description=f"メタデータ競合: {', '.join(conflicting_keys)}",
            )
            conflicts.append(conflict)

        return conflicts

    async def _resolve_conflicts_automatically(
        self, conflicts: List[VersionConflict], strategy: ConflictResolution
    ) -> List[VersionConflict]:
        """
        自動競合解決
        
        Args:
            conflicts: 競合リスト
            strategy: 解決戦略
            
        Returns:
            解決済み競合リスト
        """
        resolved_conflicts = []

        for conflict in conflicts:
            if strategy == ConflictResolution.AUTO_LATEST:
                # 最新バージョンを採用
                conflict.resolved = True
                conflict.resolved_by = "auto_resolver"
                conflict.resolved_at = datetime.utcnow()
                conflict.resolution_strategy = ConflictResolution.AUTO_LATEST

            elif strategy == ConflictResolution.AUTO_MERGE:
                # 自動マージ（簡易版）
                conflict.resolved = True
                conflict.resolved_by = "auto_merger"
                conflict.resolved_at = datetime.utcnow()
                conflict.resolution_strategy = ConflictResolution.AUTO_MERGE

            resolved_conflicts.append(conflict)

        return resolved_conflicts

    async def _create_merge_data(
        self,
        source_version: DataVersion,
        target_version: Optional[DataVersion],
        strategy: ConflictResolution,
    ) -> Any:
        """
        マージデータ作成
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン
            strategy: マージ戦略
            
        Returns:
            マージされたデータ
        """
        from .version_core import VersionCore
        version_core = VersionCore(self.repository_path)
        
        # ソースデータをロード
        source_data, _ = await version_core.checkout_data(source_version.version_id)

        if not target_version:
            return source_data

        # ターゲットデータをロード
        target_data, _ = await version_core.checkout_data(target_version.version_id)

        # 戦略に基づくマージ
        if strategy == ConflictResolution.AUTO_LATEST:
            return source_data
        elif strategy == ConflictResolution.AUTO_MERGE:
            # 簡易マージ（実際のプロジェクトでは詳細な実装が必要）
            if isinstance(source_data, pd.DataFrame) and isinstance(
                target_data, pd.DataFrame
            ):
                # DataFrameの場合、行を結合
                return pd.concat([target_data, source_data]).drop_duplicates()
            else:
                return source_data
        else:
            return source_data

    async def _record_merge_operation(
        self,
        merge_version_id: str,
        source_branch: str,
        target_branch: str,
        conflicts: List[VersionConflict],
    ) -> None:
        """
        マージ操作記録
        
        Args:
            merge_version_id: マージバージョンID
            source_branch: マージ元ブランチ
            target_branch: マージ先ブランチ
            conflicts: 競合リスト
        """
        try:
            # 競合情報をデータベース保存
            with sqlite3.connect(self.db_path) as conn:
                for conflict in conflicts:
                    conn.execute(
                        """
                        INSERT INTO conflicts
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

        except Exception as e:
            logger.error(f"マージ操作記録エラー: {e}")

    def _calculate_metadata_diff(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """
        メタデータ差分計算
        
        Args:
            meta1: メタデータ1
            meta2: メタデータ2
            
        Returns:
            メタデータ差分
        """
        diff = {"added": {}, "removed": {}, "changed": {}}

        all_keys = set(meta1.keys()) | set(meta2.keys())

        for key in all_keys:
            if key not in meta1:
                diff["added"][key] = meta2[key]
            elif key not in meta2:
                diff["removed"][key] = meta1[key]
            elif meta1[key] != meta2[key]:
                diff["changed"][key] = {"old": meta1[key], "new": meta2[key]}

        return diff

    def _calculate_data_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """
        データ差分計算
        
        Args:
            data1: データ1
            data2: データ2
            
        Returns:
            データ差分
        """
        diff = {
            "type_changed": not isinstance(data1, type(data2))
            or not isinstance(data2, type(data1)),
            "content_summary": "データが変更されました",
        }

        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            diff.update(
                {
                    "shape_diff": {"old_shape": data1.shape, "new_shape": data2.shape},
                    "columns_diff": {
                        "added": list(set(data2.columns) - set(data1.columns)),
                        "removed": list(set(data1.columns) - set(data2.columns)),
                    },
                    "row_count_diff": len(data2) - len(data1),
                }
            )

        return diff