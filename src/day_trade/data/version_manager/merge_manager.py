#!/usr/bin/env python3
"""
マージ管理モジュール

データバージョン管理システムのブランチマージ、競合検出、
競合解決機能を提供します。

Classes:
    MergeManager: マージ管理メインクラス
"""

import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import pandas as pd

from .data_operations import DataOperations
from .database import DatabaseManager
from .types import ConflictResolution, DataVersion, VersionConflict

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class MergeManager:
    """マージ管理クラス
    
    ブランチ間のマージ、競合検出、自動解決機能を提供します。
    """

    def __init__(
        self, 
        database_manager: DatabaseManager, 
        data_operations: DataOperations
    ):
        """MergeManagerの初期化
        
        Args:
            database_manager: データベース管理インスタンス
            data_operations: データ操作インスタンス
        """
        self.db_manager = database_manager
        self.data_ops = data_operations
        logger.info("マージマネージャー初期化完了")

    async def merge_branches(
        self,
        source_version: DataVersion,
        target_version: Optional[DataVersion],
        strategy: ConflictResolution = ConflictResolution.MANUAL,
        author: str = "system",
    ) -> Tuple[Any, List[VersionConflict]]:
        """ブランチをマージしてマージデータと競合情報を返す
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン（Noneの場合は空ブランチ）
            strategy: 競合解決戦略
            author: マージ実行者
            
        Returns:
            (マージ後のデータ, 競合リスト)のタプル
        """
        logger.info(
            f"ブランチマージ開始: "
            f"{source_version.version_id} -> "
            f"{target_version.version_id if target_version else 'empty'}"
        )

        try:
            # 競合検出
            conflicts = await self._detect_merge_conflicts(
                source_version, target_version
            )

            # 手動解決が必要な場合
            if conflicts and strategy == ConflictResolution.MANUAL:
                logger.warning(f"マージ競合が検出されました: {len(conflicts)}件")
                return None, conflicts

            # 自動競合解決
            if conflicts:
                conflicts = await self._resolve_conflicts_automatically(
                    conflicts, strategy, author
                )

            # マージデータ作成
            merge_data = await self._create_merge_data(
                source_version, target_version, strategy
            )

            # 競合情報をデータベースに記録
            for conflict in conflicts:
                self.db_manager.save_conflict(conflict)

            logger.info(
                f"ブランチマージ完了: 競合{len(conflicts)}件 "
                f"(解決済み: {len([c for c in conflicts if c.resolved])}件)"
            )

            return merge_data, conflicts

        except Exception as e:
            logger.error(f"ブランチマージエラー: {e}")
            raise

    async def _detect_merge_conflicts(
        self, source_version: DataVersion, target_version: Optional[DataVersion]
    ) -> List[VersionConflict]:
        """マージ競合を検出
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン
            
        Returns:
            検出された競合のリスト
        """
        conflicts = []

        if not target_version:
            return conflicts  # ターゲットが空の場合は競合なし

        # データハッシュ競合チェック
        if source_version.data_hash != target_version.data_hash:
            conflict = VersionConflict(
                conflict_id=f"data_conflict_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="data",
                description="データ内容が異なります",
            )
            conflicts.append(conflict)

        # メタデータ競合チェック
        metadata_conflicts = await self._detect_metadata_conflicts(
            source_version, target_version
        )
        conflicts.extend(metadata_conflicts)

        # スキーマ競合チェック（DataFrame の場合）
        schema_conflicts = await self._detect_schema_conflicts(
            source_version, target_version
        )
        conflicts.extend(schema_conflicts)

        logger.debug(f"競合検出完了: {len(conflicts)}件")
        return conflicts

    async def _detect_metadata_conflicts(
        self, source_version: DataVersion, target_version: DataVersion
    ) -> List[VersionConflict]:
        """メタデータ競合を検出
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン
            
        Returns:
            メタデータ競合のリスト
        """
        conflicts = []
        source_meta = source_version.metadata
        target_meta = target_version.metadata

        # 競合するキーを特定
        conflicting_keys = []
        for key in set(source_meta.keys()) & set(target_meta.keys()):
            if source_meta[key] != target_meta[key]:
                conflicting_keys.append(key)

        if conflicting_keys:
            conflict = VersionConflict(
                conflict_id=f"metadata_conflict_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="metadata",
                description=f"メタデータ競合: {', '.join(conflicting_keys)}",
            )
            conflicts.append(conflict)

        return conflicts

    async def _detect_schema_conflicts(
        self, source_version: DataVersion, target_version: DataVersion
    ) -> List[VersionConflict]:
        """スキーマ競合を検出（DataFrame専用）
        
        Args:
            source_version: マージ元バージョン  
            target_version: マージ先バージョン
            
        Returns:
            スキーマ競合のリスト
        """
        conflicts = []

        try:
            # データを読み込んで形式チェック
            source_data = await self.data_ops.load_version_data(
                source_version.version_id
            )
            target_data = await self.data_ops.load_version_data(
                target_version.version_id
            )

            # 両方がDataFrameの場合のみスキーマチェック
            if isinstance(source_data, pd.DataFrame) and isinstance(
                target_data, pd.DataFrame
            ):
                # カラム競合チェック
                source_cols = set(source_data.columns)
                target_cols = set(target_data.columns)

                if source_cols != target_cols:
                    conflict = VersionConflict(
                        conflict_id=f"schema_conflict_{int(time.time())}",
                        source_version=source_version.version_id,
                        target_version=target_version.version_id,
                        conflict_type="schema",
                        description=(
                            f"カラム構造が異なります: "
                            f"追加={source_cols - target_cols}, "
                            f"削除={target_cols - source_cols}"
                        ),
                    )
                    conflicts.append(conflict)

                # データ型競合チェック（共通カラムのみ）
                common_cols = source_cols & target_cols
                dtype_conflicts = []
                
                for col in common_cols:
                    if source_data[col].dtype != target_data[col].dtype:
                        dtype_conflicts.append(
                            f"{col}({source_data[col].dtype}->{target_data[col].dtype})"
                        )

                if dtype_conflicts:
                    conflict = VersionConflict(
                        conflict_id=f"dtype_conflict_{int(time.time())}",
                        source_version=source_version.version_id,
                        target_version=target_version.version_id,
                        conflict_type="dtype",
                        description=f"データ型競合: {', '.join(dtype_conflicts)}",
                    )
                    conflicts.append(conflict)

        except Exception as e:
            logger.warning(f"スキーマ競合検出中のエラー: {e}")

        return conflicts

    async def _resolve_conflicts_automatically(
        self,
        conflicts: List[VersionConflict],
        strategy: ConflictResolution,
        resolved_by: str,
    ) -> List[VersionConflict]:
        """競合を自動解決
        
        Args:
            conflicts: 解決する競合リスト
            strategy: 解決戦略
            resolved_by: 解決実行者
            
        Returns:
            解決処理後の競合リスト
        """
        resolved_conflicts = []

        for conflict in conflicts:
            if strategy == ConflictResolution.AUTO_LATEST:
                # 最新（ソース）を採用
                conflict.mark_resolved(resolved_by, ConflictResolution.AUTO_LATEST)
                logger.debug(f"自動解決（最新採用）: {conflict.conflict_id}")

            elif strategy == ConflictResolution.AUTO_MERGE:
                # 自動マージ戦略
                conflict.mark_resolved(resolved_by, ConflictResolution.AUTO_MERGE)
                logger.debug(f"自動解決（マージ）: {conflict.conflict_id}")

            elif strategy == ConflictResolution.KEEP_BOTH:
                # 両方保持（実装は簡易版）
                conflict.mark_resolved(resolved_by, ConflictResolution.KEEP_BOTH)
                logger.debug(f"自動解決（両方保持）: {conflict.conflict_id}")

            resolved_conflicts.append(conflict)

        return resolved_conflicts

    async def _create_merge_data(
        self,
        source_version: DataVersion,
        target_version: Optional[DataVersion],
        strategy: ConflictResolution,
    ) -> Any:
        """マージデータを作成
        
        Args:
            source_version: マージ元バージョン
            target_version: マージ先バージョン
            strategy: マージ戦略
            
        Returns:
            マージされたデータ
        """
        # ソースデータをロード
        source_data = await self.data_ops.load_version_data(
            source_version.version_id
        )

        if not target_version:
            return source_data

        # ターゲットデータをロード
        target_data = await self.data_ops.load_version_data(
            target_version.version_id
        )

        # 戦略に基づくマージ実行
        return await self._execute_merge_strategy(
            source_data, target_data, strategy
        )

    async def _execute_merge_strategy(
        self, source_data: Any, target_data: Any, strategy: ConflictResolution
    ) -> Any:
        """マージ戦略を実行
        
        Args:
            source_data: マージ元データ
            target_data: マージ先データ  
            strategy: マージ戦略
            
        Returns:
            マージ結果データ
        """
        if strategy == ConflictResolution.AUTO_LATEST:
            # 最新データを採用
            return source_data

        elif strategy == ConflictResolution.AUTO_MERGE:
            # データ型に応じた自動マージ
            return await self._auto_merge_data(source_data, target_data)

        elif strategy == ConflictResolution.KEEP_BOTH:
            # 両方を保持（辞書でラップ）
            return {
                "source": source_data,
                "target": target_data,
                "merge_timestamp": datetime.utcnow().isoformat(),
            }

        else:
            # デフォルトは最新データ
            return source_data

    async def _auto_merge_data(self, source_data: Any, target_data: Any) -> Any:
        """データの自動マージを実行
        
        Args:
            source_data: マージ元データ
            target_data: マージ先データ
            
        Returns:
            マージされたデータ
        """
        try:
            # DataFrame同士のマージ
            if isinstance(source_data, pd.DataFrame) and isinstance(
                target_data, pd.DataFrame
            ):
                return await self._merge_dataframes(source_data, target_data)

            # 辞書同士のマージ
            elif isinstance(source_data, dict) and isinstance(target_data, dict):
                merged_dict = target_data.copy()
                merged_dict.update(source_data)  # ソースが優先
                return merged_dict

            # リスト同士のマージ
            elif isinstance(source_data, list) and isinstance(target_data, list):
                # 重複を除いて結合
                return list(dict.fromkeys(target_data + source_data))

            # その他の場合はソースを優先
            else:
                return source_data

        except Exception as e:
            logger.warning(f"自動マージ失敗、ソースデータを採用: {e}")
            return source_data

    async def _merge_dataframes(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> pd.DataFrame:
        """DataFrameの自動マージ
        
        Args:
            source_df: マージ元DataFrame
            target_df: マージ先DataFrame
            
        Returns:
            マージされたDataFrame
        """
        try:
            # インデックスが同じ場合は行結合
            if source_df.index.equals(target_df.index):
                # カラムをマージ（ソース優先）
                merged_df = target_df.copy()
                for col in source_df.columns:
                    merged_df[col] = source_df[col]
                return merged_df

            else:
                # インデックスが異なる場合は縦結合（重複除去）
                combined_df = pd.concat([target_df, source_df], ignore_index=True)
                
                # 可能な場合は重複行を除去
                try:
                    return combined_df.drop_duplicates().reset_index(drop=True)
                except Exception:
                    # 重複除去に失敗した場合はそのまま返す
                    return combined_df

        except Exception as e:
            logger.warning(f"DataFrame マージエラー: {e}")
            # マージに失敗した場合はソースを返す
            return source_df

    async def resolve_conflict_manually(
        self,
        conflict_id: str,
        resolution_strategy: ConflictResolution,
        resolved_by: str,
        resolution_data: Optional[Any] = None,
    ) -> bool:
        """競合を手動解決
        
        Args:
            conflict_id: 競合ID
            resolution_strategy: 解決戦略
            resolved_by: 解決実行者
            resolution_data: 解決用データ（任意）
            
        Returns:
            解決成功した場合True
        """
        try:
            # 競合情報を作成（簡易実装）
            conflict = VersionConflict(
                conflict_id=conflict_id,
                source_version="",
                target_version="",
                conflict_type="manual",
                description="手動解決済み",
            )
            
            conflict.mark_resolved(resolved_by, resolution_strategy)
            
            # データベースに保存
            self.db_manager.save_conflict(conflict)

            logger.info(f"手動競合解決完了: {conflict_id}")
            return True

        except Exception as e:
            logger.error(f"手動競合解決エラー ({conflict_id}): {e}")
            return False

    def get_merge_statistics(self) -> dict:
        """マージ統計情報を取得
        
        Returns:
            マージ統計情報
        """
        # 実装注：実際のプロジェクトではデータベースから統計を取得
        return {
            "total_merges": 0,
            "successful_merges": 0,
            "conflicts_detected": 0,
            "auto_resolved_conflicts": 0,
            "manual_resolved_conflicts": 0,
        }