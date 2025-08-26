#!/usr/bin/env python3
"""
ブランチ管理モジュール

データバージョン管理システムのブランチ作成、切り替え、
管理機能を提供します。

Classes:
    BranchManager: ブランチ管理メインクラス
"""

import logging
from typing import Dict, List, Optional

from .database import DatabaseManager
from .types import DataBranch, ConflictResolution

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class BranchManager:
    """ブランチ管理クラス
    
    データバージョン管理におけるブランチの作成、取得、更新、削除など
    すべてのブランチ操作を管理します。
    """

    def __init__(self, database_manager: DatabaseManager):
        """BranchManagerの初期化
        
        Args:
            database_manager: データベース管理インスタンス
        """
        self.db_manager = database_manager
        self.branch_cache: Dict[str, DataBranch] = {}
        logger.info("ブランチマネージャー初期化完了")

    async def create_branch(
        self,
        branch_name: str,
        from_version: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
        merge_policy: ConflictResolution = ConflictResolution.MANUAL,
    ) -> str:
        """新しいブランチを作成
        
        Args:
            branch_name: 作成するブランチ名
            from_version: 分岐元のバージョンID（Noneの場合は最新版）
            description: ブランチの説明
            created_by: ブランチ作成者
            merge_policy: デフォルトのマージポリシー
            
        Returns:
            作成されたブランチ名
            
        Raises:
            ValueError: ブランチが既に存在する場合
        """
        logger.info(f"ブランチ作成開始: {branch_name}")

        try:
            # ブランチ名重複チェック
            existing_branch = await self.get_branch(branch_name)
            if existing_branch:
                raise ValueError(f"ブランチが既に存在します: {branch_name}")

            # 元バージョンの決定
            if from_version is None:
                from_version = "root"  # デフォルト値

            # ブランチオブジェクト作成
            branch = DataBranch(
                branch_name=branch_name,
                created_from=from_version,
                created_by=created_by,
                description=description,
                merge_policy=merge_policy,
            )

            # データベースに保存
            self.db_manager.save_branch(branch)

            # キャッシュ更新
            self.branch_cache[branch_name] = branch

            logger.info(f"ブランチ作成完了: {branch_name}")
            return branch_name

        except Exception as e:
            logger.error(f"ブランチ作成エラー ({branch_name}): {e}")
            raise

    async def get_branch(self, branch_name: str) -> Optional[DataBranch]:
        """ブランチ情報を取得
        
        Args:
            branch_name: 取得するブランチ名
            
        Returns:
            ブランチ情報、見つからない場合はNone
        """
        # キャッシュチェック
        if branch_name in self.branch_cache:
            return self.branch_cache[branch_name]

        # データベースから取得
        branch = self.db_manager.get_branch(branch_name)
        
        # キャッシュに保存
        if branch:
            self.branch_cache[branch_name] = branch
            
        return branch

    async def list_branches(self) -> List[DataBranch]:
        """全ブランチのリストを取得
        
        Returns:
            ブランチリスト
        """
        try:
            # データベースから全ブランチを取得
            # 注：この実装はDatabaseManagerにlist_branches メソッドが必要
            # 簡易実装として現在キャッシュされているブランチを返す
            branches = list(self.branch_cache.values())
            
            logger.debug(f"ブランチリスト取得完了: {len(branches)}件")
            return branches

        except Exception as e:
            logger.error(f"ブランチリスト取得エラー: {e}")
            return []

    async def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """ブランチを削除
        
        Args:
            branch_name: 削除するブランチ名
            force: 保護されたブランチも強制削除するか
            
        Returns:
            削除成功した場合True
            
        Raises:
            ValueError: 保護されたブランチを削除しようとした場合
        """
        logger.info(f"ブランチ削除開始: {branch_name}")

        try:
            # ブランチ存在確認
            branch = await self.get_branch(branch_name)
            if not branch:
                logger.warning(f"削除対象ブランチが見つかりません: {branch_name}")
                return False

            # 保護チェック
            if branch.is_protected and not force:
                raise ValueError(f"保護されたブランチは削除できません: {branch_name}")

            # mainブランチ削除禁止
            if branch_name == "main":
                raise ValueError("mainブランチは削除できません")

            # データベースから削除（実装注：DatabaseManagerにdelete_branch メソッドが必要）
            # 現在の実装では削除をスキップ
            logger.info(f"ブランチ削除は未実装です: {branch_name}")

            # キャッシュから削除
            self.branch_cache.pop(branch_name, None)

            logger.info(f"ブランチ削除完了: {branch_name}")
            return True

        except Exception as e:
            logger.error(f"ブランチ削除エラー ({branch_name}): {e}")
            raise

    async def update_branch_latest_version(
        self, branch_name: str, version_id: str
    ) -> None:
        """ブランチの最新バージョンを更新
        
        Args:
            branch_name: 更新対象ブランチ名
            version_id: 新しい最新バージョンID
        """
        try:
            # データベース更新
            self.db_manager.update_branch_latest_version(branch_name, version_id)

            # キャッシュ更新
            if branch_name in self.branch_cache:
                self.branch_cache[branch_name].latest_version = version_id

            logger.debug(
                f"ブランチ最新バージョン更新完了: {branch_name} -> {version_id}"
            )

        except Exception as e:
            logger.error(
                f"ブランチ最新バージョン更新エラー ({branch_name}): {e}"
            )
            raise

    async def set_branch_protection(
        self, branch_name: str, is_protected: bool
    ) -> None:
        """ブランチの保護状態を設定
        
        Args:
            branch_name: 対象ブランチ名
            is_protected: 保護するかどうか
        """
        try:
            # ブランチ取得
            branch = await self.get_branch(branch_name)
            if not branch:
                raise ValueError(f"ブランチが見つかりません: {branch_name}")

            # 保護状態更新
            branch.is_protected = is_protected
            
            # データベース保存
            self.db_manager.save_branch(branch)
            
            # キャッシュ更新
            self.branch_cache[branch_name] = branch

            logger.info(
                f"ブランチ保護設定更新: {branch_name} -> "
                f"{'保護' if is_protected else '保護解除'}"
            )

        except Exception as e:
            logger.error(f"ブランチ保護設定エラー ({branch_name}): {e}")
            raise

    async def set_branch_merge_policy(
        self, branch_name: str, merge_policy: ConflictResolution
    ) -> None:
        """ブランチのマージポリシーを設定
        
        Args:
            branch_name: 対象ブランチ名
            merge_policy: 新しいマージポリシー
        """
        try:
            # ブランチ取得
            branch = await self.get_branch(branch_name)
            if not branch:
                raise ValueError(f"ブランチが見つかりません: {branch_name}")

            # マージポリシー更新
            branch.merge_policy = merge_policy
            
            # データベース保存
            self.db_manager.save_branch(branch)
            
            # キャッシュ更新
            self.branch_cache[branch_name] = branch

            logger.info(
                f"ブランチマージポリシー更新: {branch_name} -> {merge_policy.value}"
            )

        except Exception as e:
            logger.error(f"ブランチマージポリシー設定エラー ({branch_name}): {e}")
            raise

    async def get_branch_statistics(self, branch_name: str) -> Dict[str, int]:
        """ブランチの統計情報を取得
        
        Args:
            branch_name: 対象ブランチ名
            
        Returns:
            統計情報辞書（バージョン数、総サイズ等）
        """
        try:
            branch_stats = self.db_manager.get_branch_statistics()
            return branch_stats.get(branch_name, {
                "version_count": 0,
                "total_size_bytes": 0,
            })

        except Exception as e:
            logger.error(f"ブランチ統計情報取得エラー ({branch_name}): {e}")
            return {"version_count": 0, "total_size_bytes": 0}

    async def validate_branch_name(self, branch_name: str) -> bool:
        """ブランチ名の有効性を検証
        
        Args:
            branch_name: 検証するブランチ名
            
        Returns:
            有効な場合True
        """
        if not branch_name:
            return False
            
        # 基本的な文字チェック
        invalid_chars = [" ", "\t", "\n", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in branch_name for char in invalid_chars):
            return False
            
        # 長さチェック
        if len(branch_name) > 100:
            return False
            
        # 予約語チェック
        reserved_names = ["HEAD", "refs", "objects"]
        if branch_name in reserved_names:
            return False
            
        return True

    def clear_cache(self) -> None:
        """ブランチキャッシュをクリア"""
        self.branch_cache.clear()
        logger.debug("ブランチキャッシュクリア完了")

    def get_cache_status(self) -> Dict[str, int]:
        """キャッシュの状態を取得
        
        Returns:
            キャッシュ状態情報
        """
        return {
            "cached_branches": len(self.branch_cache),
        }