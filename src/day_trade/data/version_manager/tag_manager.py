#!/usr/bin/env python3
"""
タグ管理モジュール

データバージョン管理システムのタグ作成、取得、削除機能を提供します。
リリース管理やバージョンのマーキングに使用されます。

Classes:
    TagManager: タグ管理メインクラス
"""

import logging
import re
from typing import Dict, List, Optional

from .database import DatabaseManager
from .types import DataTag, DataVersion

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class TagManager:
    """タグ管理クラス
    
    データバージョンに対するタグの作成、取得、削除、検索機能を提供します。
    セマンティックバージョニング（SemVer）にも対応しています。
    """

    def __init__(self, database_manager: DatabaseManager):
        """TagManagerの初期化
        
        Args:
            database_manager: データベース管理インスタンス
        """
        self.db_manager = database_manager
        self.tag_cache: Dict[str, DataTag] = {}
        logger.info("タグマネージャー初期化完了")

    async def create_tag(
        self,
        tag_name: str,
        version_id: str,
        description: str = "",
        is_release: bool = False,
        created_by: str = "system",
    ) -> str:
        """新しいタグを作成
        
        Args:
            tag_name: タグ名
            version_id: 対象バージョンID
            description: タグの説明
            is_release: リリースタグかどうか
            created_by: タグ作成者
            
        Returns:
            作成されたタグ名
            
        Raises:
            ValueError: タグが既に存在するか、バージョンが見つからない場合
        """
        logger.info(f"タグ作成開始: {tag_name} -> {version_id}")

        try:
            # タグ名バリデーション
            if not self.validate_tag_name(tag_name):
                raise ValueError(f"無効なタグ名です: {tag_name}")

            # タグ重複チェック
            existing_tag = await self.get_tag(tag_name)
            if existing_tag:
                raise ValueError(f"タグが既に存在します: {tag_name}")

            # バージョン存在確認
            version = self.db_manager.get_version(version_id)
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
            self.db_manager.save_tag(tag)

            # バージョンにタグ情報を追加
            version.tag = tag_name
            self.db_manager.update_version(version)

            # キャッシュ更新
            self.tag_cache[tag_name] = tag

            logger.info(f"タグ作成完了: {tag_name}")
            return tag_name

        except Exception as e:
            logger.error(f"タグ作成エラー ({tag_name}): {e}")
            raise

    async def get_tag(self, tag_name: str) -> Optional[DataTag]:
        """タグ情報を取得
        
        Args:
            tag_name: 取得するタグ名
            
        Returns:
            タグ情報、見つからない場合はNone
        """
        # キャッシュチェック
        if tag_name in self.tag_cache:
            return self.tag_cache[tag_name]

        # データベースから取得
        tag = self.db_manager.get_tag(tag_name)
        
        # キャッシュに保存
        if tag:
            self.tag_cache[tag_name] = tag
            
        return tag

    async def list_tags(self, version_filter: Optional[str] = None) -> List[DataTag]:
        """タグリストを取得
        
        Args:
            version_filter: フィルタリングするバージョンID（任意）
            
        Returns:
            タグのリスト
        """
        try:
            # 注：実際の実装ではDatabaseManagerにlist_tags メソッドが必要
            # 現在はキャッシュからフィルタリングして返す
            tags = list(self.tag_cache.values())
            
            if version_filter:
                tags = [tag for tag in tags if tag.version_id == version_filter]
                
            # 作成日時でソート
            tags.sort(key=lambda t: t.created_at, reverse=True)
            
            logger.debug(f"タグリスト取得完了: {len(tags)}件")
            return tags

        except Exception as e:
            logger.error(f"タグリスト取得エラー: {e}")
            return []

    async def list_release_tags(self) -> List[DataTag]:
        """リリースタグのみをリスト取得
        
        Returns:
            リリースタグのリスト
        """
        try:
            all_tags = await self.list_tags()
            release_tags = [tag for tag in all_tags if tag.is_release]
            
            logger.debug(f"リリースタグリスト取得完了: {len(release_tags)}件")
            return release_tags

        except Exception as e:
            logger.error(f"リリースタグリスト取得エラー: {e}")
            return []

    async def delete_tag(self, tag_name: str, force: bool = False) -> bool:
        """タグを削除
        
        Args:
            tag_name: 削除するタグ名
            force: リリースタグも強制削除するか
            
        Returns:
            削除成功した場合True
            
        Raises:
            ValueError: リリースタグを強制削除しない場合
        """
        logger.info(f"タグ削除開始: {tag_name}")

        try:
            # タグ存在確認
            tag = await self.get_tag(tag_name)
            if not tag:
                logger.warning(f"削除対象タグが見つかりません: {tag_name}")
                return False

            # リリースタグ保護チェック
            if tag.is_release and not force:
                raise ValueError(f"リリースタグは削除できません: {tag_name}")

            # バージョンからタグ情報を削除
            version = self.db_manager.get_version(tag.version_id)
            if version and version.tag == tag_name:
                version.tag = None
                self.db_manager.update_version(version)

            # データベースから削除
            # 注：実際の実装ではDatabaseManagerにdelete_tag メソッドが必要
            logger.info(f"タグ削除は未実装です: {tag_name}")

            # キャッシュから削除
            self.tag_cache.pop(tag_name, None)

            logger.info(f"タグ削除完了: {tag_name}")
            return True

        except Exception as e:
            logger.error(f"タグ削除エラー ({tag_name}): {e}")
            raise

    async def find_tags_by_pattern(self, pattern: str) -> List[DataTag]:
        """パターンマッチングでタグを検索
        
        Args:
            pattern: 検索パターン（正規表現対応）
            
        Returns:
            マッチしたタグのリスト
        """
        try:
            all_tags = await self.list_tags()
            regex = re.compile(pattern, re.IGNORECASE)
            
            matching_tags = [
                tag for tag in all_tags 
                if regex.search(tag.tag_name) or regex.search(tag.description)
            ]
            
            logger.debug(f"パターン検索完了 ({pattern}): {len(matching_tags)}件")
            return matching_tags

        except Exception as e:
            logger.error(f"タグパターン検索エラー ({pattern}): {e}")
            return []

    async def get_latest_release_tag(self) -> Optional[DataTag]:
        """最新のリリースタグを取得
        
        Returns:
            最新のリリースタグ、見つからない場合はNone
        """
        try:
            release_tags = await self.list_release_tags()
            
            if not release_tags:
                return None
                
            # セマンティックバージョニング対応ソート
            semver_tags = []
            other_tags = []
            
            for tag in release_tags:
                if self.is_semantic_version(tag.tag_name):
                    semver_tags.append(tag)
                else:
                    other_tags.append(tag)
            
            # セマンティックバージョンを優先
            if semver_tags:
                semver_tags.sort(key=lambda t: self.parse_semantic_version(t.tag_name), reverse=True)
                return semver_tags[0]
            elif other_tags:
                # セマンティックバージョンがない場合は作成日時で判定
                other_tags.sort(key=lambda t: t.created_at, reverse=True)
                return other_tags[0]
            
            return None

        except Exception as e:
            logger.error(f"最新リリースタグ取得エラー: {e}")
            return None

    def validate_tag_name(self, tag_name: str) -> bool:
        """タグ名の有効性を検証
        
        Args:
            tag_name: 検証するタグ名
            
        Returns:
            有効な場合True
        """
        if not tag_name or len(tag_name) == 0:
            return False
            
        # 長さチェック
        if len(tag_name) > 100:
            return False
            
        # 基本的な文字チェック
        invalid_chars = [" ", "\t", "\n", ":", "*", "?", '"', "<", ">", "|", "\\", "/"]
        if any(char in tag_name for char in invalid_chars):
            return False
            
        # 制御文字チェック
        if any(ord(char) < 32 for char in tag_name):
            return False
            
        return True

    def is_semantic_version(self, tag_name: str) -> bool:
        """セマンティックバージョニング形式かチェック
        
        Args:
            tag_name: チェックするタグ名
            
        Returns:
            セマンティックバージョン形式の場合True
        """
        # v1.2.3 または 1.2.3 形式をチェック
        semver_pattern = r'^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*))?(?:\+([a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*))?$'
        return bool(re.match(semver_pattern, tag_name))

    def parse_semantic_version(self, tag_name: str) -> tuple:
        """セマンティックバージョンをパース
        
        Args:
            tag_name: パースするタグ名
            
        Returns:
            (major, minor, patch, prerelease, build) のタプル
        """
        semver_pattern = r'^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*))?(?:\+([a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*))?$'
        match = re.match(semver_pattern, tag_name)
        
        if match:
            major, minor, patch, prerelease, build = match.groups()
            return (
                int(major),
                int(minor), 
                int(patch),
                prerelease or "",
                build or ""
            )
        
        # パースできない場合は最低値を返す
        return (0, 0, 0, "", "")

    async def suggest_next_version(self, current_tag: str, bump_type: str = "patch") -> str:
        """次のバージョン番号を提案（セマンティックバージョニング）
        
        Args:
            current_tag: 現在のタグ名
            bump_type: バンプタイプ（major, minor, patch）
            
        Returns:
            提案される次のバージョン
        """
        try:
            if not self.is_semantic_version(current_tag):
                return f"v1.0.0"  # デフォルト
                
            major, minor, patch, _, _ = self.parse_semantic_version(current_tag)
            
            if bump_type == "major":
                return f"v{major + 1}.0.0"
            elif bump_type == "minor":
                return f"v{major}.{minor + 1}.0"
            else:  # patch
                return f"v{major}.{minor}.{patch + 1}"
                
        except Exception as e:
            logger.error(f"次バージョン提案エラー ({current_tag}): {e}")
            return "v1.0.0"

    def get_tag_statistics(self) -> Dict[str, int]:
        """タグ統計情報を取得
        
        Returns:
            タグ統計情報
        """
        try:
            total_tags = len(self.tag_cache)
            release_tags = len([tag for tag in self.tag_cache.values() if tag.is_release])
            semver_tags = len([
                tag for tag in self.tag_cache.values() 
                if self.is_semantic_version(tag.tag_name)
            ])
            
            return {
                "total_tags": total_tags,
                "release_tags": release_tags,
                "semantic_version_tags": semver_tags,
                "cached_tags": len(self.tag_cache),
            }
            
        except Exception as e:
            logger.error(f"タグ統計情報取得エラー: {e}")
            return {
                "total_tags": 0,
                "release_tags": 0,
                "semantic_version_tags": 0,
                "cached_tags": 0,
            }

    def clear_cache(self) -> None:
        """タグキャッシュをクリア"""
        self.tag_cache.clear()
        logger.debug("タグキャッシュクリア完了")