#!/usr/bin/env python3
"""
データバージョン管理システム - バージョン操作統合

Issue #420: データ管理とデータ品質保証メカニズムの強化

バージョン操作の統合インターフェースを提供
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .models import DataVersion
from .version_core import VersionCore
from .version_db_operations import VersionDbOperations

try:
    from ...utils.unified_cache_manager import UnifiedCacheManager
except ImportError:
    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True


class VersionOperations:
    """バージョン操作統合クラス"""

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
        self.version_core = VersionCore(
            repository_path, cache_manager, max_versions, auto_backup
        )
        self.db_operations = VersionDbOperations(repository_path, cache_manager)

    async def commit_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        message: str,
        author: str = "system",
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """データコミット"""
        return await self.version_core.commit_data(
            data, message, author, branch, metadata
        )

    async def checkout_data(self, version_id: str) -> Tuple[Any, DataVersion]:
        """データチェックアウト"""
        return await self.version_core.checkout_data(version_id)

    async def get_version_history(
        self, branch: Optional[str] = None, limit: int = 50
    ) -> List[DataVersion]:
        """バージョン履歴取得"""
        return await self.version_core.get_version_history(branch, limit)

    async def _get_version(self, version_id: str) -> Optional[DataVersion]:
        """バージョン取得（内部使用）"""
        return await self.db_operations.get_version(version_id)

    async def _get_latest_version(self, branch: str) -> Optional[DataVersion]:
        """最新バージョン取得（内部使用）"""
        return await self.db_operations.get_latest_version(branch)

    async def _list_versions(self, branch: str) -> List[DataVersion]:
        """バージョンリスト取得（内部使用）"""
        return await self.db_operations.list_versions(branch)

    async def _find_version_by_hash(
        self, data_hash: str, branch: str
    ) -> Optional[DataVersion]:
        """ハッシュによるバージョン検索（内部使用）"""
        return await self.db_operations.find_version_by_hash(data_hash, branch)

    @property
    def version_cache(self) -> Dict[str, DataVersion]:
        """バージョンキャッシュ（後方互換性用）"""
        return self.version_core.version_cache