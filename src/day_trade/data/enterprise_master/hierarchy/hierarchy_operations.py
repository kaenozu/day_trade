#!/usr/bin/env python3
"""
階層基本操作モジュール

階層の作成、削除、関係追加・削除などの基本操作を提供します。
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..enums import HierarchyType, MasterDataType
from ..models import MasterDataHierarchy


class HierarchyOperations:
    """階層基本操作クラス
    
    階層の CRUD 操作と基本的な関係管理機能を提供します。
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._hierarchies: Dict[str, MasterDataHierarchy] = {}

    async def create_hierarchy(
        self,
        name: str,
        entity_type: MasterDataType,
        root_entity_id: str,
        level_definitions: Dict[int, str],
        hierarchy_type: HierarchyType = HierarchyType.CLASSIFICATION,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """データ階層作成
        
        Args:
            name: 階層名
            entity_type: エンティティタイプ
            root_entity_id: ルートエンティティID
            level_definitions: レベル定義（レベル番号: 説明）
            hierarchy_type: 階層タイプ
            created_by: 作成者
            metadata: メタデータ
            
        Returns:
            Tuple[bool, str, Optional[str]]: (成功フラグ, 階層ID, エラーメッセージ)
        """
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # 階層ID生成
            hierarchy_id = f"hierarchy_{entity_type.value}_{int(time.time())}"

            # 階層作成
            hierarchy = MasterDataHierarchy(
                hierarchy_id=hierarchy_id,
                name=name,
                entity_type=entity_type,
                root_entity_id=root_entity_id,
                level_definitions=level_definitions,
                metadata={
                    **metadata,
                    "hierarchy_type": hierarchy_type.value,
                    "created_by": created_by,
                    "created_at": current_time.isoformat(),
                }
            )

            # 保存
            self._hierarchies[hierarchy_id] = hierarchy

            self.logger.info(f"データ階層作成: {hierarchy_id}")
            return True, hierarchy_id, None

        except Exception as e:
            self.logger.error(f"階層作成エラー: {e}")
            return False, "", str(e)

    async def delete_hierarchy(self, hierarchy_id: str) -> Tuple[bool, Optional[str]]:
        """階層削除
        
        Args:
            hierarchy_id: 階層ID
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            if hierarchy_id not in self._hierarchies:
                return False, f"階層が見つかりません: {hierarchy_id}"

            del self._hierarchies[hierarchy_id]
            
            self.logger.info(f"階層削除: {hierarchy_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"階層削除エラー: {e}")
            return False, str(e)

    async def add_child_relationship(
        self,
        hierarchy_id: str,
        parent_id: str,
        child_id: str
    ) -> Tuple[bool, Optional[str]]:
        """親子関係追加
        
        Args:
            hierarchy_id: 階層ID
            parent_id: 親エンティティID
            child_id: 子エンティティID
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, f"階層が見つかりません: {hierarchy_id}"

            # 循環参照チェック（ここでは簡単なチェック）
            if await self._would_create_cycle(hierarchy, parent_id, child_id):
                return False, f"循環参照が発生するため追加できません: {parent_id} -> {child_id}"

            # 親子関係追加
            hierarchy.add_child(parent_id, child_id)

            self.logger.info(f"親子関係追加: {hierarchy_id}, {parent_id} -> {child_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"親子関係追加エラー: {e}")
            return False, str(e)

    async def remove_child_relationship(
        self,
        hierarchy_id: str,
        parent_id: str,
        child_id: str
    ) -> Tuple[bool, Optional[str]]:
        """親子関係削除
        
        Args:
            hierarchy_id: 階層ID
            parent_id: 親エンティティID
            child_id: 子エンティティID
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, f"階層が見つかりません: {hierarchy_id}"

            # 親子関係削除
            children = hierarchy.parent_child_mapping.get(parent_id, [])
            if child_id in children:
                children.remove(child_id)
                if not children:
                    del hierarchy.parent_child_mapping[parent_id]

            self.logger.info(f"親子関係削除: {hierarchy_id}, {parent_id} -> {child_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"親子関係削除エラー: {e}")
            return False, str(e)

    async def list_hierarchies(
        self, entity_type: Optional[MasterDataType] = None
    ) -> List[MasterDataHierarchy]:
        """階層一覧取得
        
        Args:
            entity_type: エンティティタイプフィルタ
            
        Returns:
            List[MasterDataHierarchy]: 階層一覧
        """
        try:
            hierarchies = []
            
            for hierarchy in self._hierarchies.values():
                if entity_type and hierarchy.entity_type != entity_type:
                    continue
                hierarchies.append(hierarchy)
            
            return hierarchies

        except Exception as e:
            self.logger.error(f"階層一覧取得エラー: {e}")
            return []

    async def get_hierarchy(self, hierarchy_id: str) -> Optional[MasterDataHierarchy]:
        """階層取得
        
        Args:
            hierarchy_id: 階層ID
            
        Returns:
            Optional[MasterDataHierarchy]: 階層オブジェクト
        """
        return self._hierarchies.get(hierarchy_id)

    async def update_hierarchy_metadata(
        self,
        hierarchy_id: str,
        metadata_updates: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """階層メタデータ更新
        
        Args:
            hierarchy_id: 階層ID
            metadata_updates: 更新するメタデータ
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, f"階層が見つかりません: {hierarchy_id}"

            # メタデータ更新
            hierarchy.metadata.update(metadata_updates)
            hierarchy.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

            self.logger.info(f"階層メタデータ更新: {hierarchy_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"階層メタデータ更新エラー: {e}")
            return False, str(e)

    async def _would_create_cycle(
        self, hierarchy: MasterDataHierarchy, parent_id: str, child_id: str
    ) -> bool:
        """循環参照チェック（基本版）
        
        Args:
            hierarchy: 階層オブジェクト
            parent_id: 親エンティティID
            child_id: 子エンティティID
            
        Returns:
            bool: 循環参照が発生するかどうか
        """
        try:
            # child_idから遡ってparent_idに到達できるかチェック
            visited = set()
            current_id = parent_id

            while current_id:
                if current_id == child_id:
                    return True
                
                if current_id in visited:
                    break
                
                visited.add(current_id)
                
                # 親を見つける
                parent = None
                for p_id, children in hierarchy.parent_child_mapping.items():
                    if current_id in children:
                        parent = p_id
                        break
                
                current_id = parent

            return False

        except Exception as e:
            self.logger.error(f"循環参照チェックエラー: {e}")
            return True  # エラーの場合は安全側に倒す