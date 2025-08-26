#!/usr/bin/env python3
"""
階層ナビゲーションモジュール

階層内のエンティティ間の関係を辿るナビゲーション機能を提供します。
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..models import MasterDataHierarchy


class HierarchyNavigation:
    """階層ナビゲーションクラス
    
    階層内での移動、関係の探索、パス計算などを提供します。
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._hierarchies: Optional[Dict[str, MasterDataHierarchy]] = None
    
    def set_hierarchies_ref(self, hierarchies_ref: Dict[str, MasterDataHierarchy]):
        """階層データ参照設定
        
        Args:
            hierarchies_ref: 階層データへの参照
        """
        self._hierarchies = hierarchies_ref

    async def get_children(
        self, hierarchy_id: str, parent_id: str
    ) -> Tuple[bool, List[str], Optional[str]]:
        """子エンティティ取得
        
        Args:
            hierarchy_id: 階層ID
            parent_id: 親エンティティID
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: (成功フラグ, 子エンティティIDリスト, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, [], "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, [], f"階層が見つかりません: {hierarchy_id}"

            children = hierarchy.get_children(parent_id)
            return True, children, None

        except Exception as e:
            self.logger.error(f"子エンティティ取得エラー: {e}")
            return False, [], str(e)

    async def get_descendants(
        self, hierarchy_id: str, ancestor_id: str
    ) -> Tuple[bool, List[str], Optional[str]]:
        """子孫エンティティ取得（再帰的）
        
        Args:
            hierarchy_id: 階層ID
            ancestor_id: 祖先エンティティID
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: (成功フラグ, 子孫エンティティIDリスト, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, [], "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, [], f"階層が見つかりません: {hierarchy_id}"

            descendants = []
            visited = set()

            def collect_descendants(current_id: str):
                if current_id in visited:
                    return
                visited.add(current_id)
                
                children = hierarchy.get_children(current_id)
                for child_id in children:
                    descendants.append(child_id)
                    collect_descendants(child_id)

            collect_descendants(ancestor_id)
            return True, descendants, None

        except Exception as e:
            self.logger.error(f"子孫エンティティ取得エラー: {e}")
            return False, [], str(e)

    async def get_ancestors(
        self, hierarchy_id: str, descendant_id: str
    ) -> Tuple[bool, List[str], Optional[str]]:
        """祖先エンティティ取得
        
        Args:
            hierarchy_id: 階層ID
            descendant_id: 子孫エンティティID
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: (成功フラグ, 祖先エンティティIDリスト, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, [], "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, [], f"階層が見つかりません: {hierarchy_id}"

            ancestors = []
            
            # 逆引き辞書作成（子 -> 親）
            child_to_parent = {}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                for child_id in children:
                    child_to_parent[child_id] = parent_id

            # 祖先をたどる
            current_id = descendant_id
            while current_id in child_to_parent:
                parent_id = child_to_parent[current_id]
                ancestors.append(parent_id)
                current_id = parent_id

            ancestors.reverse()  # ルートから順に並べる
            return True, ancestors, None

        except Exception as e:
            self.logger.error(f"祖先エンティティ取得エラー: {e}")
            return False, [], str(e)

    async def get_siblings(
        self, hierarchy_id: str, entity_id: str
    ) -> Tuple[bool, List[str], Optional[str]]:
        """兄弟エンティティ取得
        
        Args:
            hierarchy_id: 階層ID
            entity_id: エンティティID
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: (成功フラグ, 兄弟エンティティIDリスト, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, [], "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, [], f"階層が見つかりません: {hierarchy_id}"

            # 親を見つける
            parent_id = None
            for parent, children in hierarchy.parent_child_mapping.items():
                if entity_id in children:
                    parent_id = parent
                    break

            if parent_id is None:
                return True, [], None  # 親がいない場合は兄弟もいない

            # 兄弟取得（自分自身を除く）
            siblings = [
                child_id for child_id in hierarchy.get_children(parent_id)
                if child_id != entity_id
            ]

            return True, siblings, None

        except Exception as e:
            self.logger.error(f"兄弟エンティティ取得エラー: {e}")
            return False, [], str(e)

    async def get_hierarchy_level(
        self, hierarchy_id: str, entity_id: str
    ) -> Tuple[bool, int, Optional[str]]:
        """階層レベル取得
        
        Args:
            hierarchy_id: 階層ID
            entity_id: エンティティID
            
        Returns:
            Tuple[bool, int, Optional[str]]: (成功フラグ, レベル番号, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, -1, "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, -1, f"階層が見つかりません: {hierarchy_id}"

            # ルートからの距離を計算
            if entity_id == hierarchy.root_entity_id:
                return True, 0, None

            success, ancestors, error_msg = await self.get_ancestors(hierarchy_id, entity_id)
            if not success:
                return False, -1, error_msg

            level = len(ancestors)
            return True, level, None

        except Exception as e:
            self.logger.error(f"階層レベル取得エラー: {e}")
            return False, -1, str(e)

    async def get_hierarchy_path(
        self, hierarchy_id: str, entity_id: str
    ) -> Tuple[bool, List[str], Optional[str]]:
        """階層パス取得（ルートから指定エンティティまで）
        
        Args:
            hierarchy_id: 階層ID
            entity_id: エンティティID
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: (成功フラグ, パス, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, [], "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, [], f"階層が見つかりません: {hierarchy_id}"

            if entity_id == hierarchy.root_entity_id:
                return True, [entity_id], None

            success, ancestors, error_msg = await self.get_ancestors(hierarchy_id, entity_id)
            if not success:
                return False, [], error_msg

            # ルートから指定エンティティまでのパス
            path = [hierarchy.root_entity_id] + ancestors + [entity_id]
            # 重複除去
            unique_path = []
            for item in path:
                if item not in unique_path:
                    unique_path.append(item)

            return True, unique_path, None

        except Exception as e:
            self.logger.error(f"階層パス取得エラー: {e}")
            return False, [], str(e)

    async def find_common_ancestor(
        self, hierarchy_id: str, entity_id1: str, entity_id2: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """共通の祖先を取得
        
        Args:
            hierarchy_id: 階層ID
            entity_id1: エンティティID1
            entity_id2: エンティティID2
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (成功フラグ, 共通祖先ID, エラーメッセージ)
        """
        try:
            # 両方の祖先を取得
            success1, ancestors1, error1 = await self.get_ancestors(hierarchy_id, entity_id1)
            success2, ancestors2, error2 = await self.get_ancestors(hierarchy_id, entity_id2)
            
            if not success1:
                return False, None, error1
            if not success2:
                return False, None, error2

            # 共通の祖先を探す
            ancestors1_set = set(ancestors1)
            ancestors2_set = set(ancestors2)
            
            common_ancestors = ancestors1_set.intersection(ancestors2_set)
            
            if not common_ancestors:
                return True, None, None
            
            # 最も近い共通祖先を選択（ルートに最も遠い祖先）
            # 両方のリストで最後に現れる共通祖先
            closest_ancestor = None
            for ancestor in reversed(ancestors1):
                if ancestor in common_ancestors:
                    closest_ancestor = ancestor
                    break
                    
            return True, closest_ancestor, None

        except Exception as e:
            self.logger.error(f"共通祖先取得エラー: {e}")
            return False, None, str(e)

    async def get_subtree(
        self, hierarchy_id: str, root_id: str
    ) -> Tuple[bool, Dict[str, List[str]], Optional[str]]:
        """サブツリー取得
        
        Args:
            hierarchy_id: 階層ID
            root_id: サブツリーのルートエンティティID
            
        Returns:
            Tuple[bool, Dict[str, List[str]], Optional[str]]: (成功フラグ, サブツリー構造, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, {}, "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, {}, f"階層が見つかりません: {hierarchy_id}"

            subtree = {}
            visited = set()

            def build_subtree(current_id: str):
                if current_id in visited:
                    return
                visited.add(current_id)
                
                children = hierarchy.get_children(current_id)
                subtree[current_id] = children
                
                for child_id in children:
                    build_subtree(child_id)

            build_subtree(root_id)
            return True, subtree, None

        except Exception as e:
            self.logger.error(f"サブツリー取得エラー: {e}")
            return False, {}, str(e)