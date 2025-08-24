#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - 階層管理

このモジュールは、マスターデータ間の階層関係の管理を担当します。
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .enums import HierarchyType, MasterDataType
from .models import MasterDataHierarchy, MasterDataEntity


class HierarchyManager:
    """階層管理クラス
    
    マスターデータ間の階層関係の作成、管理、ナビゲーションを担当します。
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

            # 循環参照チェック
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

    async def get_hierarchy_statistics(
        self, hierarchy_id: str
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """階層統計情報取得
        
        Args:
            hierarchy_id: 階層ID
            
        Returns:
            Tuple[bool, Dict[str, Any], Optional[str]]: (成功フラグ, 統計情報, エラーメッセージ)
        """
        try:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, {}, f"階層が見つかりません: {hierarchy_id}"

            stats = {
                "hierarchy_id": hierarchy_id,
                "name": hierarchy.name,
                "entity_type": hierarchy.entity_type.value,
                "root_entity_id": hierarchy.root_entity_id,
                "total_nodes": 1,  # ルートノード
                "total_relationships": 0,
                "max_depth": 0,
                "level_counts": {},
                "leaf_nodes": [],
                "internal_nodes": [],
            }

            # 全ノードをカウント
            all_nodes = {hierarchy.root_entity_id}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                all_nodes.add(parent_id)
                all_nodes.update(children)
                stats["total_relationships"] += len(children)

            stats["total_nodes"] = len(all_nodes)

            # 各ノードの深度とタイプを計算
            for node_id in all_nodes:
                success, level, _ = await self.get_hierarchy_level(hierarchy_id, node_id)
                if success:
                    stats["max_depth"] = max(stats["max_depth"], level)
                    
                    # レベル別カウント
                    if level not in stats["level_counts"]:
                        stats["level_counts"][level] = 0
                    stats["level_counts"][level] += 1

                    # リーフノード vs 内部ノード
                    children = hierarchy.get_children(node_id)
                    if not children:
                        stats["leaf_nodes"].append(node_id)
                    else:
                        stats["internal_nodes"].append(node_id)

            return True, stats, None

        except Exception as e:
            self.logger.error(f"階層統計情報取得エラー: {e}")
            return False, {}, str(e)

    async def validate_hierarchy(
        self, hierarchy_id: str
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """階層整合性検証
        
        Args:
            hierarchy_id: 階層ID
            
        Returns:
            Tuple[bool, Dict[str, Any], Optional[str]]: (成功フラグ, 検証結果, エラーメッセージ)
        """
        try:
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, {}, f"階層が見つかりません: {hierarchy_id}"

            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "orphaned_nodes": [],
                "cycles": [],
                "unreachable_nodes": [],
            }

            # 循環参照チェック
            cycles = await self._detect_cycles(hierarchy)
            if cycles:
                validation_result["is_valid"] = False
                validation_result["errors"].append("循環参照が検出されました")
                validation_result["cycles"] = cycles

            # 孤立ノードチェック
            reachable_nodes = await self._get_reachable_nodes(hierarchy)
            all_nodes = {hierarchy.root_entity_id}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                all_nodes.add(parent_id)
                all_nodes.update(children)

            unreachable = all_nodes - reachable_nodes
            if unreachable:
                validation_result["warnings"].append("到達不可能なノードがあります")
                validation_result["unreachable_nodes"] = list(unreachable)

            return True, validation_result, None

        except Exception as e:
            self.logger.error(f"階層検証エラー: {e}")
            return False, {}, str(e)

    async def _would_create_cycle(
        self, hierarchy: MasterDataHierarchy, parent_id: str, child_id: str
    ) -> bool:
        """循環参照チェック
        
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

    async def _detect_cycles(self, hierarchy: MasterDataHierarchy) -> List[List[str]]:
        """循環参照検出
        
        Args:
            hierarchy: 階層オブジェクト
            
        Returns:
            List[List[str]]: 検出された循環参照のリスト
        """
        try:
            cycles = []
            visited = set()
            rec_stack = set()

            def dfs_cycle_detection(node_id: str, path: List[str]) -> bool:
                if node_id in rec_stack:
                    # 循環を発見
                    cycle_start = path.index(node_id)
                    cycle = path[cycle_start:] + [node_id]
                    cycles.append(cycle)
                    return True

                if node_id in visited:
                    return False

                visited.add(node_id)
                rec_stack.add(node_id)

                children = hierarchy.get_children(node_id)
                for child_id in children:
                    if dfs_cycle_detection(child_id, path + [child_id]):
                        pass  # 循環検出済み

                rec_stack.remove(node_id)
                return False

            # 全ノードから循環検出を実行
            all_nodes = {hierarchy.root_entity_id}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                all_nodes.add(parent_id)
                all_nodes.update(children)

            for node_id in all_nodes:
                if node_id not in visited:
                    dfs_cycle_detection(node_id, [node_id])

            return cycles

        except Exception as e:
            self.logger.error(f"循環参照検出エラー: {e}")
            return []

    async def _get_reachable_nodes(self, hierarchy: MasterDataHierarchy) -> Set[str]:
        """到達可能ノード取得
        
        Args:
            hierarchy: 階層オブジェクト
            
        Returns:
            Set[str]: 到達可能ノードのセット
        """
        try:
            reachable = set()
            
            def dfs_reachable(node_id: str):
                if node_id in reachable:
                    return
                reachable.add(node_id)
                
                children = hierarchy.get_children(node_id)
                for child_id in children:
                    dfs_reachable(child_id)

            dfs_reachable(hierarchy.root_entity_id)
            return reachable

        except Exception as e:
            self.logger.error(f"到達可能ノード取得エラー: {e}")
            return set()

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