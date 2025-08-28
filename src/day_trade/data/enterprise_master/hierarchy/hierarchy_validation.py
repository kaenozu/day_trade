#!/usr/bin/env python3
"""
階層検証モジュール

階層の整合性検証、統計情報の生成、循環参照検出などを提供します。
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import MasterDataHierarchy


class HierarchyValidation:
    """階層検証クラス
    
    階層構造の健全性をチェックし、統計情報を提供します。
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
            if not self._hierarchies:
                return False, {}, "階層データが設定されていません"
                
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
                level = await self._calculate_node_level(hierarchy, node_id)
                if level >= 0:
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
            if not self._hierarchies:
                return False, {}, "階層データが設定されていません"
                
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

            # 重複する親子関係チェック
            duplicates = await self._check_duplicate_relationships(hierarchy)
            if duplicates:
                validation_result["warnings"].append("重複する親子関係があります")
                validation_result["duplicate_relationships"] = duplicates

            return True, validation_result, None

        except Exception as e:
            self.logger.error(f"階層検証エラー: {e}")
            return False, {}, str(e)

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

    async def _calculate_node_level(self, hierarchy: MasterDataHierarchy, node_id: str) -> int:
        """ノードレベル計算
        
        Args:
            hierarchy: 階層オブジェクト
            node_id: ノードID
            
        Returns:
            int: ノードレベル（エラーの場合は-1）
        """
        try:
            if node_id == hierarchy.root_entity_id:
                return 0

            # 逆引き辞書作成（子 -> 親）
            child_to_parent = {}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                for child_id in children:
                    child_to_parent[child_id] = parent_id

            # 祖先をたどってレベルを計算
            level = 0
            current_id = node_id
            visited = set()

            while current_id in child_to_parent:
                if current_id in visited:
                    return -1  # 循環参照の可能性
                visited.add(current_id)
                
                parent_id = child_to_parent[current_id]
                level += 1
                current_id = parent_id
                
                if current_id == hierarchy.root_entity_id:
                    return level

            return -1  # ルートに到達できない

        except Exception as e:
            self.logger.error(f"ノードレベル計算エラー: {e}")
            return -1

    async def _check_duplicate_relationships(self, hierarchy: MasterDataHierarchy) -> List[Tuple[str, str]]:
        """重複する親子関係チェック
        
        Args:
            hierarchy: 階層オブジェクト
            
        Returns:
            List[Tuple[str, str]]: 重複する関係のリスト（親ID, 子ID）
        """
        try:
            duplicates = []
            seen_relationships = set()

            for parent_id, children in hierarchy.parent_child_mapping.items():
                for child_id in children:
                    relationship = (parent_id, child_id)
                    if relationship in seen_relationships:
                        duplicates.append(relationship)
                    else:
                        seen_relationships.add(relationship)

            return duplicates

        except Exception as e:
            self.logger.error(f"重複関係チェックエラー: {e}")
            return []

    async def analyze_hierarchy_balance(
        self, hierarchy_id: str
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """階層バランス分析
        
        Args:
            hierarchy_id: 階層ID
            
        Returns:
            Tuple[bool, Dict[str, Any], Optional[str]]: (成功フラグ, バランス分析結果, エラーメッセージ)
        """
        try:
            if not self._hierarchies:
                return False, {}, "階層データが設定されていません"
                
            hierarchy = self._hierarchies.get(hierarchy_id)
            if not hierarchy:
                return False, {}, f"階層が見つかりません: {hierarchy_id}"

            balance_info = {
                "is_balanced": True,
                "branching_factors": {},  # レベル別の分岐係数
                "depth_variance": 0.0,    # 深度のばらつき
                "recommendations": [],
            }

            # 各レベルでの分岐係数を計算
            all_nodes = {hierarchy.root_entity_id}
            for parent_id, children in hierarchy.parent_child_mapping.items():
                all_nodes.add(parent_id)
                all_nodes.update(children)

            level_nodes = {}
            for node_id in all_nodes:
                level = await self._calculate_node_level(hierarchy, node_id)
                if level >= 0:
                    if level not in level_nodes:
                        level_nodes[level] = []
                    level_nodes[level].append(node_id)

            # 各レベルでの分岐係数計算
            for level, nodes in level_nodes.items():
                if level == max(level_nodes.keys()):  # 最下位レベルは除外
                    continue
                    
                total_children = 0
                internal_nodes = 0
                
                for node_id in nodes:
                    children = hierarchy.get_children(node_id)
                    if children:
                        total_children += len(children)
                        internal_nodes += 1
                
                if internal_nodes > 0:
                    avg_branching = total_children / internal_nodes
                    balance_info["branching_factors"][level] = {
                        "average": avg_branching,
                        "total_children": total_children,
                        "internal_nodes": internal_nodes
                    }
                    
                    if avg_branching > 10:
                        balance_info["is_balanced"] = False
                        balance_info["recommendations"].append(
                            f"レベル{level}で分岐が多すぎます (平均{avg_branching:.1f})"
                        )

            # 深度のばらつき計算
            leaf_depths = []
            for node_id in all_nodes:
                if not hierarchy.get_children(node_id):  # リーフノード
                    level = await self._calculate_node_level(hierarchy, node_id)
                    if level >= 0:
                        leaf_depths.append(level)

            if len(leaf_depths) > 1:
                avg_depth = sum(leaf_depths) / len(leaf_depths)
                variance = sum((d - avg_depth) ** 2 for d in leaf_depths) / len(leaf_depths)
                balance_info["depth_variance"] = variance
                
                if variance > 4.0:  # 閾値は調整可能
                    balance_info["is_balanced"] = False
                    balance_info["recommendations"].append(
                        f"リーフノードの深度にばらつきがあります (分散: {variance:.1f})"
                    )

            return True, balance_info, None

        except Exception as e:
            self.logger.error(f"階層バランス分析エラー: {e}")
            return False, {}, str(e)