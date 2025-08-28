#!/usr/bin/env python3
"""
階層管理モジュール

階層管理機能を複数のファイルに分割して保守性を向上させます。
"""

from .hierarchy_operations import HierarchyOperations
from .hierarchy_navigation import HierarchyNavigation
from .hierarchy_validation import HierarchyValidation


class HierarchyManager:
    """階層管理メインクラス
    
    各機能を統合し、単一のインターフェースを提供します。
    """
    
    def __init__(self):
        self.operations = HierarchyOperations()
        self.navigation = HierarchyNavigation()
        self.validation = HierarchyValidation()
        
        # 共通データ参照設定
        self.navigation.set_hierarchies_ref(self.operations._hierarchies)
        self.validation.set_hierarchies_ref(self.operations._hierarchies)
    
    # 基本操作の委譲
    async def create_hierarchy(self, *args, **kwargs):
        return await self.operations.create_hierarchy(*args, **kwargs)
    
    async def delete_hierarchy(self, *args, **kwargs):
        return await self.operations.delete_hierarchy(*args, **kwargs)
    
    async def add_child_relationship(self, *args, **kwargs):
        return await self.operations.add_child_relationship(*args, **kwargs)
    
    async def remove_child_relationship(self, *args, **kwargs):
        return await self.operations.remove_child_relationship(*args, **kwargs)
    
    async def list_hierarchies(self, *args, **kwargs):
        return await self.operations.list_hierarchies(*args, **kwargs)
    
    # ナビゲーション機能の委譲
    async def get_children(self, *args, **kwargs):
        return await self.navigation.get_children(*args, **kwargs)
    
    async def get_descendants(self, *args, **kwargs):
        return await self.navigation.get_descendants(*args, **kwargs)
    
    async def get_ancestors(self, *args, **kwargs):
        return await self.navigation.get_ancestors(*args, **kwargs)
    
    async def get_siblings(self, *args, **kwargs):
        return await self.navigation.get_siblings(*args, **kwargs)
    
    async def get_hierarchy_level(self, *args, **kwargs):
        return await self.navigation.get_hierarchy_level(*args, **kwargs)
    
    async def get_hierarchy_path(self, *args, **kwargs):
        return await self.navigation.get_hierarchy_path(*args, **kwargs)
    
    # 検証機能の委譲
    async def validate_hierarchy(self, *args, **kwargs):
        return await self.validation.validate_hierarchy(*args, **kwargs)
    
    async def get_hierarchy_statistics(self, *args, **kwargs):
        return await self.validation.get_hierarchy_statistics(*args, **kwargs)


__all__ = [
    "HierarchyManager",
    "HierarchyOperations", 
    "HierarchyNavigation",
    "HierarchyValidation"
]