#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - エンティティ管理モジュール

エンティティ管理機能の統合インターフェース
"""

from .entity_crud import EntityCRUD
from .entity_search import EntitySearch
from .entity_statistics import EntityStatistics

__all__ = [
    "EntityCRUD",
    "EntitySearch", 
    "EntityStatistics",
]


class EntityManager:
    """統合エンティティ管理クラス"""
    
    def __init__(self, governance_manager, quality_manager):
        self._entity_cache = {}
        
        # 各コンポーネント初期化
        self.crud = EntityCRUD(governance_manager, quality_manager)
        self.search = EntitySearch(self._entity_cache)
        self.statistics = EntityStatistics(self._entity_cache)
        
        # EntityCRUDのキャッシュを共有
        self.crud._entity_cache = self._entity_cache
    
    # CRUD操作の委譲
    async def create_entity(self, *args, **kwargs):
        return await self.crud.create_entity(*args, **kwargs)
    
    async def get_entity(self, entity_id: str):
        return await self.crud.get_entity(entity_id)
    
    async def update_entity(self, *args, **kwargs):
        return await self.crud.update_entity(*args, **kwargs)
    
    async def delete_entity(self, *args, **kwargs):
        return await self.crud.delete_entity(*args, **kwargs)
    
    async def archive_entity(self, *args, **kwargs):
        return await self.crud.archive_entity(*args, **kwargs)
    
    async def restore_entity(self, *args, **kwargs):
        return await self.crud.restore_entity(*args, **kwargs)
    
    # 検索操作の委譲
    async def list_entities(self, *args, **kwargs):
        return await self.search.list_entities(*args, **kwargs)
    
    async def search_entities(self, *args, **kwargs):
        return await self.search.search_entities(*args, **kwargs)
    
    async def advanced_search(self, *args, **kwargs):
        return await self.search.advanced_search(*args, **kwargs)
    
    async def find_duplicates(self, *args, **kwargs):
        return await self.search.find_duplicates(*args, **kwargs)
    
    async def find_related_entities(self, *args, **kwargs):
        return await self.search.find_related_entities(*args, **kwargs)
    
    # 統計操作の委譲
    async def get_entity_statistics(self):
        return await self.statistics.get_entity_statistics()
    
    async def get_quality_trends(self, *args, **kwargs):
        return await self.statistics.get_quality_trends(*args, **kwargs)
    
    async def get_source_system_analysis(self):
        return await self.statistics.get_source_system_analysis()
    
    async def get_governance_compliance_report(self):
        return await self.statistics.get_governance_compliance_report()
    
    async def get_data_lineage_analysis(self):
        return await self.statistics.get_data_lineage_analysis()
    
    async def generate_health_score(self):
        return await self.statistics.generate_health_score()

    async def bulk_update_entities(self, update_operations, updated_by):
        """一括エンティティ更新"""
        results = {
            "total_operations": len(update_operations),
            "successful": 0,
            "failed": 0,
            "errors": []
        }

        for operation in update_operations:
            entity_id = operation.get("entity_id")
            updated_attributes = operation.get("attributes", {})
            metadata = operation.get("metadata", {})

            success, error_msg = await self.update_entity(
                entity_id, updated_attributes, updated_by, metadata
            )

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "entity_id": entity_id,
                    "error": error_msg
                })

        return results