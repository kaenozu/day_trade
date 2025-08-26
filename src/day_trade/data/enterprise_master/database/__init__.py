#!/usr/bin/env python3
"""
データベース操作モジュール

データベース関連の機能を分割して保守性を向上させます。
"""

from .database_schema import DatabaseSchema
from .entity_operations import EntityOperations
from .request_operations import RequestOperations
from .history_tracking import HistoryTracking


class DatabaseOperations:
    """データベース操作メインクラス
    
    各データベース操作機能を統合し、単一のインターフェースを提供します。
    """
    
    def __init__(self, db_path: str = "enterprise_mdm.db"):
        self.db_path = db_path
        self.schema = DatabaseSchema(db_path)
        self.entities = EntityOperations(db_path)
        self.requests = RequestOperations(db_path)
        self.history = HistoryTracking(db_path)
    
    # エンティティ操作の委譲
    async def save_entity(self, *args, **kwargs):
        return await self.entities.save_entity(*args, **kwargs)
    
    async def get_entity(self, *args, **kwargs):
        return await self.entities.get_entity(*args, **kwargs)
    
    async def list_entities(self, *args, **kwargs):
        return await self.entities.list_entities(*args, **kwargs)
    
    # リクエスト操作の委譲
    async def save_change_request(self, *args, **kwargs):
        return await self.requests.save_change_request(*args, **kwargs)
    
    async def get_change_request(self, *args, **kwargs):
        return await self.requests.get_change_request(*args, **kwargs)
    
    # ガバナンス操作の委譲
    async def save_governance_policy(self, *args, **kwargs):
        return await self.requests.save_governance_policy(*args, **kwargs)
    
    async def save_hierarchy(self, *args, **kwargs):
        return await self.requests.save_hierarchy(*args, **kwargs)
    
    # 履歴操作の委譲
    async def record_change_history(self, *args, **kwargs):
        return await self.history.record_change_history(*args, **kwargs)
    
    async def record_quality_history(self, *args, **kwargs):
        return await self.history.record_quality_history(*args, **kwargs)
    
    async def get_entity_lineage(self, *args, **kwargs):
        return await self.history.get_entity_lineage(*args, **kwargs)
    
    # カタログ操作
    async def get_catalog_data(self, *args, **kwargs):
        return await self.entities.get_catalog_data(*args, **kwargs)
    
    # ユーティリティ操作
    async def cleanup_database(self, *args, **kwargs):
        return await self.schema.cleanup_database(*args, **kwargs)


__all__ = [
    "DatabaseOperations",
    "DatabaseSchema",
    "EntityOperations", 
    "RequestOperations",
    "HistoryTracking"
]