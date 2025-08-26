#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - エンティティ検索

エンティティの検索と一覧表示機能を提供
"""

import logging
from typing import Any, Dict, List, Optional

from ..enums import MasterDataType
from ..models import MasterDataEntity


class EntitySearch:
    """エンティティ検索クラス"""

    def __init__(self, entity_cache: Dict[str, MasterDataEntity]):
        self.logger = logging.getLogger(__name__)
        self._entity_cache = entity_cache

    async def list_entities(
        self,
        entity_type: Optional[MasterDataType] = None,
        is_active: Optional[bool] = None,
        is_golden_record: Optional[bool] = None,
        min_quality_score: Optional[float] = None,
    ) -> List[MasterDataEntity]:
        """エンティティ一覧取得"""
        try:
            entities = []

            for entity in self._entity_cache.values():
                # フィルタ適用
                if entity_type and entity.entity_type != entity_type:
                    continue

                if is_active is not None and entity.is_active != is_active:
                    continue

                if (
                    is_golden_record is not None
                    and entity.is_golden_record != is_golden_record
                ):
                    continue

                if (
                    min_quality_score is not None
                    and entity.quality_score < min_quality_score
                ):
                    continue

                entities.append(entity)

            # 更新日時の降順でソート
            entities.sort(key=lambda e: e.updated_at, reverse=True)
            return entities

        except Exception as e:
            self.logger.error(f"エンティティ一覧取得エラー: {e}")
            return []

    async def search_entities(
        self,
        search_criteria: Dict[str, Any],
        entity_type: Optional[MasterDataType] = None,
    ) -> List[MasterDataEntity]:
        """エンティティ検索"""
        try:
            matching_entities = []

            for entity in self._entity_cache.values():
                if entity_type and entity.entity_type != entity_type:
                    continue

                # 検索条件チェック
                matches = True
                for attr_name, search_value in search_criteria.items():
                    entity_value = entity.attributes.get(attr_name)

                    if entity_value is None:
                        matches = False
                        break

                    # 部分一致検索（文字列の場合）
                    if isinstance(search_value, str) and isinstance(entity_value, str):
                        if search_value.lower() not in entity_value.lower():
                            matches = False
                            break
                    # 完全一致検索（その他の型）
                    elif entity_value != search_value:
                        matches = False
                        break

                if matches:
                    matching_entities.append(entity)

            return matching_entities

        except Exception as e:
            self.logger.error(f"エンティティ検索エラー: {e}")
            return []

    async def advanced_search(
        self,
        filters: Dict[str, Any],
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """高度な検索機能"""
        try:
            entities = list(self._entity_cache.values())
            matched_entities = []

            # フィルタ適用
            for entity in entities:
                if self._matches_filters(entity, filters):
                    matched_entities.append(entity)

            # ソート
            reverse_sort = sort_order.lower() == "desc"
            if sort_by == "updated_at":
                matched_entities.sort(key=lambda e: e.updated_at, reverse=reverse_sort)
            elif sort_by == "created_at":
                matched_entities.sort(key=lambda e: e.created_at, reverse=reverse_sort)
            elif sort_by == "quality_score":
                matched_entities.sort(
                    key=lambda e: e.quality_score, reverse=reverse_sort
                )
            elif sort_by == "entity_id":
                matched_entities.sort(key=lambda e: e.entity_id, reverse=reverse_sort)

            # ページネーション
            total_count = len(matched_entities)
            if limit:
                end_index = offset + limit
                matched_entities = matched_entities[offset:end_index]
            else:
                matched_entities = matched_entities[offset:]

            return {
                "entities": matched_entities,
                "total_count": total_count,
                "returned_count": len(matched_entities),
                "offset": offset,
                "limit": limit,
            }

        except Exception as e:
            self.logger.error(f"高度検索エラー: {e}")
            return {
                "entities": [],
                "total_count": 0,
                "returned_count": 0,
                "offset": offset,
                "limit": limit,
                "error": str(e),
            }

    async def find_duplicates(
        self, entity_type: Optional[MasterDataType] = None
    ) -> List[List[MasterDataEntity]]:
        """重複エンティティ検出"""
        try:
            # プライマリキーでグループ化
            primary_key_groups: Dict[str, List[MasterDataEntity]] = {}

            for entity in self._entity_cache.values():
                if entity_type and entity.entity_type != entity_type:
                    continue

                if not entity.is_active:
                    continue

                pk = entity.primary_key
                if pk not in primary_key_groups:
                    primary_key_groups[pk] = []
                primary_key_groups[pk].append(entity)

            # 重複グループを抽出
            duplicates = []
            for entities in primary_key_groups.values():
                if len(entities) > 1:
                    duplicates.append(entities)

            return duplicates

        except Exception as e:
            self.logger.error(f"重複検出エラー: {e}")
            return []

    async def find_related_entities(
        self, entity_id: str, max_depth: int = 2
    ) -> Dict[str, Any]:
        """関連エンティティ検索"""
        try:
            entity = self._entity_cache.get(entity_id)
            if not entity:
                return {"error": f"エンティティが見つかりません: {entity_id}"}

            related_entities = {}
            visited = set()

            def find_relations(current_entity: MasterDataEntity, depth: int):
                if depth > max_depth or current_entity.entity_id in visited:
                    return

                visited.add(current_entity.entity_id)

                for related_id in current_entity.related_entities:
                    related_entity = self._entity_cache.get(related_id)
                    if related_entity:
                        if depth not in related_entities:
                            related_entities[depth] = []
                        related_entities[depth].append(related_entity)
                        find_relations(related_entity, depth + 1)

            find_relations(entity, 1)

            return {
                "source_entity": entity,
                "related_entities": related_entities,
                "total_related": sum(len(entities) for entities in related_entities.values()),
            }

        except Exception as e:
            self.logger.error(f"関連エンティティ検索エラー: {e}")
            return {"error": str(e)}

    def _matches_filters(self, entity: MasterDataEntity, filters: Dict[str, Any]) -> bool:
        """フィルタ条件のマッチング判定"""
        try:
            for filter_key, filter_value in filters.items():
                if filter_key == "entity_type":
                    if entity.entity_type.value != filter_value:
                        return False
                elif filter_key == "is_active":
                    if entity.is_active != filter_value:
                        return False
                elif filter_key == "is_golden_record":
                    if entity.is_golden_record != filter_value:
                        return False
                elif filter_key == "governance_level":
                    if entity.governance_level.value != filter_value:
                        return False
                elif filter_key == "min_quality_score":
                    if entity.quality_score < filter_value:
                        return False
                elif filter_key == "max_quality_score":
                    if entity.quality_score > filter_value:
                        return False
                elif filter_key == "source_systems":
                    if not any(
                        source in entity.source_systems for source in filter_value
                    ):
                        return False
                elif filter_key == "created_after":
                    if entity.created_at < filter_value:
                        return False
                elif filter_key == "created_before":
                    if entity.created_at > filter_value:
                        return False
                elif filter_key == "updated_after":
                    if entity.updated_at < filter_value:
                        return False
                elif filter_key == "updated_before":
                    if entity.updated_at > filter_value:
                        return False
                elif filter_key.startswith("attr_"):
                    # 属性フィルタ
                    attr_name = filter_key[5:]  # "attr_" を除去
                    attr_value = entity.attributes.get(attr_name)
                    if isinstance(filter_value, dict):
                        # 複雑なフィルタ条件（範囲、パターンマッチなど）
                        if not self._matches_complex_filter(attr_value, filter_value):
                            return False
                    else:
                        # 単純な一致
                        if attr_value != filter_value:
                            return False

            return True

        except Exception as e:
            self.logger.error(f"フィルタマッチングエラー: {e}")
            return False

    def _matches_complex_filter(self, value: Any, filter_config: Dict[str, Any]) -> bool:
        """複雑なフィルタ条件のマッチング"""
        try:
            if "equals" in filter_config:
                return value == filter_config["equals"]

            if "contains" in filter_config and isinstance(value, str):
                return filter_config["contains"].lower() in value.lower()

            if "starts_with" in filter_config and isinstance(value, str):
                return value.startswith(filter_config["starts_with"])

            if "ends_with" in filter_config and isinstance(value, str):
                return value.endswith(filter_config["ends_with"])

            if "min" in filter_config:
                return value >= filter_config["min"]

            if "max" in filter_config:
                return value <= filter_config["max"]

            if "in" in filter_config:
                return value in filter_config["in"]

            if "not_in" in filter_config:
                return value not in filter_config["not_in"]

            if "regex" in filter_config and isinstance(value, str):
                import re

                return bool(re.search(filter_config["regex"], value))

            return True

        except Exception as e:
            self.logger.error(f"複雑フィルタマッチングエラー: {e}")
            return False