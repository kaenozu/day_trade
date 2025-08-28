#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - エンティティCRUD操作

エンティティの基本的なCRUD操作を提供
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..enums import DataGovernanceLevel, MasterDataType
from ..models import MasterDataEntity


class EntityCRUD:
    """エンティティCRUD操作クラス"""

    def __init__(self, governance_manager, quality_manager):
        self.logger = logging.getLogger(__name__)
        self.governance_manager = governance_manager
        self.quality_manager = quality_manager
        self._entity_cache: Dict[str, MasterDataEntity] = {}

    async def create_entity(
        self,
        entity_type: MasterDataType,
        primary_key: str,
        attributes: Dict[str, Any],
        source_system: str,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """マスターデータエンティティ作成"""
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # エンティティID生成
            entity_id = self._generate_entity_id(entity_type, primary_key)

            # 重複チェック
            if entity_id in self._entity_cache:
                return False, "", f"エンティティID '{entity_id}' は既に存在します"

            # ガバナンスポリシー取得
            policy = self.governance_manager.get_applicable_policy(entity_type)
            governance_level = (
                policy.governance_level if policy else DataGovernanceLevel.STANDARD
            )

            # データバリデーション
            validation_result = await self.quality_manager.validate_entity_attributes(
                attributes, entity_type, policy
            )
            if not validation_result["is_valid"]:
                error_msg = f"データバリデーションエラー: {', '.join(validation_result['errors'])}"
                return False, "", error_msg

            # エンティティ作成
            entity = MasterDataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=attributes,
                metadata={**metadata, "source_system": source_system},
                governance_level=governance_level,
                created_by=created_by,
                updated_by=created_by,
                source_systems=[source_system],
            )

            # 品質スコア計算
            quality_score, quality_metrics = await self.quality_manager.calculate_entity_quality(
                entity, policy
            )
            entity.quality_score = quality_score

            # ゴールデンレコード判定
            entity.is_golden_record = await self._determine_golden_record_status(
                entity, policy
            )

            # キャッシュに保存
            self._entity_cache[entity_id] = entity

            self.logger.info(f"エンティティ作成完了: {entity_id}")
            return True, entity_id, None

        except Exception as e:
            self.logger.error(f"エンティティ作成エラー: {e}")
            return False, "", str(e)

    async def get_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """エンティティ取得"""
        try:
            return self._entity_cache.get(entity_id)
        except Exception as e:
            self.logger.error(f"エンティティ取得エラー: {e}")
            return None

    async def update_entity(
        self,
        entity_id: str,
        updated_attributes: Dict[str, Any],
        updated_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ更新"""
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # 更新前のバックアップ
            old_attributes = entity.attributes.copy()
            old_version = entity.version

            # 属性更新
            entity.attributes.update(updated_attributes)
            entity.version += 1
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = updated_by

            # メタデータ更新
            if metadata:
                entity.metadata.update(metadata)

            # データバリデーション
            policy = self.governance_manager.get_applicable_policy(entity.entity_type)
            validation_result = await self.quality_manager.validate_entity_attributes(
                entity.attributes, entity.entity_type, policy
            )
            if not validation_result["is_valid"]:
                # バリデーションエラーの場合、元に戻す
                entity.attributes = old_attributes
                entity.version = old_version
                error_msg = f"データバリデーションエラー: {', '.join(validation_result['errors'])}"
                return False, error_msg

            # 品質スコア再計算
            quality_score, quality_metrics = await self.quality_manager.calculate_entity_quality(
                entity, policy
            )
            entity.quality_score = quality_score

            # ゴールデンレコード状態再評価
            entity.is_golden_record = await self._determine_golden_record_status(
                entity, policy
            )

            self.logger.info(f"エンティティ更新完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ更新エラー: {e}")
            return False, str(e)

    async def delete_entity(
        self, entity_id: str, deleted_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ削除（論理削除）"""
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # 論理削除
            entity.is_active = False
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = deleted_by
            entity.metadata["deleted_at"] = entity.updated_at.isoformat()
            entity.metadata["deleted_by"] = deleted_by

            self.logger.info(f"エンティティ削除完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ削除エラー: {e}")
            return False, str(e)

    async def archive_entity(
        self, entity_id: str, archived_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティアーカイブ"""
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # アーカイブ処理
            entity.is_active = False
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = archived_by
            entity.metadata["archived_at"] = entity.updated_at.isoformat()
            entity.metadata["archived_by"] = archived_by
            entity.metadata["archive_reason"] = "Manual archive"

            self.logger.info(f"エンティティアーカイブ完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティアーカイブエラー: {e}")
            return False, str(e)

    async def restore_entity(
        self, entity_id: str, restored_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ復元"""
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            if entity.is_active:
                return False, f"エンティティは既にアクティブです: {entity_id}"

            # 復元処理
            entity.is_active = True
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = restored_by
            entity.metadata["restored_at"] = entity.updated_at.isoformat()
            entity.metadata["restored_by"] = restored_by

            # 削除・アーカイブ関連のメタデータをクリア
            for key in [
                "deleted_at",
                "deleted_by",
                "archived_at",
                "archived_by",
                "archive_reason",
            ]:
                entity.metadata.pop(key, None)

            self.logger.info(f"エンティティ復元完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ復元エラー: {e}")
            return False, str(e)

    def _generate_entity_id(self, entity_type: MasterDataType, primary_key: str) -> str:
        """エンティティID生成"""
        timestamp = int(time.time())
        hash_part = hashlib.md5(primary_key.encode()).hexdigest()[:8]
        return f"{entity_type.value}_{hash_part}_{timestamp}"

    async def _determine_golden_record_status(
        self, entity: MasterDataEntity, policy=None
    ) -> bool:
        """ゴールデンレコード状態判定"""
        try:
            # 品質スコアしきい値
            quality_threshold = 90.0
            if policy:
                quality_threshold = policy.quality_threshold

            # 品質スコアチェック
            if entity.quality_score < quality_threshold:
                return False

            # ソースシステムチェック
            if not entity.source_systems:
                return False

            # エンティティタイプ別の特別条件
            if entity.entity_type == MasterDataType.FINANCIAL_INSTRUMENTS:
                # 金融商品の場合、ISINコードが必須
                if "isin" not in entity.attributes or not entity.attributes["isin"]:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"ゴールデンレコード判定エラー: {e}")
            return False